from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from med_vlm.data.augmentations import build_training_augmentor
from med_vlm.data.loader import load_dataset_by_enum
from med_vlm.data.preprocessor import ImagePreprocessor
from med_vlm.enums import DatasetName, ModelName, PEFTMethod
from med_vlm.metrics.efficiency import (
    measure_flops,
    measure_inference_efficiency,
    measure_model_params,
)
from med_vlm.models.loader import load_model_bundle
from med_vlm.models.peft_adapters import apply_peft_adapter
from med_vlm.models.registry import ModelBundle
from med_vlm.seeding import set_seed
from med_vlm.training.evaluation import evaluate_vqa
from med_vlm.training.spec_registry import MODEL_SPECS
from med_vlm.training.trainer import TrainConfig, train_vqa_generation
from med_vlm.utils import adalora_compute_total_steps


def collate_raw(batch: list[dict]) -> list[dict]:
    """Eval dataloader collate: keep raw examples for generation."""
    return batch


def _select_n(ds, n: Optional[int], seed: int = 42):
    if n is None:
        return ds
    n = min(int(n), len(ds))
    return ds.shuffle(seed=seed).select(range(n))


def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_name = ModelName[cfg["model"]]
    dataset_name = DatasetName[cfg["dataset"]]
    peft_method = PEFTMethod[cfg.get("peft", "none")]
    zero_shot = bool(cfg.get("zero_shot", False))

    train_cfg_dict = cfg.get("train", {})
    eval_cfg_dict = cfg.get("eval", {})
    logging_cfg = cfg.get("logging", {})
    # data.no_image (v7_txt, 2026-10-09): text-only arm. Same question, same
    # chat template, without {"type": "image"} and processor(images=None); the
    # pre-processing and the augmentation (image-only) have no effect.
    no_image = bool(cfg.get("data", {}).get("no_image", False))

    outputs_dir = Path(cfg.get("outputs_dir", "./outputs"))
    _ensure_dir(outputs_dir)
    # results_json_dir (v7_txt): where to write the efficiency JSON (default: outputs_dir).
    results_json_dir = Path(cfg.get("results_json_dir", outputs_dir))
    _ensure_dir(results_json_dir)

    train_batch_size = int(train_cfg_dict.get("batch_size", 1))
    eval_batch_size = int(eval_cfg_dict.get("batch_size", 4))

    train_cfg = TrainConfig(
        lr=float(train_cfg_dict.get("lr", 1e-4)),
        warmup_ratio=float(train_cfg_dict.get("warmup_ratio", 0.05)),
        epochs=int(train_cfg_dict.get("epochs", 1)),
        grad_accum_steps=int(train_cfg_dict.get("grad_accum_steps", 16)),
        max_grad_norm=float(train_cfg_dict.get("max_grad_norm", 1.0)),
        log_every=int(train_cfg_dict.get("log_every", 10)),
        max_steps=train_cfg_dict.get("max_steps", None),
        eval_every=(
            int(train_cfg_dict["eval_every"])
            if train_cfg_dict.get("eval_every") is not None
            else None
        ),
        checkpoint_every=(
            int(train_cfg_dict["checkpoint_every"])
            if train_cfg_dict.get("checkpoint_every") is not None
            else None
        ),
        compile=bool(train_cfg_dict.get("compile", False)),
        compile_backend=str(train_cfg_dict.get("compile_backend", "eager")),
        start_step=int(train_cfg_dict.get("start_step", 0)),
        seed=int(train_cfg_dict.get("seed", 0)),
        scheduler_total_steps=(
            int(train_cfg_dict["scheduler_total_steps"])
            if train_cfg_dict.get("scheduler_total_steps") is not None
            else None
        ),
    )
    set_seed(train_cfg.seed)

    if train_cfg.compile and bool(train_cfg_dict.get("gradient_checkpointing", True)):
        warnings.warn(
            "Both 'compile' and 'gradient_checkpointing' are enabled. "
            "This is supported but may increase initial compilation time due to "
            "graph breaks introduced by activation recomputation.",
            stacklevel=2,
        )

    max_length = int(train_cfg_dict.get("max_length", 256))

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0 and local_rank == 0

    wandb_project = logging_cfg.get("wandb_project", None)
    wandb_run_name = logging_cfg.get("wandb_run_name", cfg.get("run_name", None))
    if wandb_project and not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"
    writer = (
        wandb.init(project=wandb_project, name=wandb_run_name)
        if wandb_project and is_main_process
        else None
    )

    train_split = train_cfg_dict.get("split", "train")
    eval_split = eval_cfg_dict.get("split", "validation")

    preprocessing_cfg = cfg.get("preprocessing", {})
    preprocessor = (
        ImagePreprocessor.from_config(preprocessing_cfg) if preprocessing_cfg else None
    )

    augmentation_cfg = cfg.get("augmentation", {})
    augment_fn = (
        build_training_augmentor(augmentation_cfg) if augmentation_cfg else None
    )

    if no_image:
        print("[no_image] text-decoder-only branch: no {'type': 'image'} in the message, "
              "processor(images=None); preprocessing/augmentation without effect", flush=True)
    train_ds = load_dataset_by_enum(
        dataset_name, split=train_split, preprocessor=preprocessor, drop_image=no_image,
        arrow_paths=cfg.get("data", {}).get("arrows"),
    )
    eval_ds = load_dataset_by_enum(
        dataset_name, split=eval_split, preprocessor=preprocessor, drop_image=no_image,
        arrow_paths=cfg.get("data", {}).get("arrows"),
    )

    train_num = train_cfg_dict.get("num_samples", None)
    eval_num = eval_cfg_dict.get("num_samples", None)

    train_ds = _select_n(train_ds, train_num, seed=42)
    eval_ds = _select_n(eval_ds, eval_num, seed=123)

    clip_encoder_cfg = cfg.get("clip_encoder", None)
    if clip_encoder_cfg:
        from med_granite_vision.model import build_med_granite_vision

        clip_id = clip_encoder_cfg["id"]
        clip_out_dim = int(clip_encoder_cfg["out_dim"])
        encoder_type = str(clip_encoder_cfg.get("encoder_type", "clip"))
        alignment_loss_weight = float(
            clip_encoder_cfg.get("alignment_loss_weight", 0.0)
        )
        model_obj, granite_processor, _ = build_med_granite_vision(
            encoder_id=clip_id,
            encoder_out_dim=clip_out_dim,
            encoder_type=encoder_type,
            alignment_loss_weight=alignment_loss_weight,
        )
        bundle = ModelBundle(model=model_obj, processor=granite_processor)
        # Reuse the Granite Vision spec for collation / generation
        model_name = ModelName.GRANITE_VISION
    else:
        bundle = load_model_bundle(model_name)
        alignment_loss_weight = 0.0

    build_spec = MODEL_SPECS[model_name]
    spec = build_spec(max_length=max_length, augment_fn=augment_fn, no_image=no_image)

    train_collate = spec.build_collator(bundle.processor)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collate,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_raw,
    )

    if hasattr(bundle.model, "gradient_checkpointing_enable"):
        if bool(train_cfg_dict.get("gradient_checkpointing", True)):
            # use_reentrant=False: recomputation runs without autograd tracking,
            # so intermediate activations (e.g. 4 GiB attention matrices at
            # max_length=8192) are freed immediately instead of being accumulated
            # for the backward pass.  This is also PEFT's recommended mode.
            bundle.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    if peft_method != PEFTMethod.NONE:
        peft_cfg = cfg.get("peft_config", {}) or {}

        if peft_method == PEFTMethod.ADALORA:
            total_steps = adalora_compute_total_steps(
                num_examples=len(train_ds),
                batch_size=train_batch_size,
                grad_accum=train_cfg.grad_accum_steps,
                epochs=train_cfg.epochs,
            )
            peft_cfg = {
                **peft_cfg,
                "total_step": int(peft_cfg.get("total_step", total_steps)),
            }

        bundle.model = apply_peft_adapter(
            bundle.model,
            peft_method=peft_method,
            peft_config=peft_cfg,
        )

    if hasattr(bundle.model, "enable_input_require_grads"):
        bundle.model.enable_input_require_grads()

    efficiency_params = measure_model_params(bundle.model)
    print(f"[params] {efficiency_params}", flush=True)

    max_new_tokens = int(
        eval_cfg_dict.get("max_new_tokens", spec.default_max_new_tokens)
    )

    zero_shot = bool(cfg.get("zero_shot", False))
    train_stats = None
    if not zero_shot:
        tag_for_save = (
            cfg.get("run_name")
            or f"{model_name.name.lower()}_{peft_method.value}_\
                {dataset_name.name.lower()}_{_now_tag()}"
        )
        model_save_dir = outputs_dir / "models" / tag_for_save
        # `train.checkpoint_dir` (optional) relocates the intermediate
        # accelerator states; absent, the historical default is kept.
        checkpoint_dir = Path(
            train_cfg_dict.get("checkpoint_dir", "checkpoints")
        ) / tag_for_save
        resume_from_checkpoint = (
            Path(train_cfg_dict["resume_from_checkpoint"])
            if train_cfg_dict.get("resume_from_checkpoint") is not None
            else None
        )
        # When alignment loss is enabled, wrap the model so train_vqa_generation
        # receives a model whose forward() automatically adds the SigLIP alignment
        # loss to the task loss.  The wrapper is transparent to the rest of the loop.
        train_model = bundle.model
        if alignment_loss_weight > 0.0:
            from med_granite_vision.model import AlignmentAugmentedModel

            train_model = AlignmentAugmentedModel(bundle.model)
        train_stats = train_vqa_generation(
            model=train_model,
            train_dl=train_dl,
            cfg=train_cfg,
            writer=writer,
            eval_dl=eval_dl,
            processor=bundle.processor,
            spec=spec,
            eval_max_new_tokens=max_new_tokens,
            save_dir=model_save_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    metrics = evaluate_vqa(
        model=bundle.model,
        processor=bundle.processor,
        spec=spec,
        eval_dl=eval_dl,
        max_new_tokens=max_new_tokens,
    )

    efficiency_inference = measure_inference_efficiency(
        model=bundle.model,
        processor=bundle.processor,
        spec=spec,
        eval_dl=eval_dl,
        max_new_tokens=max_new_tokens,
    )

    # Flush inference tensors before FLOPs profiling so the generation loop's
    # activations don't inflate peak memory during the calflops forward pass.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure FLOPs after training so the forward pass does not fragment GPU
    # memory before the training loop starts.  FLOPs depend only on model
    # architecture (not weights), so the result is identical to measuring before
    # training.
    try:
        _sample_for_flops = next(
            iter(
                DataLoader(
                    eval_ds.select(range(1)), batch_size=1, collate_fn=collate_raw
                )
            )
        )
        flops_collated = spec.build_collator(bundle.processor)([_sample_for_flops[0]])
        efficiency_flops = measure_flops(
            model=bundle.model,
            sample_batch=flops_collated,
        )
    except Exception:
        efficiency_flops = {"flops": None, "macs": None}

    result = {
        "model": model_name.value,
        "dataset": dataset_name.value,
        "peft": peft_method.value,
        "clip_encoder": clip_encoder_cfg["id"] if clip_encoder_cfg else None,
        "no_image": no_image,
        "train": asdict(train_cfg),
        "train_max_length": max_length,
        "eval": {
            "split": eval_split,
            "num_samples": eval_num,
            "max_new_tokens": max_new_tokens,
        },
        "metrics": metrics,
        "efficiency": {
            "params": efficiency_params,
            "flops": efficiency_flops,
            "training": (
                {
                    "train_wall_time_s": train_stats.train_wall_time_s,
                    "train_throughput_sps": train_stats.train_throughput_sps,
                    "train_peak_memory_gb": train_stats.train_peak_memory_gb,
                }
                if train_stats is not None
                else None
            ),
            "inference": efficiency_inference,
        },
        "wandb_run_id": writer.id if writer else None,
    }

    tag = (
        cfg.get("run_name")
        or f"{model_name.name.lower()}_{peft_method.value}_{dataset_name.name.lower()}\
            _{_now_tag()}"
    )
    out_path = results_json_dir / f"{tag}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if writer:
        writer.log(
            {
                f"eval/{k}": float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
        )
        wandb.finish()

    return {"result_path": str(out_path), **result}


def _apply_seed_override(cfg: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    """Overrides train.seed in the config and suffixes `_seedN` to run_name/
    wandb_run_name, to run multiple seeds of the same YAML without colliding
    outputs/checkpoints/results. `seed=None` is a no-op (returns `cfg`
    unchanged), to allow normal CLI usage without `--seed`."""
    if seed is None:
        return cfg
    cfg = dict(cfg)
    cfg["train"] = {**cfg.get("train", {}), "seed": seed}
    suffix = f"_seed{seed}"
    if cfg.get("run_name"):
        cfg["run_name"] = f"{cfg['run_name']}{suffix}"
    if cfg.get("logging", {}).get("wandb_run_name"):
        cfg["logging"] = {
            **cfg["logging"],
            "wandb_run_name": f"{cfg['logging']['wandb_run_name']}{suffix}",
        }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overrides train.seed of the config and appends _seedN to run_name/"
        "wandb_run_name, to run multiple seeds of the same YAML without "
        "clashing outputs/checkpoints/results.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_seed_override(cfg, args.seed)

    out = run_experiment(cfg)
    print(f"Saved results to: {out['result_path']}")
    print("Metrics:", out["metrics"])


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    main()


# TO RUN:
# CUDA_VISIBLE_DEVICES=3 poetry run python -m med_vlm.run_experiment \
#   --config configs/blip_lora_slake.yaml

# ALSO, set WANDB_API_KEY or run `wandb login` before launching experiments.
