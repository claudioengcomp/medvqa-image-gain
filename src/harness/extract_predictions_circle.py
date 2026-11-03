"""Extract per-item predictions from the generative pipeline (med_vlm --
Qwen2-VL, SmolVLM2 etc.) for ONE condition (C0/C1/C2/C4_*) of any dataset.
Loads the PEFT adapter (LoRA/DoRA) already trained via
`python -m med_vlm.run_experiment` and runs real text generation over the
condition's `.arrow` file (through harness/circle_data_bridge.py).
Writes a complete results record (harness/results_schema.py, including
'predictions') ready for paired McNemar/bootstrap analysis via
harness.stats.compute_vdd.compute_summary_vs_baseline -- the same role played by
harness/extract_predictions_m3ae.py, but for the generative paradigm.

`--adapter-dir` is optional: when omitted, the base checkpoint is run
directly (zero-shot, no PeftModel.from_pretrained). This is the case for
MedVLThinker-7B, which is evaluated zero-shot because it is already
instruction-tuned for medical VQA and is not fine-tuned in this study.

`--progress-checkpoint` is optional: when supplied, every prediction is
persisted incrementally to JSONL (harness/inference_checkpoint.py) and
the run resumes from there if the process dies midway -- worth supplying on
any battery that runs for hours. Do not confuse it with `--adapter-dir`, which points at MODEL
weights -- this one tracks INFERENCE progress. extract_predictions_m3ae.py
already uses `--checkpoint` for model weights (a Sacred .ckpt), which is why
the flag here is named `--progress-checkpoint`: no collision between the two.

This script only runs in an environment with `med_vlm` and a GPU stack
installed, so it is NOT covered by the local pytest suite (no GPU/torch on a
development laptop). The pure data-translation and scoring logic is isolated
in harness/circle_data_bridge.py and harness/circle_predictions.py,
which are tested locally.

Usage (cwd = repository root, with med_vlm/ already installed):
    python -m harness.inference.extract_predictions_circle \
        --model QWEN2VL --peft LORA --dataset SLAKE --condition C0 \
        --condition-arrow path/to/finetune_arrows/vqa_slack_test.arrow \
        --adapter-dir outputs/models/qwen2vl_lora_slake \
        --output results/QWEN2VL_LORA_SLAKE_C0.json

    # zero-shot (no adapter):
    python -m harness.inference.extract_predictions_circle \
        --model MEDVLTHINKER --peft ZEROSHOT --dataset SLAKE --condition C0 \
        --condition-arrow path/to/finetune_arrows/vqa_slack_test.arrow \
        --output results/MEDVLTHINKER_ZEROSHOT_SLAKE_C0.json

    # with a progress checkpoint (resumes on its own if the run dies midway):
    python -m harness.inference.extract_predictions_circle \
        --model QWEN2VL --peft LORA --dataset KVASIR --condition C0 \
        --condition-arrow path/to/finetune_arrows/vqa_kvasir_test.arrow \
        --adapter-dir outputs/models/qwen2vl_lora_kvasir \
        --progress-checkpoint results/.progress/QWEN2VL_LORA_KVASIR_C0.jsonl \
        --output results/QWEN2VL_LORA_KVASIR_C0.json
"""
from __future__ import annotations

import argparse
import json

from harness.circle_data_bridge import load_condition_dataset
from harness.circle_predictions import build_results_record, score_prediction
from harness.inference_checkpoint import append_checkpoint, load_checkpoint


def _collate_raw(batch: list[dict]) -> list[dict]:
    return batch


def extract_predictions(
    model_name_str: str,
    adapter_dir: str | None,
    condition_arrow: str,
    batch_size: int = 4,
    max_new_tokens: int | None = None,
    progress_checkpoint: str | None = None,
    no_image: bool = False,
    instruction: str | None = None,
    instruction_mode: str = "system",
    save_raw: bool = False,   # 14/10: also records the raw generated text and the question (disabled by default; records identical to before)
) -> list[dict]:
    """Load the base model (plus the trained PEFT adapter, if `adapter_dir`
    is given) and run generation over every question in `condition_arrow`,
    returning the list of per-item predictions
    ({"qid", "predicted", "gold", "correct"}). `adapter_dir=None` runs the
    base checkpoint directly, without PEFT (zero-shot evaluation).

    `progress_checkpoint`, when supplied, enables incremental persistence:
    items already resolved in a previous run (same file) are skipped, and
    every newly processed item is appended to the file as soon as its batch
    completes -- a crash midway loses at most the in-flight batch, not the
    whole condition. `progress_checkpoint=None` (the default) preserves the
    behaviour that predated this parameter, with no persistence at all."""
    import torch
    from torch.utils.data import DataLoader

    from med_vlm.enums import ModelName
    from med_vlm.models.loader import load_model_bundle
    from med_vlm.training.spec_registry import MODEL_SPECS

    model_name = ModelName[model_name_str]
    bundle = load_model_bundle(model_name)
    if adapter_dir is not None:
        from peft import PeftModel
        bundle.model = PeftModel.from_pretrained(bundle.model, adapter_dir)
    bundle.model.eval()
    if torch.cuda.is_available():
        bundle.model.to("cuda")

    # short instruction (v7_zeroshot, 10/10/2026): system prompt (templates that accept it) or question prefix (LLaVA-Med)
    if instruction and instruction_mode == "system":
        spec = MODEL_SPECS[model_name](no_image=no_image, system_prompt=instruction)
    elif instruction and instruction_mode == "system_append":   # MedVLThinker _instr: VISAPP prompt + short instruction
        from med_vlm.training.spec_registry import _MEDVLTHINKER_SYSTEM_PROMPT
        spec = MODEL_SPECS[model_name](no_image=no_image, system_prompt=_MEDVLTHINKER_SYSTEM_PROMPT + " " + instruction.strip())
    else:
        spec = MODEL_SPECS[model_name](no_image=no_image)
    prefix = (instruction.strip() + " ") if (instruction and instruction_mode == "prefix") else ""
    max_new_tokens = max_new_tokens or spec.default_max_new_tokens

    eval_ds = load_condition_dataset(condition_arrow)

    predictions, done_qids = load_checkpoint(progress_checkpoint)
    pending_examples = eval_ds
    if done_qids:
        # Do NOT use `filter_pending(list(eval_ds), ...)` here: `list(eval_ds)`
        # forces the `datasets.Image()` feature to decode ALL images in the
        # arrow at once (including the already-finished ones, which are
        # discarded right afterwards) and holds all of it in a single
        # list[dict] in host RAM -- a real leak observed in production:
        # processes resumed from a progress-checkpoint reached 66-82GB of RES
        # each, against ~1GB for those starting from scratch, and took the
        # whole host down (RAM 251GB/251GB, swap 100%, SIGKILL on concurrent
        # processes). `eval_ds["id"]` touches only the id column (cheap, no
        # image decoding) and `.select()` returns a lazy view of the
        # HuggingFace Dataset -- the same lazy behaviour as the no-checkpoint
        # path (`pending_examples = eval_ds`), just with the already-finished
        # indices removed before the DataLoader starts iterating.
        keep_indices = [i for i, qid in enumerate(eval_ds["id"]) if int(qid) not in done_qids]
        pending_examples = eval_ds.select(keep_indices)
        print(f"[progress-checkpoint] resuming from {progress_checkpoint}: "
              f"{len(done_qids)} already done, {len(pending_examples)} remaining")

    eval_dl = DataLoader(
        pending_examples, batch_size=batch_size, shuffle=False, collate_fn=_collate_raw
    )

    with torch.no_grad():
        for batch in eval_dl:
            if not batch:
                continue
            gen_batch = [{**ex, "question": prefix + ex["question"]} for ex in batch] if prefix else batch
            raw_preds = spec.generate_fn(
                bundle.model, bundle.processor, gen_batch, max_new_tokens=max_new_tokens
            )
            batch_predictions = []
            for example, raw_pred in zip(batch, raw_preds):
                extracted, correct = score_prediction(
                    raw_pred, example["question"], example["answers"]
                )
                rec = {
                    "qid": int(example["id"]),
                    "predicted": extracted,
                    "gold": str(example["answers"][0]) if example["answers"] else "",
                    "correct": bool(correct),
                }
                if save_raw: rec.update(raw=raw_pred, question=example["question"])
                batch_predictions.append(rec)
            predictions.extend(batch_predictions)
            append_checkpoint(progress_checkpoint, batch_predictions)
    return predictions


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         help="Model key of the underlying training framework, e.g.: QWEN2VL")
    parser.add_argument("--peft", required=True,
                         help="Fine-tuning method key, only used in the prefix of "
                              "the results record. One of LORA, DORA or "
                              "ZEROSHOT.")
    parser.add_argument("--dataset", required=True,
                         help="Human-readable name written to the results record, e.g.: SLAKE")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--condition-arrow", required=True)
    parser.add_argument("--adapter-dir", required=False, default=None,
                         help="Directory of the PEFT adapter (LoRA/DoRA). Omit for "
                              "zero-shot evaluation of the base checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--progress-checkpoint", required=False, default=None,
                         help="JSONL path used to persist/resume INFERENCE "
                              "progress (not model weights -- see "
                              "harness/inference_checkpoint.py). Omitting it keeps "
                              "the checkpoint-free behaviour.")
    parser.add_argument("--no-image", action="store_true", default=False,
                         help="v7_txt (2026-10-09): text-only arm -- the question goes without "
                              "{'type': 'image'} and the processor receives images=None; the image "
                              "from the arrow is ignored (only questions, answers and qids are read).")
    parser.add_argument("--instruction", default=None, help="short answer instruction (zero-shot); identical in C0 and TXT")
    parser.add_argument("--instruction-mode", default="system", choices=["system", "prefix", "system_append"],
                         help="system = system message of the template; prefix = prefix of the question text (templates without a system role)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-raw", action="store_true", default=False, help="14/10: records the raw generated text and the question in each record")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predictions = extract_predictions(
        args.model, args.adapter_dir, args.condition_arrow,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        progress_checkpoint=args.progress_checkpoint,
        no_image=args.no_image,
        instruction=args.instruction, instruction_mode=args.instruction_mode, save_raw=args.save_raw,
    )
    record = build_results_record(
        f"{args.model}_{args.peft}", args.dataset, args.condition, predictions,
    )
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2, ensure_ascii=False)
    print(f"{record['model']}/{args.dataset}/{args.condition}: "
          f"accuracy={record['accuracy']:.4f} "
          f"({record['n_correct']}/{record['n']}) -> {args.output}")
