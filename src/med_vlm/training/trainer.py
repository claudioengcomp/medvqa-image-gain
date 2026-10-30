import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from med_vlm.training.evaluation import evaluate_vqa
from med_vlm.training.specs import ModelSpec
from med_vlm.utils import log_metrics


@dataclass(frozen=True)
class TrainStats:
    """Wall-clock and memory statistics collected during one training run."""

    train_wall_time_s: float
    train_throughput_sps: float
    train_peak_memory_gb: Optional[float]


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    epochs: int = 1
    grad_accum_steps: int = 16
    max_grad_norm: float = 1.0
    log_every: int = 10
    max_steps: int | None = None
    eval_every: int | None = None
    checkpoint_every: int | None = None
    compile: bool = True
    compile_backend: str = "eager"
    start_step: int = 0
    seed: int = 0
    # Total number of OPTIMIZER steps for the scheduler (cosine + warmup), independent of
    # max_steps. Needed on resumption (16/09/2026): max_steps counts micro-batches and stops the
    # loop, but was also used as the scheduler total, recreating the cosine over a total
    # ~grad_accum times larger than the original training's (lr returned to the peak after the resume).
    # None = comportamento antigo.
    scheduler_total_steps: int | None = None

    def __post_init__(self) -> None:
        if self.eval_every is not None and self.eval_every <= 0:
            raise ValueError(
                f"eval_every must be a positive integer, got {self.eval_every}"
            )
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError(
                "checkpoint_every must be a positive integer, "
                f"got {self.checkpoint_every}"
            )


class runningAverage:
    def __init__(self):
        self.running_loss = 0.0
        self.running_count = 0

    def update(self, loss_val: float):
        self.running_loss += loss_val
        self.running_count += 1

    def average(self) -> float:
        return self.running_loss / max(self.running_count, 1)

    def reset(self):
        self.running_loss = 0.0
        self.running_count = 0


def should_log(global_step: int, log_every: int) -> bool:
    return global_step % log_every == 0


def log_train_metrics(
    *,
    accelerator: Accelerator,
    writer: Any | None,
    step: int,
    loss: float,
    lr: float,
) -> None:
    if not accelerator.is_main_process:
        return

    log_metrics(
        metrics={"loss": loss, "lr": lr},
        step=step,
        writer=writer,
        prefix="train",
    )


def create_accelerator(cfg: TrainConfig) -> Accelerator:
    return Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        mixed_precision="bf16",
    )


def create_optimizer(
    model: Any,
    cfg: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer],
) -> torch.optim.Optimizer:
    return optimizer or AdamW(model.parameters(), lr=cfg.lr)


def resolve_total_optimizer_steps(cfg: TrainConfig, steps_per_epoch: int | None) -> int:
    """Total number of optimizer steps the scheduler should cover.

    Priority: explicit `scheduler_total_steps` (resumes) > `max_steps` (historical
    behavior, which treats max_steps as optimizer steps) > ceil(steps_per_epoch /
    grad_accum) * epochs > 1000 (dataloader without len)."""
    if cfg.scheduler_total_steps is not None:
        return int(cfg.scheduler_total_steps)
    if cfg.max_steps is not None:
        return int(cfg.max_steps)
    if steps_per_epoch is None:
        return 1000
    return math.ceil(steps_per_epoch / cfg.grad_accum_steps) * cfg.epochs


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_optimizer_steps: int,
) -> Any:
    from transformers import get_cosine_schedule_with_warmup

    warmup_steps = max(1, int(total_optimizer_steps * cfg.warmup_ratio))
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )


def prepare_training(
    accelerator: Accelerator,
    model: Any,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
) -> Tuple[Any, torch.optim.Optimizer, DataLoader]:
    return accelerator.prepare(model, optimizer, train_dl)


def train_step(
    *,
    accelerator: Accelerator,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    max_grad_norm: float | None,
    scheduler: Any | None = None,
) -> float:
    """
    Runs one optimizer step (respects accelerate accumulation) and returns the loss
    value (float).  Returns ``float('nan')`` and skips the backward/optimizer step
    when the loss is non-finite (NaN or Inf) to prevent model weight corruption.
    """
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            # Clear any gradients that accumulated in previous sub-steps so the
            # next accumulation window starts clean.
            model.zero_grad()
            return float(loss.detach().float().item())

        accelerator.backward(loss)

        if max_grad_norm is not None and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return float(loss.detach().float().item())


def train_vqa_generation(
    *,
    model: Any,
    train_dl: DataLoader,
    cfg: TrainConfig,
    writer: Any | None = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eval_dl: Any | None = None,
    processor: Any | None = None,
    spec: ModelSpec | None = None,
    eval_max_new_tokens: int | None = None,
    save_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
) -> TrainStats:
    if cfg.eval_every is not None and any(
        dep is None for dep in (eval_dl, processor, spec)
    ):
        raise ValueError(
            "eval_every is set but eval_dl, processor, and spec must all be provided"
        )

    accelerator = create_accelerator(cfg)
    optimizer = create_optimizer(model, cfg, optimizer)

    try:
        steps_per_epoch = len(train_dl)
    except TypeError:
        steps_per_epoch = None
    total_optimizer_steps = resolve_total_optimizer_steps(cfg, steps_per_epoch)

    scheduler = create_scheduler(optimizer, cfg, total_optimizer_steps)

    if cfg.compile:
        model = torch.compile(model, dynamic=True, backend=cfg.compile_backend)

    model, optimizer, train_dl = prepare_training(
        accelerator, model, optimizer, train_dl
    )
    scheduler = accelerator.prepare(scheduler)

    if resume_from_checkpoint is not None:
        accelerator.load_state(str(resume_from_checkpoint))

    model.train()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    global_step = cfg.start_step
    total_samples = 0
    loss_meter = runningAverage()

    def _batches():
        # Flattens the two loops (epoch x batch) into a single sequence, so that
        # a "break" when max_steps is reached ends the training for
        # real — and not just the inner loop, leaving the outer one
        # continue to the next epoch silently.
        for _epoch in range(cfg.epochs):
            for batch in train_dl:
                yield batch

    for batch in _batches():
        loss_val = train_step(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            batch=batch,
            max_grad_norm=cfg.max_grad_norm,
            scheduler=scheduler,
        )

        global_step += 1
        if isinstance(batch, dict):
            if "input_ids" in batch and hasattr(batch["input_ids"], "shape"):
                batch_size = int(batch["input_ids"].shape[0])
            else:
                batch_size = next(
                    (
                        int(value.shape[0])
                        for value in batch.values()
                        if hasattr(value, "shape")
                    ),
                    1,
                )
        else:
            batch_size = len(batch) if hasattr(batch, "__len__") else 1
        total_samples += batch_size
        loss_meter.update(loss_val)

        if should_log(global_step, cfg.log_every):
            avg_loss = loss_meter.average()

            log_train_metrics(
                accelerator=accelerator,
                writer=writer,
                step=global_step,
                loss=avg_loss,
                lr=optimizer.param_groups[0]["lr"],
            )

            loss_meter.reset()

        if (
            cfg.checkpoint_every is not None
            and global_step % cfg.checkpoint_every == 0
            and checkpoint_dir is not None
        ):
            ckpt_path = checkpoint_dir / f"step_{global_step}"
            accelerator.save_state(str(ckpt_path))

        if cfg.eval_every is not None and global_step % cfg.eval_every == 0:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            eval_metrics = evaluate_vqa(
                model=unwrapped,
                processor=processor,
                spec=spec,
                eval_dl=eval_dl,
                accelerator=accelerator,
                max_new_tokens=eval_max_new_tokens,
                prepare_model=False,
            )
            if accelerator.is_main_process:
                log_metrics(
                    metrics=eval_metrics,
                    step=global_step,
                    writer=writer,
                    prefix="eval",
                )
            accelerator.wait_for_everyone()
            model.train()

        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            # Stops training and epoch at once: without this, a
            # break aqui so sairia da malha interna e a malha externa
            # (epochs) continuaria rodando silenciosamente.
            break

    if use_cuda:
        torch.cuda.synchronize()

    wall_time = time.perf_counter() - t_start
    throughput = total_samples / wall_time if wall_time > 0 else 0.0
    peak_memory_gb: Optional[float] = (
        torch.cuda.max_memory_allocated() / (1024**3) if use_cuda else None
    )

    if save_dir is not None and accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
        if processor is not None:
            processor.save_pretrained(save_dir)

    return TrainStats(
        train_wall_time_s=wall_time,
        train_throughput_sps=throughput,
        train_peak_memory_gb=peak_memory_gb,
    )
