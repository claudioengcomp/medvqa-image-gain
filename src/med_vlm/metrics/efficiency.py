"""Computational efficiency metrics.

Provides three functions:
- measure_model_params   — counts total and trainable parameters.
- measure_flops          — estimates FLOPs/MACs via calflops for one forward pass.
- measure_inference_efficiency — times and measures peak GPU memory for inference.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional

import torch

from med_vlm.training.specs import ModelSpec

# Keys that are NOT valid model.forward() arguments and must be excluded from
# FLOPs measurement kwargs (they belong to generate() or the training loop).
_FLOPS_EXCLUDED_KEYS: frozenset[str] = frozenset({"labels", "max_new_tokens"})


def measure_model_params(model: Any) -> Dict[str, Any]:
    """Return total, trainable parameter counts and trainable ratio.

    Args:
        model: A ``torch.nn.Module`` (possibly PEFT-wrapped).

    Returns:
        Dict with keys ``total_params``, ``trainable_params``, ``trainable_ratio``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": ratio,
    }


def measure_flops(
    model: Any,
    sample_batch: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Estimate FLOPs and MACs for a single forward pass using calflops.

    Only the prefill / encoder forward pass is measured (``forward_mode='forward'``).
    Keys that are not valid ``model.forward()`` arguments — namely ``labels`` and
    ``max_new_tokens`` — are stripped from the batch before the measurement so that:

    * no loss computation is triggered (which would fail with gradient checkpointing
      enabled but no actual gradients flowing);
    * no ``TypeError`` is raised for kwargs that belong to ``generate()``.

    calflops calls ``model.eval()`` internally; the model is restored to training
    mode in the ``finally`` block so that the subsequent training loop is unaffected.

    Args:
        model: ``torch.nn.Module`` (possibly PEFT-wrapped).
        sample_batch: A single-sample batch dict (same format as the training
                      collator output).

    Returns:
        Dict with keys ``flops`` and ``macs`` (floats or ``None`` on failure).
    """
    try:
        from calflops import calculate_flops  # type: ignore[import]

        forward_kwargs = {
            k: v for k, v in sample_batch.items() if k not in _FLOPS_EXCLUDED_KEYS
        }

        _had_gc = getattr(model, "is_gradient_checkpointing", False)
        try:
            # Gradient checkpointing is not needed for a FLOPs count — disabling
            # it reduces peak memory by removing the recomputation overhead that
            # torch.compile inserts via tag_activation_checkpoint.
            if _had_gc and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

            with torch.inference_mode():
                flops, macs, _ = calculate_flops(
                    model=model,
                    kwargs=forward_kwargs,
                    print_results=False,
                    output_as_string=False,
                )
        finally:
            # calflops calls model.eval(); restore training mode so the training
            # loop that follows is not affected.
            if _had_gc and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            model.train()
            # Release CUDA cache so tensors allocated during the calflops forward
            # pass do not fragment the memory pool before training begins.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"flops": float(flops), "macs": float(macs)}
    except Exception as exc:
        warnings.warn(
            f"measure_flops: FLOPs/MACs measurement failed and will be recorded as "
            f"null. Reason: {exc}",
            stacklevel=2,
        )
        return {"flops": None, "macs": None}


def measure_inference_efficiency(
    model: Any,
    processor: Any,
    spec: ModelSpec,
    eval_dl: Any,
    max_new_tokens: int,
) -> Dict[str, Optional[float]]:
    """Time and measure peak GPU memory for the full inference pass over eval_dl.

    This mirrors the generation loop inside ``evaluate_vqa`` but only tracks
    wall-clock time and GPU memory — it does **not** re-compute quality metrics.

    Args:
        model: Model already placed on the target device (not wrapped by Accelerator).
        processor: Processor/tokenizer for the model.
        spec: :class:`~med_vlm.training.specs.ModelSpec` providing ``generate_fn``.
        eval_dl: DataLoader that yields ``list[dict]`` raw examples.
        max_new_tokens: Forwarded to the generation function.

    Returns:
        Dict with keys:
        - ``infer_total_s``       — total wall-clock seconds.
        - ``infer_latency_s``     — seconds per sample (total / n_samples).
        - ``infer_throughput_sps``— samples per second.
        - ``infer_peak_memory_gb``— peak GPU memory in GB.
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    n_samples = 0
    t_start = time.perf_counter()

    for examples in eval_dl:
        spec.generate_fn(model, processor, examples, max_new_tokens=max_new_tokens)
        n_samples += len(examples)

    if use_cuda:
        torch.cuda.synchronize()

    total_s = time.perf_counter() - t_start
    latency_s = total_s / max(n_samples, 1)
    throughput_sps = n_samples / total_s if total_s > 0 else 0.0

    peak_memory_gb: Optional[float] = None
    if use_cuda:
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "infer_total_s": total_s,
        "infer_latency_s": latency_s,
        "infer_throughput_sps": throughput_sps,
        "infer_peak_memory_gb": peak_memory_gb,
    }
