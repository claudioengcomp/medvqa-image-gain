"""Tests for med_vlm.metrics.efficiency."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from med_vlm.metrics.efficiency import (
    measure_flops,
    measure_inference_efficiency,
    measure_model_params,
)
from med_vlm.training.specs import ModelSpec

# ── Helpers ───────────────────────────────────────────────────────────────────


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)  # 8 params, all trainable
        self.frozen = nn.Linear(4, 2, bias=False)  # 8 params, frozen
        self.frozen.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + self.frozen(x)


# ── measure_model_params ──────────────────────────────────────────────────────


def test_measure_model_params_counts_total():
    model = TinyModel()
    result = measure_model_params(model)
    assert result["total_params"] == 16  # 8 + 8


def test_measure_model_params_counts_trainable():
    model = TinyModel()
    result = measure_model_params(model)
    assert result["trainable_params"] == 8  # only linear is trainable


def test_measure_model_params_ratio():
    model = TinyModel()
    result = measure_model_params(model)
    assert result["trainable_ratio"] == 0.5


def test_measure_model_params_all_frozen():
    model = TinyModel()
    model.requires_grad_(False)
    result = measure_model_params(model)
    assert result["trainable_params"] == 0
    assert result["trainable_ratio"] == 0.0


def test_measure_model_params_empty_model():
    model = nn.Module()
    result = measure_model_params(model)
    assert result["total_params"] == 0
    assert result["trainable_params"] == 0
    assert result["trainable_ratio"] == 0.0


# ── measure_flops ─────────────────────────────────────────────────────────────


def test_measure_flops_returns_none_on_failure():
    """When calflops is unavailable or raises, return None values gracefully."""
    model = TinyModel()
    sample_batch = {"x": torch.zeros(1, 4)}

    with patch.dict("sys.modules", {"calflops": None}):
        result = measure_flops(model, sample_batch)

    assert result["flops"] is None
    assert result["macs"] is None


def test_measure_flops_returns_none_on_calflops_exception():
    """calflops raising any exception produces None values."""
    model = TinyModel()
    sample_batch = {"x": torch.zeros(1, 4)}

    mock_calflops = MagicMock()
    mock_calflops.calculate_flops.side_effect = RuntimeError("unsupported model")

    with patch.dict("sys.modules", {"calflops": mock_calflops}):
        result = measure_flops(model, sample_batch)

    assert result["flops"] is None
    assert result["macs"] is None


def test_measure_flops_returns_floats_when_calflops_succeeds():
    """When calflops succeeds, values are floats."""
    model = TinyModel()
    sample_batch = {"x": torch.zeros(1, 4)}

    mock_calflops = MagicMock()
    mock_calflops.calculate_flops.return_value = (1024.0, 512.0, "ignored")

    with patch.dict("sys.modules", {"calflops": mock_calflops}):
        result = measure_flops(model, sample_batch)

    assert result["flops"] == 1024.0
    assert result["macs"] == 512.0


def test_measure_flops_excludes_labels_and_max_new_tokens():
    """labels and max_new_tokens must be stripped before calling calflops."""
    model = TinyModel()
    sample_batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.zeros(1, 4, dtype=torch.long),
        "max_new_tokens": 32,
    }

    mock_calflops = MagicMock()
    mock_calflops.calculate_flops.return_value = (2048.0, 1024.0, "ignored")

    with patch.dict("sys.modules", {"calflops": mock_calflops}):
        result = measure_flops(model, sample_batch)

    assert result["flops"] == 2048.0
    assert result["macs"] == 1024.0
    forwarded_kwargs = mock_calflops.calculate_flops.call_args.kwargs["kwargs"]
    assert "labels" not in forwarded_kwargs
    assert "max_new_tokens" not in forwarded_kwargs
    assert "input_ids" in forwarded_kwargs
    assert "attention_mask" in forwarded_kwargs


# ── measure_inference_efficiency ─────────────────────────────────────────────


def _make_spec(n_preds_per_batch: int = 2) -> ModelSpec:
    def fake_generate(model, processor, examples, *, max_new_tokens: int):
        return ["pred"] * len(examples)

    return ModelSpec(
        prompt_fn=lambda q: q,
        build_collator=lambda p: lambda batch: {},
        generate_fn=fake_generate,
    )


def test_measure_inference_efficiency_returns_expected_keys():
    eval_dl = [
        [{"question": "q1", "answers": ["a1"]}, {"question": "q2", "answers": ["a2"]}],
        [{"question": "q3", "answers": ["a3"]}],
    ]
    spec = _make_spec()
    result = measure_inference_efficiency(
        model=MagicMock(),
        processor=MagicMock(),
        spec=spec,
        eval_dl=eval_dl,
        max_new_tokens=8,
    )

    assert set(result.keys()) == {
        "infer_total_s",
        "infer_latency_s",
        "infer_throughput_sps",
        "infer_peak_memory_gb",
    }


def test_measure_inference_efficiency_timing_is_positive():
    eval_dl = [
        [{"question": "q", "answers": ["a"]}],
    ]
    spec = _make_spec()
    result = measure_inference_efficiency(
        model=MagicMock(),
        processor=MagicMock(),
        spec=spec,
        eval_dl=eval_dl,
        max_new_tokens=4,
    )

    assert result["infer_total_s"] >= 0.0
    assert result["infer_latency_s"] >= 0.0
    assert result["infer_throughput_sps"] >= 0.0


def test_measure_inference_efficiency_throughput_consistent():
    """throughput = n_samples / total_time (roughly)."""
    eval_dl = [
        [{"question": "q1", "answers": ["a1"]}, {"question": "q2", "answers": ["a2"]}],
    ]
    spec = _make_spec()
    result = measure_inference_efficiency(
        model=MagicMock(),
        processor=MagicMock(),
        spec=spec,
        eval_dl=eval_dl,
        max_new_tokens=4,
    )

    total_s = result["infer_total_s"]
    throughput = result["infer_throughput_sps"]
    if total_s > 0:
        expected = 2 / total_s
        assert abs(throughput - expected) < 1e-6


def test_measure_inference_efficiency_empty_dl():
    """Empty dataloader: zero samples → latency 0, throughput 0."""
    spec = _make_spec()
    result = measure_inference_efficiency(
        model=MagicMock(),
        processor=MagicMock(),
        spec=spec,
        eval_dl=[],
        max_new_tokens=4,
    )
    # With no samples, throughput must be 0 (0 / time = 0).
    # Latency is total_s / max(0,1) = total_s, which is near-zero but not
    # guaranteed to be exactly 0.0 due to loop overhead.
    assert result["infer_throughput_sps"] == 0.0
    assert result["infer_latency_s"] >= 0.0
