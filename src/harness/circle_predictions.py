"""Pure per-item scoring logic for the generative pipeline (Qwen2-VL,
SmolVLM2 etc. via `med_vlm`) -- reuses the REAL functions from
`med_vlm.metrics.process_text` (extract_answer, normalize_text), without
reimplementing anything, but isolated from the parts of `med_vlm` that require
torch/transformers (models/loader.py, training/*) -- the __init__.py files of
`med_vlm` and `med_vlm.metrics` are empty, so importing only
`med_vlm.metrics.process_text` does not drag in any heavy dependency,
allowing this module to be tested locally without the heavy stack installed.

See docs/superpowers/specs/2026-08-18-circle-bridge-design.md §4."""
from __future__ import annotations

from med_vlm.metrics.process_text import answer_matches, extract_answer, normalize_text

from harness.results_schema import validate_results_dict


def score_prediction(
    raw_prediction: str, question: str, gold_answers: list[str]
) -> tuple[str, bool]:
    """Extracts the central answer from `raw_prediction` (removes reasoning
    blocks, 'the answer is:' marker, truncates open answer) and
    checks whether it matches any of the `gold_answers`, normalized -- the exact
    same logic used internally by `med_vlm.metrics.accuracy.accuracy`
    (extract_answer applied first, then normalize_text), applied
    here per item instead of in batch, to record `predicted`/`correct` per
    question."""
    extracted = extract_answer(raw_prediction, question)
    extracted_norm = normalize_text(extracted)
    gold_norms = [normalize_text(a) for a in gold_answers]
    correct = answer_matches(extracted_norm, gold_norms)
    return extracted, correct


def build_results_record(
    model: str, dataset: str, condition: str, predictions: list[dict]
) -> dict:
    """Builds the complete results record (harness.results_schema) from
    a list of already-scored per-item predictions ({"qid", "predicted",
    "gold", "correct"}), computing accuracy/n/n_correct directly
    from them -- same format recorded by harness/extract_predictions_m3ae.py,
    inherits the same statistical apparatus (LDV, bootstrap, paired McNemar)."""
    n = len(predictions)
    n_correct = sum(1 for p in predictions if p["correct"])
    record = {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "accuracy": (n_correct / n) if n else 0.0,
        "n": n,
        "n_correct": n_correct,
        "predictions": predictions,
    }
    validate_results_dict(record)
    return record
