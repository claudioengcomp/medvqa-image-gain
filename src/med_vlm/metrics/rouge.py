from typing import Dict, List, Union

import evaluate

from med_vlm.metrics.process_text import normalize_text

_rouge = None


def rouge_scores(
    preds: List[str], refs: Union[List[str], List[List[str]]]
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        preds: List of predicted strings.
        refs:  List of reference strings or list of reference lists (only the first
               reference per sample is used for the aggregate score).

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rougeL``.
    """
    if not preds or not refs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if len(preds) != len(refs):
        raise ValueError(
            f"preds and refs must have equal length, got {len(preds)} and {len(refs)}"
        )

    global _rouge
    if _rouge is None:
        _rouge = evaluate.load("rouge")

    preds_norm = [normalize_text(p) for p in preds]

    if isinstance(refs[0], list):
        refs_flat = [normalize_text(r[0]) for r in refs]  # type: ignore[index]
    else:
        refs_flat = [normalize_text(r) for r in refs]  # type: ignore[arg-type]

    result = _rouge.compute(predictions=preds_norm, references=refs_flat)
    return {
        "rouge1": float(result.get("rouge1", 0.0)),
        "rouge2": float(result.get("rouge2", 0.0)),
        "rougeL": float(result.get("rougeL", 0.0)),
    }
