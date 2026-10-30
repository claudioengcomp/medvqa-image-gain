from typing import Dict, List, Union

import evaluate

from med_vlm.metrics.process_text import normalize_text

_meteor = None


def meteor_score(
    preds: List[str], refs: Union[List[str], List[List[str]]]
) -> Dict[str, float]:
    """Compute METEOR score.

    Args:
        preds: List of predicted strings.
        refs:  List of reference strings or list of reference lists (only the first
               reference per sample is used).

    Returns:
        Dict with key ``meteor``.
    """
    if not preds or not refs:
        return {"meteor": 0.0}
    if len(preds) != len(refs):
        raise ValueError(
            f"preds and refs must have equal length, got {len(preds)} and {len(refs)}"
        )

    global _meteor
    if _meteor is None:
        _meteor = evaluate.load("meteor")

    preds_norm = [normalize_text(p) for p in preds]

    if isinstance(refs[0], list):
        refs_flat = [normalize_text(r[0]) for r in refs]  # type: ignore[index]
    else:
        refs_flat = [normalize_text(r) for r in refs]  # type: ignore[arg-type]

    result = _meteor.compute(predictions=preds_norm, references=refs_flat)
    return {"meteor": float(result.get("meteor", 0.0))}
