from typing import Dict, List, Union

_MEDICAL_BERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
_FALLBACK_BERT_MODEL = "bert-base-uncased"


def bert_score_metric(
    preds: List[str],
    refs: Union[List[str], List[List[str]]],
) -> Dict[str, float]:
    """Compute BERTScore F1 between predictions and references.

    Uses BiomedBERT for medical-domain accuracy, falling back to
    bert-base-uncased if the model is unavailable.

    Args:
        preds: Predicted strings (already normalized / extracted).
        refs:  Ground-truth strings or list of acceptable answers per sample.
               When multiple references are given, the first is used.

    Returns:
        {"bert_score_f1": float}  — mean F1 across all samples (0.0–1.0).
    """
    from bert_score import score as bs_score  # lazy import — heavy dependency

    flat_refs = [r[0] if isinstance(r, list) else r for r in refs]

    for model_name in (_MEDICAL_BERT_MODEL, _FALLBACK_BERT_MODEL):
        try:
            _, _, f1 = bs_score(
                preds,
                flat_refs,
                model_type=model_name,
                verbose=False,
                device=None,  # auto-select CPU/GPU
            )
            return {"bert_score_f1": float(f1.mean())}
        except Exception:
            continue

    # Both models failed — return 0 so the rest of evaluation still succeeds
    return {"bert_score_f1": 0.0}
