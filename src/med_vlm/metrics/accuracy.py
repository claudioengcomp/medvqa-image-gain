from typing import List, Union

from med_vlm.metrics.process_text import answer_matches, normalize_text

Refs = Union[List[str], List[List[str]]]


def accuracy(preds: List[str], refs: Refs) -> float:
    assert len(preds) == len(refs)
    if not preds:
        return 0.0

    correct = 0
    for p, r in zip(preds, refs):
        p_norm = normalize_text(p)

        if isinstance(r, list):
            ref_norms = [normalize_text(x) for x in r]
            if answer_matches(p_norm, ref_norms):
                correct += 1
        else:
            if answer_matches(p_norm, [normalize_text(r)]):
                correct += 1

    return correct / len(preds)
