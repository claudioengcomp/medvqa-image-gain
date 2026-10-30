from typing import Dict, List, Union

import sacrebleu

from med_vlm.metrics.process_text import normalize_text


def bleu_score(
    preds: List[str], refs: Union[List[str], List[List[str]]]
) -> Dict[str, float]:
    assert len(preds) == len(refs)

    preds_norm = [normalize_text(p) for p in preds]
    if len(preds_norm) == 0:
        return {"bleu": 0.0}

    if isinstance(refs[0], list):
        max_refs = max(len(r) for r in refs)
        ref_streams: List[List[str]] = []
        for j in range(max_refs):
            stream = []
            for r in refs:
                stream.append(
                    normalize_text(r[j]) if j < len(r) else normalize_text(r[0])
                )
            ref_streams.append(stream)
    else:
        ref_streams = [[normalize_text(r) for r in refs]]

    score = sacrebleu.corpus_bleu(
        preds_norm,
        ref_streams,
        smooth_method="exp",
    )
    return {"bleu": score.score / 100.0}
