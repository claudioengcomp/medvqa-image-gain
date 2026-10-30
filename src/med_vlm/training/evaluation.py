from typing import Any, Dict, List

from accelerate import Accelerator

from med_vlm.metrics.accuracy import accuracy
from med_vlm.metrics.bertscore import bert_score_metric
from med_vlm.metrics.bleu import bleu_score
from med_vlm.metrics.meteor import meteor_score
from med_vlm.metrics.process_text import extract_answer
from med_vlm.metrics.rouge import rouge_scores
from med_vlm.training.specs import ModelSpec


def evaluate_vqa_predictions(
    preds: List[str], refs: List[List[str]], questions: List[str] | None = None
) -> Dict[str, float]:
    """
    preds:     list[str] model outputs (raw)
    refs:      list[list[str]] acceptable answers per example
    questions: list[str] original questions; used for answer extraction.
               When provided, predictions are pre-processed with extract_answer()
               before all string-matching metrics are computed.
    """
    if questions is not None:
        extracted = [extract_answer(p, q) for p, q in zip(preds, questions)]
    else:
        extracted = preds

    acc = accuracy(extracted, refs)
    bleu = bleu_score(extracted, refs)
    rouge = rouge_scores(extracted, refs)
    meteor = meteor_score(extracted, refs)
    bert = bert_score_metric(extracted, refs)

    return {"accuracy": acc, **bleu, **rouge, **meteor, **bert}


def evaluate_vqa(
    *,
    model: Any,
    processor: Any,
    spec: ModelSpec,
    eval_dl,
    accelerator: Accelerator | None = None,
    max_new_tokens: int | None = None,
    prepare_model: bool = True,
) -> Dict[str, float]:
    """
    eval_dl should yield list[example_dict] (raw examples).

    Set prepare_model=False when the model is already on the correct device and
    should not be passed through accelerator.prepare() again (e.g. when called
    mid-training with an unwrapped-but-already-placed model).
    """
    accelerator = accelerator or Accelerator()
    if prepare_model:
        model = accelerator.prepare(model)

    preds_all: List[str] = []
    refs_all: List[List[str]] = []
    questions_all: List[str] = []

    max_new_tokens = max_new_tokens or spec.default_max_new_tokens

    for examples in eval_dl:
        preds = spec.generate_fn(
            model, processor, examples, max_new_tokens=max_new_tokens
        )

        preds_all.extend(preds)
        refs_all.extend([x["answers"] for x in examples])
        questions_all.extend([x["question"] for x in examples])

    if not accelerator.is_main_process:
        return {
            "accuracy": 0.0,
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "meteor": 0.0,
            "bert_score_f1": 0.0,
        }

    # preds_all = accelerator.gather_for_metrics(preds_all)
    # refs_all = accelerator.gather_for_metrics(refs_all)

    return evaluate_vqa_predictions(preds_all, refs_all, questions_all)
