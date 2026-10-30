from unittest.mock import MagicMock, Mock

from med_vlm.training.evaluation import evaluate_vqa, evaluate_vqa_predictions
from med_vlm.training.specs import ModelSpec


def test_evaluate_vqa_predictions_returns_metrics():
    preds = ["cat", "dog"]
    refs = [["cat"], ["dog"]]

    result = evaluate_vqa_predictions(preds, refs)

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "bleu" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["bleu"] <= 1.0


def test_evaluate_vqa_loop_calls_generate_and_uses_answers():
    eval_dl = [
        [
            {"answers": ["cat"], "question": "q1"},
            {"answers": ["dog"], "question": "q2"},
        ],
        [{"answers": ["fish"], "question": "q3"}],
    ]

    def fake_generate(model, processor, examples, *, max_new_tokens: int):
        return ["p"] * len(examples)

    spec = ModelSpec(
        prompt_fn=lambda q: q,
        build_collator=lambda processor: lambda batch: {},
        generate_fn=fake_generate,
        default_max_new_tokens=7,
    )

    model = MagicMock()
    processor = MagicMock()

    result = evaluate_vqa(
        model=model,
        processor=processor,
        spec=spec,
        eval_dl=eval_dl,
        accelerator=None,
        max_new_tokens=None,
    )

    assert "accuracy" in result
    assert "bleu" in result


def test_evaluate_vqa_prepare_model_false_skips_prepare():
    """When prepare_model=False the model should not be re-wrapped."""
    eval_dl = [[{"answers": ["yes"], "question": "q"}]]

    def fake_generate(model, processor, examples, *, max_new_tokens: int):
        return ["yes"] * len(examples)

    spec = ModelSpec(
        prompt_fn=lambda q: q,
        build_collator=lambda processor: lambda batch: {},
        generate_fn=fake_generate,
        default_max_new_tokens=7,
    )

    model = MagicMock()
    processor = MagicMock()
    accelerator = MagicMock()
    accelerator.is_main_process = True

    result = evaluate_vqa(
        model=model,
        processor=processor,
        spec=spec,
        eval_dl=eval_dl,
        accelerator=accelerator,
        max_new_tokens=None,
        prepare_model=False,
    )

    accelerator.prepare.assert_not_called()
    assert "accuracy" in result


def test_evaluate_vqa_non_main_process_returns_zero_placeholders():
    eval_dl = [[{"answers": ["cat"], "question": "q"}]]

    def fake_generate(model, processor, examples, *, max_new_tokens: int):
        return ["cat"] * len(examples)

    spec = ModelSpec(
        prompt_fn=lambda q: q,
        build_collator=lambda processor: lambda batch: {},
        generate_fn=fake_generate,
        default_max_new_tokens=7,
    )

    model = MagicMock()
    processor = MagicMock()
    accelerator = MagicMock()
    accelerator.is_main_process = False
    accelerator.prepare.return_value = model

    result = evaluate_vqa(
        model=model,
        processor=processor,
        spec=spec,
        eval_dl=eval_dl,
        accelerator=accelerator,
        max_new_tokens=None,
    )

    assert result["accuracy"] == 0.0
    assert result["rouge1"] == 0.0
    assert result["rougeL"] == 0.0
    assert result["meteor"] == 0.0
    assert set(result.keys()) == {
        "accuracy",
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "meteor",
        "bert_score_f1",
    }


def test_evaluate_vqa_forwards_max_new_tokens():
    eval_dl = [[{"answers": ["cat"], "question": "q"}]]

    gen = Mock(return_value=["cat"])
    spec = ModelSpec(
        prompt_fn=lambda q: q,
        build_collator=lambda processor: lambda batch: {},
        generate_fn=gen,
        default_max_new_tokens=16,
    )

    model = MagicMock()
    processor = MagicMock()

    evaluate_vqa(
        model=model, processor=processor, spec=spec, eval_dl=eval_dl, max_new_tokens=99
    )

    gen.assert_called_once()
    kwargs = gen.call_args.kwargs
    assert kwargs["max_new_tokens"] == 99
