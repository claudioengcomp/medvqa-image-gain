from unittest.mock import MagicMock, Mock, patch

import torch

from med_vlm.enums import ModelName
from med_vlm.training.spec_registry import MODEL_SPECS, build_blip2_spec
from med_vlm.training.specs import ModelSpec


def test_build_blip2_spec_returns_model_spec():
    spec = build_blip2_spec()
    assert isinstance(spec, ModelSpec)


def test_build_blip2_spec_default_max_length():
    spec = build_blip2_spec()
    assert spec.default_max_length == 256


def test_build_blip2_spec_custom_max_length():
    spec = build_blip2_spec(max_length=512)
    assert spec.default_max_length == 512


def test_build_blip2_spec_default_max_new_tokens():
    spec = build_blip2_spec()
    assert spec.default_max_new_tokens == 16


def test_build_blip2_spec_has_prompt_fn():
    spec = build_blip2_spec()
    assert spec.prompt_fn is not None


def test_build_blip2_spec_has_build_collator():
    spec = build_blip2_spec()
    assert callable(spec.build_collator)


def test_build_blip2_spec_has_generate_fn():
    spec = build_blip2_spec()
    assert callable(spec.generate_fn)


def test_model_specs_registry_contains_blip():
    assert ModelName.BLIP in MODEL_SPECS


def test_model_specs_blip_is_callable():
    assert callable(MODEL_SPECS[ModelName.BLIP])


def test_model_specs_blip_returns_spec():
    spec = MODEL_SPECS[ModelName.BLIP]()
    assert isinstance(spec, ModelSpec)


def test_build_collator_callable_with_processor():
    spec = build_blip2_spec()
    processor = Mock()
    with patch(
        "med_vlm.training.spec_registry.build_causal_lm_vqa_collator"
    ) as mock_build:
        mock_build.return_value = Mock()
        _result = spec.build_collator(processor)
        mock_build.assert_called_once()


def test_generate_fn_callable_with_required_args_no_patch():
    spec = build_blip2_spec()

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = torch.tensor([[1, 2, 3]])

    processor = MagicMock()
    processor.return_value = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
    }
    processor.tokenizer.batch_decode.return_value = ["Answer: A"]

    examples = [{"image": "img", "question": "Q?", "answers": ["A"]}]

    out = spec.generate_fn(model, processor, examples, max_new_tokens=16)
    assert out == ["A"]
