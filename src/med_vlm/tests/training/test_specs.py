import pytest
import torch

from med_vlm.training.specs import ModelSpec


def dummy_prompt_fn(text: str) -> str:
    return f"Prompt: {text}"


def dummy_build_collator(model):
    def collate_fn(batch):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    return collate_fn


def dummy_generate_fn(model, tokenizer, batch, max_tokens):
    return ["generated text"]


def test_model_spec_initialization():
    spec = ModelSpec(
        prompt_fn=dummy_prompt_fn,
        build_collator=dummy_build_collator,
        generate_fn=dummy_generate_fn,
    )
    assert spec.prompt_fn is not None
    assert spec.build_collator is not None
    assert spec.generate_fn is not None
    assert spec.default_max_length == 256
    assert spec.default_max_new_tokens == 16


def test_model_spec_custom_defaults():
    spec = ModelSpec(
        prompt_fn=dummy_prompt_fn,
        build_collator=dummy_build_collator,
        generate_fn=dummy_generate_fn,
        default_max_length=512,
        default_max_new_tokens=32,
    )
    assert spec.default_max_length == 512
    assert spec.default_max_new_tokens == 32


def test_model_spec_frozen():
    spec = ModelSpec(
        prompt_fn=dummy_prompt_fn,
        build_collator=dummy_build_collator,
        generate_fn=dummy_generate_fn,
    )
    with pytest.raises(AttributeError):
        spec.default_max_length = 128


def test_prompt_fn_callable():
    spec = ModelSpec(
        prompt_fn=dummy_prompt_fn,
        build_collator=dummy_build_collator,
        generate_fn=dummy_generate_fn,
    )
    result = spec.prompt_fn("test")
    assert result == "Prompt: test"


def test_build_collator_callable():
    spec = ModelSpec(
        prompt_fn=dummy_prompt_fn,
        build_collator=dummy_build_collator,
        generate_fn=dummy_generate_fn,
    )
    collator = spec.build_collator(None)
    batch = [{"text": "sample"}]
    result = collator(batch)
    assert "input_ids" in result


def test_generate_fn_callable():
    spec = ModelSpec(
        prompt_fn=dummy_prompt_fn,
        build_collator=dummy_build_collator,
        generate_fn=dummy_generate_fn,
    )
    result = spec.generate_fn(None, None, [], 16)
    assert isinstance(result, list)
    assert len(result) > 0
