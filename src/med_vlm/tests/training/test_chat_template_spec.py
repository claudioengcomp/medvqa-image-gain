from med_vlm.enums import ModelName
from med_vlm.training.spec_registry import MODEL_SPECS, build_chat_template_spec
from med_vlm.training.specs import ModelSpec


def test_build_chat_template_spec_returns_model_spec():
    spec = build_chat_template_spec()
    assert isinstance(spec, ModelSpec)


def test_build_chat_template_spec_default_max_length():
    spec = build_chat_template_spec()
    assert spec.default_max_length == 2048


def test_build_chat_template_spec_custom_max_length():
    spec = build_chat_template_spec(max_length=1024)
    assert spec.default_max_length == 1024


def test_build_chat_template_spec_default_max_new_tokens():
    spec = build_chat_template_spec()
    assert spec.default_max_new_tokens == 64


def test_model_specs_contains_granite_vision():
    assert ModelName.GRANITE_VISION in MODEL_SPECS


def test_model_specs_contains_qwen2vl():
    assert ModelName.QWEN2VL in MODEL_SPECS


def test_model_specs_contains_smolvlm():
    assert ModelName.SMOLVLM in MODEL_SPECS


def test_model_specs_contains_llava_med():
    assert ModelName.LLAVA_MED in MODEL_SPECS


def test_chat_template_spec_callable():
    assert callable(MODEL_SPECS[ModelName.GRANITE_VISION])
