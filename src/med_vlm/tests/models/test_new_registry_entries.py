from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from med_vlm.enums import ModelName
from med_vlm.models.registry import MODEL_REGISTRY


def test_registry_has_granite_vision():
    assert ModelName.GRANITE_VISION in MODEL_REGISTRY


def test_registry_has_qwen2vl():
    assert ModelName.QWEN2VL in MODEL_REGISTRY


def test_registry_has_smolvlm():
    assert ModelName.SMOLVLM in MODEL_REGISTRY


def test_registry_has_llava_med():
    assert ModelName.LLAVA_MED in MODEL_REGISTRY


def test_registry_has_pubmedclip():
    assert ModelName.PUBMEDCLIP in MODEL_REGISTRY


def test_granite_vision_uses_auto_model_for_vision_seq():
    config = MODEL_REGISTRY[ModelName.GRANITE_VISION]
    assert config.model_cls == AutoModelForImageTextToText
    assert config.processor_cls == AutoProcessor


def test_qwen2vl_uses_qwen_class():
    config = MODEL_REGISTRY[ModelName.QWEN2VL]
    assert config.model_cls == Qwen2VLForConditionalGeneration
