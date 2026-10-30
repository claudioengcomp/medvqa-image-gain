import pytest
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Qwen2VLForConditionalGeneration,
)

import med_vlm.enums
from med_vlm.models.registry import MODEL_REGISTRY, ModelBundle, ModelConfig


def test_model_bundle_creation():
    bundle = ModelBundle(model="test_model")
    assert bundle.model == "test_model"
    assert bundle.tokenizer is None
    assert bundle.processor is None


def test_model_bundle_with_all_fields():
    bundle = ModelBundle(model="model", tokenizer="tokenizer", processor="processor")
    assert bundle.model == "model"
    assert bundle.tokenizer == "tokenizer"
    assert bundle.processor == "processor"


def test_model_config_creation():
    config = ModelConfig(model_cls=AutoModel, tokenizer_cls=AutoTokenizer)
    assert config.model_cls == AutoModel
    assert config.tokenizer_cls == AutoTokenizer
    assert config.processor_cls is None


def test_model_config_frozen():
    config = ModelConfig(model_cls=AutoModel)
    with pytest.raises(AttributeError):
        config.model_cls = AutoTokenizer


def test_model_registry_has_all_entries():
    assert med_vlm.enums.ModelName.MEDCLIP in MODEL_REGISTRY
    assert med_vlm.enums.ModelName.BIOMEDCLIP in MODEL_REGISTRY
    assert med_vlm.enums.ModelName.BLIP in MODEL_REGISTRY
    assert med_vlm.enums.ModelName.MEDVLM in MODEL_REGISTRY


def test_medclip_registry_config():
    config = MODEL_REGISTRY[med_vlm.enums.ModelName.MEDCLIP]
    assert config.model_cls == AutoModel
    assert config.tokenizer_cls == AutoTokenizer
    assert config.processor_cls is None


def test_blip_registry_config():
    config = MODEL_REGISTRY[med_vlm.enums.ModelName.BLIP]
    assert config.model_cls == Blip2ForConditionalGeneration
    assert config.processor_cls == Blip2Processor
    assert config.tokenizer_cls is None


def test_medvlm_registry_config():
    config = MODEL_REGISTRY[med_vlm.enums.ModelName.MEDVLM]
    assert config.model_cls == Qwen2VLForConditionalGeneration
    assert config.processor_cls == AutoProcessor
    assert config.tokenizer_cls is None
