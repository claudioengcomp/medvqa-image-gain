from dataclasses import dataclass
from typing import Any, Optional

from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    SmolVLMForConditionalGeneration,
)

import med_vlm.enums


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Optional[Any] = None
    processor: Optional[Any] = None


@dataclass(frozen=True)
class ModelConfig:
    model_cls: Any
    tokenizer_cls: Optional[Any] = None
    processor_cls: Optional[Any] = None


MODEL_REGISTRY: dict[med_vlm.enums.ModelName, ModelConfig] = {
    med_vlm.enums.ModelName.MEDCLIP: ModelConfig(
        model_cls=AutoModel,
        tokenizer_cls=AutoTokenizer,
    ),
    med_vlm.enums.ModelName.BIOMEDCLIP: ModelConfig(
        model_cls=AutoModel,
        tokenizer_cls=AutoTokenizer,
    ),
    med_vlm.enums.ModelName.BLIP: ModelConfig(
        model_cls=Blip2ForConditionalGeneration,
        processor_cls=Blip2Processor,
    ),
    med_vlm.enums.ModelName.MEDVLM: ModelConfig(
        model_cls=Qwen2VLForConditionalGeneration,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.GRANITE_VISION: ModelConfig(
        model_cls=AutoModelForImageTextToText,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.QWEN2VL: ModelConfig(
        model_cls=Qwen2VLForConditionalGeneration,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.SMOLVLM: ModelConfig(
        model_cls=SmolVLMForConditionalGeneration,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.MEDVLTHINKER: ModelConfig(
        model_cls=Qwen2_5_VLForConditionalGeneration,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.QWEN25VL: ModelConfig(
        model_cls=Qwen2_5_VLForConditionalGeneration,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.MEDGEMMA: ModelConfig(
        model_cls=AutoModelForImageTextToText,
        processor_cls=AutoProcessor,
    ),
    # v7 (2026-10-13): zero-shot novos
    med_vlm.enums.ModelName.LINGSHU: ModelConfig(model_cls=Qwen2_5_VLForConditionalGeneration, processor_cls=AutoProcessor),
    med_vlm.enums.ModelName.HUATUO_QWEN25: ModelConfig(model_cls=Qwen2_5_VLForConditionalGeneration, processor_cls=AutoProcessor),
    med_vlm.enums.ModelName.INTERNVL3: ModelConfig(model_cls=AutoModelForImageTextToText, processor_cls=AutoProcessor),
    med_vlm.enums.ModelName.GEMMA3: ModelConfig(model_cls=AutoModelForImageTextToText, processor_cls=AutoProcessor),   # like MedGemma (Gemma 3 family)
    med_vlm.enums.ModelName.LLAVA_MED: ModelConfig(
        model_cls=AutoModelForImageTextToText,
        processor_cls=AutoProcessor,
    ),
    med_vlm.enums.ModelName.PUBMEDCLIP: ModelConfig(
        model_cls=AutoModel,
        processor_cls=AutoProcessor,
    ),
}
