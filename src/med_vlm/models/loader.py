"""Model loader and PEFT wrapper helpers.

This module provides a small, typed scaffold for loading CLIP-like / VLM models
and attaching PEFT adapters (LoRA / AdaLoRA / DoRA) using the `peft` library.
"""

from functools import lru_cache

import torch

import med_vlm.enums
from med_vlm.models.registry import MODEL_REGISTRY, ModelBundle

# Extra from_pretrained kwargs for models that benefit from specific dtypes.
# Kept out of ModelConfig to preserve its frozen / hashable contract.
_LOAD_KWARGS: dict[med_vlm.enums.ModelName, dict] = {
    med_vlm.enums.ModelName.GRANITE_VISION: {"dtype": "auto"},
    # bfloat16 ensures vision encoder and LM head share the same dtype;
    # "auto" would load SmolVLM's vision encoder as float16 and LM as bfloat16.
    med_vlm.enums.ModelName.QWEN2VL: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.SMOLVLM: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.MEDVLTHINKER: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.QWEN25VL: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.MEDGEMMA: {"dtype": torch.bfloat16},
    # v7 (2026-10-13): new zero-shot (and MedVLM-R1, registered before without dtype)
    med_vlm.enums.ModelName.MEDVLM: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.LINGSHU: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.HUATUO_QWEN25: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.INTERNVL3: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.GEMMA3: {"dtype": torch.bfloat16},
    med_vlm.enums.ModelName.LLAVA_MED: {"dtype": "auto", "trust_remote_code": True},
    # BiomedCLIP is an OpenCLIP model registered via custom code on HF Hub.
    med_vlm.enums.ModelName.BIOMEDCLIP: {"trust_remote_code": True},
}

# Extra from_pretrained kwargs for the processor (separate from model kwargs).
_PROCESSOR_LOAD_KWARGS: dict[med_vlm.enums.ModelName, dict] = {
    med_vlm.enums.ModelName.LLAVA_MED: {"trust_remote_code": True},
    med_vlm.enums.ModelName.BIOMEDCLIP: {"trust_remote_code": True},
}

# Qwen2VL uses dynamic resolution; without a pixel cap it generates thousands of
# visual tokens that exceed max_length and break the image-token/input_ids count
# invariant.  512×512 yields ~336 visual tokens, safely under our 2048 max_length.
# Qwen2.5-VL (MEDVLTHINKER's base) shares the same dynamic-resolution vision
# encoder family and is subject to the same overflow risk.
_QWEN2VL_MAX_PIXELS: int = 512 * 512
_DYNAMIC_RESOLUTION_MODELS = (
    med_vlm.enums.ModelName.QWEN2VL,
    med_vlm.enums.ModelName.MEDVLTHINKER,
    med_vlm.enums.ModelName.QWEN25VL,
    med_vlm.enums.ModelName.MEDVLM,          # v7 (2026-10-13): Qwen2-VL
    med_vlm.enums.ModelName.LINGSHU,         # v7 (2026-10-13): Qwen2.5-VL
    med_vlm.enums.ModelName.HUATUO_QWEN25,   # v7 (2026-10-13): Qwen2.5-VL
)

# SmolVLM2's default processor config (do_image_splitting=True,
# size.longest_edge=1536) tiles each image into up to a 4x4 grid of 384x384
# crops plus one global thumbnail (up to 17 tiles). Each 384x384 tile always
# contributes a *fixed* 81 visual tokens (384/14=27 patches/side, /3
# scale_factor = 9 tokens/side, 9x9=81) -- 384 matches vision_config's native
# image_size, so this per-tile token count cannot be changed by overriding
# max_image_size/do_image_splitting. Overriding them (single view capped to
# 756px) breaks the model's internal invariant that
# <image> placeholder tokens per tile == image_hidden_states length, which
# is computed from the *native* 384px tile, not from a caller override --
# it fails with "At least one sample has <image> tokens not
# divisible by patch_size" (verified via transformers' SmolVLM
# inputs_merger source, med_vlm/models/loader.py history).
#
# 17 tiles * 81 tokens ~= 1,377 visual tokens per single Kvasir image, ~4x
# Qwen2VL's capped ~336 (see _QWEN2VL_MAX_PIXELS above) -- an asymmetry
# never addressed when the Qwen2VL cap was added, confirmed as the dominant
# cause of SmolVLM2 training running ~4x slower per step than Qwen2VL
# despite identical batch/grad_accum/epoch config (2026-08-31 campaign).
#
# The only valid knob for controlling total tokens is `size.longest_edge`,
# the pre-split resize cap that controls how MANY native 384px tiles a image
# gets split into -- max_image_size and do_image_splitting must stay at
# their defaults (384 / True) to keep every tile's token count consistent
# with what the model's inputs_merger expects. size=768 (=2x384) yields a
# 2x2 grid + 1 global thumbnail = 5 tiles x 81 = 405 tokens for a typical
# landscape Kvasir frame (verified empirically against the real processor,
# pixel_values.shape == (1, 5, 3, 384, 384)) -- close to Qwen2VL's ~336 and
# nowhere near the original ~1,377, without touching the fixed-tile-size
# invariant.
_SMOLVLM_SIZE_LONGEST_EDGE: int = 768


@lru_cache
def load_model_bundle(model_name: med_vlm.enums.ModelName) -> ModelBundle:
    """
    Loads a model bundle (model, tokenizer, processor) based on the provided model
    name enum.
    Args:
        model_name (enums.ModelName): The enum value representing the model to load.
    Returns:
        ModelBundle: A dataclass containing the loaded model, tokenizer, and processor
        (if available).
    Raises:
        ValueError: If the provided model name is not supported or if there is an issue
        during loading.
    """
    config = MODEL_REGISTRY[model_name]
    load_kwargs = _LOAD_KWARGS.get(model_name, {})

    if model_name == med_vlm.enums.ModelName.MEDVLM:
        # v7 (2026-10-13): o config.json do MedVLM-R1 traz use_cache = null; o transformers 5 valida os campos e recusa. So aqui: le o config,
        # swaps a null use_cache for True and loads with that config (weights unchanged).
        from transformers import PretrainedConfig, Qwen2VLConfig
        cfg_dict, _ = PretrainedConfig.get_config_dict(model_name.value)
        if cfg_dict.get("use_cache") is None: cfg_dict["use_cache"] = True
        model = config.model_cls.from_pretrained(model_name.value, config=Qwen2VLConfig.from_dict(cfg_dict), **load_kwargs)
    else:
        model = config.model_cls.from_pretrained(model_name.value, **load_kwargs)

    # SmolVLM's vision encoder sub-module may load in float16 even when
    # dtype=bfloat16 is passed; cast the whole model to ensure consistency.
    if model_name == med_vlm.enums.ModelName.SMOLVLM:
        model = model.to(torch.bfloat16)

    tokenizer = (
        config.tokenizer_cls.from_pretrained(model_name.value)
        if config.tokenizer_cls
        else None
    )

    if model_name == med_vlm.enums.ModelName.MEDVLM:
        # v7 (2026-10-13): AutoProcessor/AutoTokenizer releem o config.json (use_cache = null) e falham; o processor e montado das pecas,
        # each one read from the repository's own file (preprocessor_config.json, tokenizer, chat_template.json).
        import json as _json
        from huggingface_hub import hf_hub_download
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor
        processor = Qwen2VLProcessor(image_processor=Qwen2VLImageProcessor.from_pretrained(model_name.value), tokenizer=Qwen2TokenizerFast.from_pretrained(model_name.value),
                                     video_processor=Qwen2VLVideoProcessor.from_pretrained(model_name.value),
                                     chat_template=_json.load(open(hf_hub_download(model_name.value, "chat_template.json")))["chat_template"])
    else:
        processor = (
            config.processor_cls.from_pretrained(
                model_name.value, **_PROCESSOR_LOAD_KWARGS.get(model_name, {})
            )
            if config.processor_cls
            else None
        )

    if model_name in _DYNAMIC_RESOLUTION_MODELS and processor is not None:
        img_proc = getattr(processor, "image_processor", None)
        if img_proc is not None:
            img_proc.max_pixels = _QWEN2VL_MAX_PIXELS
            img_proc.min_pixels = 4 * 28 * 28

    if model_name == med_vlm.enums.ModelName.SMOLVLM and processor is not None:
        img_proc = getattr(processor, "image_processor", None)
        if img_proc is not None:
            img_proc.size = {"longest_edge": _SMOLVLM_SIZE_LONGEST_EDGE}

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
    )
