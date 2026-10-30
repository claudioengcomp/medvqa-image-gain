from unittest.mock import patch

import pytest

import med_vlm.enums
from med_vlm.models.loader import load_model_bundle
from med_vlm.models.registry import ModelBundle


def test_load_model_bundle_success(mock_model_config):
    with patch(
        "med_vlm.models.loader.MODEL_REGISTRY",
        {med_vlm.enums.ModelName.BLIP: mock_model_config},
    ):
        result = load_model_bundle(med_vlm.enums.ModelName.BLIP)

        assert isinstance(result, ModelBundle)
        assert result.model is not None
        assert result.processor is not None


def test_load_model_bundle_no_tokenizer(mock_model_config):
    mock_model_config.tokenizer_cls = None

    with patch(
        "med_vlm.models.loader.MODEL_REGISTRY",
        {med_vlm.enums.ModelName.BLIP: mock_model_config},
    ):
        result = load_model_bundle(med_vlm.enums.ModelName.BLIP)

        assert result.tokenizer is None
        assert result.model is not None


def test_load_model_bundle_missing_from_registry(mock_model_config):
    with patch("med_vlm.models.loader.MODEL_REGISTRY", {}):
        with pytest.raises(KeyError):
            load_model_bundle(med_vlm.enums.ModelName.BLIP)
    with pytest.raises(ValueError):
        load_model_bundle(med_vlm.enums.ModelName("BERT"))


def test_load_model_bundle_caching(mock_model_config):
    with patch(
        "med_vlm.models.loader.MODEL_REGISTRY",
        {med_vlm.enums.ModelName.BLIP: mock_model_config},
    ):
        r1 = load_model_bundle(med_vlm.enums.ModelName.BLIP)
        r2 = load_model_bundle(med_vlm.enums.ModelName.BLIP)

        assert r1 is r2


def test_load_model_bundle_smolvlm_caps_visual_tokens_to_match_qwen2vl(
    mock_model_config,
):
    """SmolVLM2's default processor tiles each image into up to 17 crops of
    384x384 (~1,377 visual tokens) -- ~4x Qwen2VL's capped ~336. Each tile's
    token count is fixed by the model's native 384px resolution and cannot
    be changed directly (overriding max_image_size/do_image_splitting
    breaks the <image>-token/image_hidden_states invariant -- see the
    _SMOLVLM_SIZE_LONGEST_EDGE comment in loader.py). The only safe knob is
    `size`, the pre-split resize cap that controls how many native tiles an
    image gets split into; 768 (=2x384) yields a 2x2 grid + 1 global
    thumbnail = 5 tiles x 81 = 405 tokens, comparable to Qwen2VL's ~336."""
    with patch(
        "med_vlm.models.loader.MODEL_REGISTRY",
        {med_vlm.enums.ModelName.SMOLVLM: mock_model_config},
    ):
        result = load_model_bundle(med_vlm.enums.ModelName.SMOLVLM)

        img_proc = result.processor.image_processor
        assert img_proc.size == {"longest_edge": 768}
