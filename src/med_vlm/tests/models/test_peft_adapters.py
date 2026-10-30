from unittest.mock import patch

import pytest

import med_vlm.enums
from med_vlm.models.peft_adapters import DEFAULT_LORA_CONFIG, apply_peft_adapter


def test_apply_peft_adapter_lora_default_config(mock_model):
    model = mock_model

    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(model, med_vlm.enums.PEFTMethod.LORA)
        mock_get_peft.assert_called_once()
        assert mock_get_peft.call_args[0][0] == model


def test_apply_peft_adapter_adalora_default_config(mock_model):
    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(
            mock_model,
            med_vlm.enums.PEFTMethod.ADALORA,
            {"total_step": 100},
        )
        mock_get_peft.assert_called_once()
        assert mock_get_peft.call_args[0][0] == mock_model


def test_apply_peft_adapter_custom_config(mock_model):
    model = mock_model
    custom_config = {"r": 16, "lora_alpha": 64}

    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(model, med_vlm.enums.PEFTMethod.LORA, custom_config)
        mock_get_peft.assert_called_once()
        config = mock_get_peft.call_args[0][1]
        assert config.r == 16
        assert config.lora_alpha == 64


def test_apply_peft_adapter_unsupported_method(mock_model):
    model = mock_model

    with pytest.raises(ValueError, match="Unsupported peft_method"):
        apply_peft_adapter(model, "unsupported_method")


def test_apply_peft_adapter_config_merge(mock_model):
    model = mock_model
    custom_config = {"r": 16}

    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(model, med_vlm.enums.PEFTMethod.LORA, custom_config)
        config = mock_get_peft.call_args[0][1]
        assert config.r == 16
        assert config.lora_alpha == DEFAULT_LORA_CONFIG["lora_alpha"]


def test_apply_peft_adapter_adalora_requires_total_step(mock_model):
    with pytest.raises(ValueError, match="total_step"):
        apply_peft_adapter(mock_model, med_vlm.enums.PEFTMethod.ADALORA)


def test_apply_peft_adapter_none_returns_model(mock_model):
    assert apply_peft_adapter(mock_model, med_vlm.enums.PEFTMethod.NONE) is mock_model
