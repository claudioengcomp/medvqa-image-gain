from unittest.mock import patch

import med_vlm.enums
from med_vlm.models.peft_adapters import DEFAULT_LORA_CONFIG, apply_peft_adapter


def test_apply_peft_adapter_dora_default_config(mock_model):
    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(mock_model, med_vlm.enums.PEFTMethod.DORA)
        mock_get_peft.assert_called_once()
        config = mock_get_peft.call_args[0][1]
        assert config.use_dora is True


def test_apply_peft_adapter_dora_custom_r(mock_model):
    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(mock_model, med_vlm.enums.PEFTMethod.DORA, {"r": 32})
        config = mock_get_peft.call_args[0][1]
        assert config.r == 32
        assert config.use_dora is True


def test_apply_peft_adapter_dora_preserves_defaults(mock_model):
    with patch("med_vlm.models.peft_adapters.get_peft_model") as mock_get_peft:
        apply_peft_adapter(mock_model, med_vlm.enums.PEFTMethod.DORA)
        config = mock_get_peft.call_args[0][1]
        assert config.lora_alpha == DEFAULT_LORA_CONFIG["lora_alpha"]
