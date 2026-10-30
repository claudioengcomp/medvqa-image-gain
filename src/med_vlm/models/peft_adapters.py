from typing import Any, Dict, Optional

from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model

import med_vlm.enums

# AdaLoRA only supports nn.Linear / Conv1D targets — Embedding layers raise ValueError.
# These module names are stripped from target_modules before building AdaLoraConfig.
_ADALORA_UNSUPPORTED_TARGETS = {"embed_tokens", "lm_head", "wte", "wpe"}


def _filter_adalora_targets(target_modules):
    if not target_modules:
        return target_modules
    return [m for m in target_modules if m not in _ADALORA_UNSUPPORTED_TARGETS]


DEFAULT_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ],
    "bias": "none",
    "task_type": TaskType.SEQ_2_SEQ_LM,
}


def apply_peft_adapter(
    model: Any,
    peft_method: med_vlm.enums.PEFTMethod = med_vlm.enums.PEFTMethod.LORA,
    peft_config: Optional[Dict] = None,
) -> Any:
    """
    Applies a PEFT adapter to the given model based on the specified method
    and configuration.
    Args:        model (Any): The base model to which the PEFT adapter will be applied.
        peft_method (enums.PEFTMethod): The PEFT method to use (e.g., LORA, ADALORA).
        Default is LORA.
        peft_config (Optional[Dict]): A dictionary containing additional configuration
        parameters for the PEFT adapter. If None, default parameters will be used.
    Returns:
        Any: The model with the applied PEFT adapter.
    Raises:
        ValueError: If an unsupported PEFT method is provided.
    """
    peft_config = peft_config or {}

    if peft_method == med_vlm.enums.PEFTMethod.LORA:
        cfg = {**DEFAULT_LORA_CONFIG, **peft_config}
        config = LoraConfig(**cfg)

    elif peft_method == med_vlm.enums.PEFTMethod.DORA:
        cfg = {**DEFAULT_LORA_CONFIG, **peft_config, "use_dora": True}
        config = LoraConfig(**cfg)

    elif peft_method == med_vlm.enums.PEFTMethod.ADALORA:
        total_step = peft_config.get("total_step")
        if not isinstance(total_step, int) or total_step <= 0:
            raise ValueError(
                "AdaLoRA requires peft_config['total_step'] to be an int > 0"
            )

        cfg = {**DEFAULT_LORA_CONFIG, **peft_config}
        # AdaLoRA only supports Linear/Conv1D — strip Embedding targets silently.
        if "target_modules" in cfg:
            cfg["target_modules"] = _filter_adalora_targets(cfg["target_modules"])
        config = AdaLoraConfig(**cfg)

    elif peft_method == med_vlm.enums.PEFTMethod.NONE:
        return model

    else:
        raise ValueError(f"Unsupported peft_method: {peft_method}")

    return get_peft_model(model, config)
