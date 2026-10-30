from __future__ import annotations

from typing import Any, List

import torch

from med_vlm.utils import blip2_vqa_prompt


@torch.no_grad()
def generate_blip2(
    model: Any,
    processor: Any,
    examples: list[dict],
    max_new_tokens: int = 16,
) -> List[str]:
    model.eval()

    images = [x["image"] for x in examples]
    prompts = [blip2_vqa_prompt(x["question"]) for x in examples]

    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    texts = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    preds: List[str] = []
    for t in texts:
        preds.append(t.split("Answer:", 1)[-1].strip() if "Answer:" in t else t.strip())
    return preds


@torch.no_grad()
def generate_chat_template(
    model: Any,
    processor: Any,
    examples: list[dict],
    max_new_tokens: int = 64,
    system_prompt: str | None = None,
    no_image: bool = False,
) -> List[str]:
    """
    Generate predictions for chat-template models: Granite Vision, Qwen2-VL,
    SmolVLM, LLaVA-Med, MedVLThinker. Processes one example at a time to avoid
    padding complications during inference.

    `no_image` (v7_txt, 2026-10-09): message with only the question text and
    processor(text=..., images=None); same rule as the training collator.

    `system_prompt`, when given, is prepended as a `role="system"` message --
    used by reasoning models (e.g. MedVLThinker) that require an explicit
    instruction to emit <think>/<answer> tags; every other model leaves this
    unset and keeps the previous user-only message list unchanged.
    """
    model.eval()
    preds: List[str] = []

    for ex in examples:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": ([] if no_image else [{"type": "image"}]) + [
                    {"type": "text", "text": ex["question"]},
                ],
            }
        )
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt],
            images=None if no_image else [ex["image"]],
            return_tensors="pt",
        )
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[1]
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_ids = out_ids[0][input_len:]
        tokenizer = getattr(processor, "tokenizer", processor)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        preds.append(text.strip())

    return preds
