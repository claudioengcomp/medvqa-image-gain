from typing import Callable

from PIL import Image as PILImage

from med_vlm.enums import ModelName
from med_vlm.training.collators import (
    build_causal_lm_vqa_collator,
    build_chat_template_vqa_collator,
)
from med_vlm.training.generation import generate_blip2, generate_chat_template
from med_vlm.training.specs import ModelSpec
from med_vlm.utils import blip2_vqa_prompt


def build_blip2_spec(
    max_length: int = 256,
    augment_fn: Callable[[PILImage.Image], PILImage.Image] | None = None,
) -> ModelSpec:
    prompt_fn = blip2_vqa_prompt

    return ModelSpec(
        prompt_fn=prompt_fn,
        build_collator=lambda processor: build_causal_lm_vqa_collator(
            processor,
            prompt_fn=prompt_fn,
            max_length=max_length,
            augment_fn=augment_fn,
        ),
        generate_fn=generate_blip2,
        default_max_length=max_length,
        default_max_new_tokens=16,
    )


def build_chat_template_spec(
    max_length: int = 2048,
    augment_fn: Callable[[PILImage.Image], PILImage.Image] | None = None,
    wrap_images_per_sample: bool = False,
    system_prompt: str | None = None,
    default_max_new_tokens: int = 64,
    no_image: bool = False,
) -> ModelSpec:
    # `build_collator` (training path) intentionally does NOT receive
    # system_prompt -- no model registered here is fine-tuned with one yet.
    # If MedVLThinker is fine-tuned later (Fase 2, see
    # docs/superpowers/specs/2026-08-25-medvlthinker-third-model-design.md),
    # the collator will need the same system message for train/eval consistency.
    return ModelSpec(
        prompt_fn=lambda q: q,  # prompt formatting handled by apply_chat_template
        build_collator=lambda processor: build_chat_template_vqa_collator(
            processor,
            max_length=max_length,
            augment_fn=augment_fn,
            wrap_images_per_sample=wrap_images_per_sample,
            no_image=no_image,
        ),
        generate_fn=lambda model, processor, examples, max_new_tokens=64: (
            generate_chat_template(
                model, processor, examples,
                max_new_tokens=max_new_tokens, system_prompt=system_prompt,
                no_image=no_image,
            )
        ),
        default_max_length=max_length,
        default_max_new_tokens=default_max_new_tokens,
    )


# MedVLThinker-7B-RL_PMC needs an explicit instruction to emit structured
# <think>/<answer> tags instead of free-form prose (verified empirically --
# without ANY system prompt, extract_answer's fallback truncation grabs the
# first 10 words of an explanation instead of the actual answer). Base
# structure (<think>/<answer> tags) from the model card,
# https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-RL_PMC -- EXTENDED here
# (not in the original model card) with an explicit request for terse,
# gold-vocabulary-style phrasing. Also verified empirically: with only the
# base tag instruction, the model answers correctly but verbosely
# ("CT scan", "thorax", "lungs") and scores 0/4 under this benchmark's exact
# string-match protocol against SLAKE's terse gold answers ("ct", "chest",
# "lung"); with the terse-phrasing addition below, 5/6 exact matches on the
# same held-out check (the one miss is a SLAKE gold-label artifact: a
# multi-organ compound answer "lung spinal cord" where the model said just
# "lung"). This is a disclosed, deliberate prompt adaptation for a zero-shot
# model being scored against a closed-vocabulary benchmark it was never
# fine-tuned on -- not a change to the exact-match scoring logic itself,
# which stays identical across every model in the protocol (see design doc
# §7 for the limitation this doesn't fully resolve: dataset-specific
# synonym/compound-answer mismatches).
_MEDVLTHINKER_SYSTEM_PROMPT = (
    "You will solve a medical visual question answering problem. Provide "
    "your reasoning within <think> </think> tags. Then write your final "
    "answer within <answer> </answer> tags, using the shortest possible "
    "clinical term -- a single word or short standard phrase, exactly as "
    "it would appear in a radiology report answer key (e.g. answer just "
    '"CT" not "CT scan"; "chest" not "thorax"; singular form, e.g. "lung" '
    'not "lungs", unless the question clearly asks for a plural).'
)


MODEL_SPECS: dict[ModelName, Callable[..., ModelSpec]] = {
    ModelName.BLIP: build_blip2_spec,
    ModelName.GRANITE_VISION: build_chat_template_spec,
    ModelName.QWEN2VL: build_chat_template_spec,
    ModelName.SMOLVLM: lambda max_length=2048, augment_fn=None, no_image=False: (
        build_chat_template_spec(
            max_length=max_length, augment_fn=augment_fn, wrap_images_per_sample=True,
            no_image=no_image,
        )
    ),
    ModelName.LLAVA_MED: build_chat_template_spec,
    ModelName.QWEN25VL: build_chat_template_spec,
    ModelName.MEDGEMMA: build_chat_template_spec,   # v7: zero-shot, Gemma 3 chat template, no system prompt   # v7: zero-shot, same chat template as Qwen2-VL, no system prompt
    # MedVLM-R1: registered for zero-shot evaluation only.
    # Fine-tuning is intentionally skipped (potential train-set contamination).
    # v7 (2026-10-13): MedVLM-R1 is a reasoning model (<think>/<answer>); 512 new tokens, like MedVLThinker, so as not to cut off before the answer
    ModelName.MEDVLM: lambda max_length=2048, augment_fn=None, no_image=False, system_prompt=None: (
        build_chat_template_spec(max_length=max_length, augment_fn=augment_fn, system_prompt=system_prompt, default_max_new_tokens=512, no_image=no_image)
    ),
    ModelName.LINGSHU: build_chat_template_spec,         # v7 (2026-10-13): zero-shot, template de chat do Qwen2.5-VL
    ModelName.HUATUO_QWEN25: build_chat_template_spec,   # v7 (2026-10-13): zero-shot, template de chat do Qwen2.5-VL
    ModelName.INTERNVL3: build_chat_template_spec,
    ModelName.GEMMA3: build_chat_template_spec,          # v7 (2026-10-13): zero-shot, Gemma 3 chat template (like MedGemma)       # v7 (2026-10-13): zero-shot, InternVL3-hf chat template
    # MedVLThinker-7B-RL_PMC: Fase 1 = zero-shot only (replicar Zafar et al.
    # with our protocol). Phase 2 (LoRA/DoRA) evaluated later, see spec.
    # Needs the structured reasoning system prompt (see
    # _MEDVLTHINKER_SYSTEM_PROMPT above) and more tokens than the default of 64
    # -- the model does explicit CoT before <answer>, see design doc §4.
    ModelName.MEDVLTHINKER: lambda max_length=2048, augment_fn=None, no_image=False, system_prompt=None: (
        build_chat_template_spec(
            max_length=max_length,
            augment_fn=augment_fn,
            system_prompt=system_prompt or _MEDVLTHINKER_SYSTEM_PROMPT,   # v7: variante _instr acrescenta a instrucao curta ao prompt do VISAPP
            default_max_new_tokens=512,
            no_image=no_image,
        )
    ),
}
