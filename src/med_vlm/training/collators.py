from typing import Any, Callable, Dict, List

import torch


def choose_first_answer(answers: List[str]) -> str:
    return answers[0] if answers else ""


def build_causal_lm_vqa_collator(
    processor: Any,
    *,
    prompt_fn: Callable[[str], str],
    max_length: int = 256,
    truncation: bool = True,
    augment_fn: Callable | None = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    """
    Builds batches where text = prompt + answer and labels are masked for the prompt
    tokens.
    Works well for BLIP2-style conditional generation where processor(text=...,
    images=...).

    Args:
        augment_fn: Optional callable ``PIL.Image -> PIL.Image`` applied to each
            image in the batch before the processor.  Pass only for the training
            collator; leave ``None`` for eval.
    """
    if getattr(processor, "tokenizer", None) is None:
        raise ValueError("Processor must expose tokenizer at processor.tokenizer")

    tok = processor.tokenizer
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id is None")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [x["image"] for x in batch]
        if augment_fn is not None:
            images = [augment_fn(img) for img in images]
        questions = [x["question"] for x in batch]
        answers = [choose_first_answer(x["answers"]) for x in batch]

        prompts = [prompt_fn(q) for q in questions]
        full_texts = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]

        inputs = processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        prompt_enc = tok(
            prompts,
            padding=False,
            truncation=truncation,
            max_length=max_length,
            return_tensors=None,
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

        labels = inputs["input_ids"].clone()
        labels[labels == pad_id] = -100

        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.size(1))
            labels[i, :plen] = -100

        inputs["labels"] = labels
        return inputs

    return collate


def build_chat_template_vqa_collator(
    processor: Any,
    *,
    max_length: int = 512,
    truncation: bool = True,
    augment_fn: Callable | None = None,
    wrap_images_per_sample: bool = False,
    no_image: bool = False,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    """
    Collator for models that use apply_chat_template (Granite Vision, Qwen2-VL,
    SmolVLM, LLaVA-Med). Builds batches with label masking on the prompt portion.

    no_image (v7_txt, 2026-10-09, text-only arm): when True the user
    message contains ONLY {"type": "text", "text": question} (without {"type": "image"}),
    the chat template is the same, and the processor is called with images=None --
    never an empty list nor a list of lists (SmolVLM), never the image processor
    separately (Qwen2-VL, image_grid_thw). Question and answer identical to training
    with image; no sentence warns the model that the image is missing.

    Truncation is applied *after* the processor call to avoid the image-token /
    input_ids count mismatch that these processors enforce.

    Args:
        augment_fn: Optional callable ``PIL.Image -> PIL.Image`` applied to each
            image in the batch before the processor.  Pass only for the training
            collator; leave ``None`` for eval.
        wrap_images_per_sample: When ``True``, images are passed as a list-of-lists
            ``[[img_0], [img_1], ...]`` so each sample's images are grouped together.
            Required by SmolVLM's processor; other processors accept a flat list.
    """
    if not hasattr(processor, "apply_chat_template"):
        raise ValueError("Processor must expose apply_chat_template")

    tok = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tok, "pad_token_id", None)

    def _make_messages(question: str, answer: str | None = None) -> list:
        content: list = ([] if no_image else [{"type": "image"}]) + [{"type": "text", "text": question}]
        msgs: list = [{"role": "user", "content": content}]
        if answer is not None:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return msgs

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = None if no_image else [x["image"] for x in batch]
        if augment_fn is not None and not no_image:
            images = [augment_fn(img) for img in images]
        questions = [x["question"] for x in batch]
        answers = [choose_first_answer(x["answers"]) for x in batch]

        full_texts = [
            processor.apply_chat_template(
                _make_messages(q, a),
                tokenize=False,
                add_generation_prompt=False,
            )
            for q, a in zip(questions, answers)
        ]
        prompt_texts = [
            processor.apply_chat_template(
                _make_messages(q),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]

        # Do NOT pass truncation/max_length here; these processors validate that
        # the number of image tokens in text matches input_ids, which truncation
        # would break.  We truncate manually below.
        proc_images = None if no_image else ([[img] for img in images] if wrap_images_per_sample else images)
        inputs = processor(
            text=full_texts,
            images=proc_images,
            padding=True,
            return_tensors="pt",
        )

        p_enc = processor(
            text=prompt_texts,
            images=proc_images,
            padding=True,
            return_tensors="pt",
        )
        prompt_lens: list[int] = p_enc["attention_mask"].sum(dim=1).tolist()

        # Manual post-hoc truncation to max_length; text tensors only.
        # pixel_values / image_sizes / image_grid_thw must NOT be truncated:
        # slicing them along dim-1 would corrupt tile counts and break the
        # image-token / input_ids invariant checked by LLaVA-Next / SmolVLM.
        # input_ids is only truncated when ALL image placeholder tokens fit
        # within max_length (i.e. the last <image> position < max_length).
        _TEXT_KEYS = {"input_ids", "attention_mask", "token_type_ids", "labels"}
        if truncation and inputs["input_ids"].size(1) > max_length:
            # Find the last image-token position across the batch.
            # Image token id: look up <image> from tokenizer if available,
            # otherwise skip truncation to be safe.
            img_token_id = getattr(tok, "convert_tokens_to_ids", lambda x: None)(
                getattr(tok, "image_token", "<image>")
            )
            safe_to_truncate = True
            if img_token_id is not None and img_token_id != tok.unk_token_id:
                image_mask = inputs["input_ids"] == img_token_id
                if image_mask.any():
                    last_img_pos = image_mask.nonzero()[:, 1].max().item()
                    safe_to_truncate = last_img_pos < max_length
            if safe_to_truncate:
                for key in list(inputs.keys()):
                    if key not in _TEXT_KEYS:
                        continue
                    if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() >= 2:
                        inputs[key] = inputs[key][:, :max_length]

        labels = inputs["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.size(1))
            labels[i, :plen] = -100

        inputs["labels"] = labels
        return inputs

    return collate
