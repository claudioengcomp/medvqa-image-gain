from unittest.mock import MagicMock

import pytest
import torch

from med_vlm.training.collators import (
    build_chat_template_vqa_collator,
)


def test_build_chat_template_collator_no_apply_chat_template():
    """Raises if processor lacks apply_chat_template."""
    processor = MagicMock(spec=[])  # no attributes at all
    with pytest.raises(ValueError, match="apply_chat_template"):
        build_chat_template_vqa_collator(processor)


def _make_processor(pad_id=0):
    """Return a mock processor that exposes apply_chat_template."""
    processor = MagicMock()
    processor.apply_chat_template = MagicMock(return_value="chat text")
    processor.tokenizer = MagicMock(pad_token_id=pad_id)
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    return processor


def test_build_chat_template_collator_returns_callable():
    processor = _make_processor()
    collator = build_chat_template_vqa_collator(processor)
    assert callable(collator)


def test_chat_template_collator_output_has_labels():
    processor = _make_processor()
    collator = build_chat_template_vqa_collator(processor)

    batch = [{"image": MagicMock(), "question": "What is this?", "answers": ["yes"]}]
    result = collator(batch)
    assert "labels" in result


def test_chat_template_collator_labels_shape_matches_input_ids():
    processor = _make_processor()
    collator = build_chat_template_vqa_collator(processor)

    batch = [{"image": MagicMock(), "question": "What organ?", "answers": ["liver"]}]
    result = collator(batch)
    assert result["labels"].shape == result["input_ids"].shape
