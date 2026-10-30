from unittest.mock import MagicMock, Mock

import pytest
import torch

from med_vlm.training.collators import build_causal_lm_vqa_collator, choose_first_answer


def test_choose_first_answer_with_answers():
    """Test choose_first_answer returns first answer when list is not empty."""
    answers = ["answer1", "answer2", "answer3"]
    result = choose_first_answer(answers)
    assert result == "answer1"


def test_choose_first_answer_empty_list():
    """Test choose_first_answer returns empty string for empty list."""
    answers = []
    result = choose_first_answer(answers)
    assert result == ""


def test_choose_first_answer_single_answer():
    """Test choose_first_answer with single answer."""
    answers = ["only_answer"]
    result = choose_first_answer(answers)
    assert result == "only_answer"


def test_build_causal_lm_vqa_collator_no_tokenizer():
    """Test that collator raises error when processor has no tokenizer."""
    processor = Mock(spec=[])
    with pytest.raises(ValueError, match="Processor must expose tokenizer"):
        build_causal_lm_vqa_collator(processor, prompt_fn=lambda x: x)


def test_build_causal_lm_vqa_collator_no_pad_token():
    """Test that collator raises error when tokenizer has no pad_token_id."""
    processor = Mock()
    processor.tokenizer = Mock(pad_token_id=None)
    with pytest.raises(ValueError, match="pad_token_id is None"):
        build_causal_lm_vqa_collator(processor, prompt_fn=lambda x: x)


def test_build_causal_lm_vqa_collator_valid():
    """Test successful collator creation with valid processor."""
    processor = Mock()
    processor.tokenizer = Mock(pad_token_id=0)
    processor.tokenizer.side_effect = Mock()

    collator_fn = build_causal_lm_vqa_collator(
        processor, prompt_fn=lambda x: f"Q: {x}", max_length=256
    )
    assert callable(collator_fn)


def test_collator_output_structure():
    """Test that collator returns correct output structure."""
    processor = MagicMock()
    processor.tokenizer = Mock(pad_token_id=0)

    # Mock processor return
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    processor.tokenizer.return_value = {"input_ids": [[1, 2]]}

    collator_fn = build_causal_lm_vqa_collator(
        processor,
        prompt_fn=lambda x: f"Q: {x}",
    )

    batch = [
        {"image": Mock(), "question": "What is this?", "answers": ["answer1"]},
    ]

    result = collator_fn(batch)
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result


def test_collator_masks_prompt_tokens():
    """Test that collator masks prompt tokens in labels."""
    processor = MagicMock()
    processor.tokenizer = Mock(pad_token_id=0)

    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    processor.tokenizer.return_value = {"input_ids": [[1, 2]]}

    collator_fn = build_causal_lm_vqa_collator(
        processor,
        prompt_fn=lambda x: f"Q: {x}",
    )

    batch = [
        {"image": Mock(), "question": "What?", "answers": ["answer"]},
    ]

    result = collator_fn(batch)
    labels = result["labels"]

    # First two tokens should be masked (-100)
    assert labels[0, 0] == -100
    assert labels[0, 1] == -100
