from unittest.mock import MagicMock, patch

import pytest
import torch

from med_vlm.training.generation import generate_blip2, generate_chat_template


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.tokenizer.batch_decode.return_value = [
        "Question: What is this? Answer: A cat",
        "Question: Count items Answer: 5 items",
        "No answer format text",
    ]
    return processor


def test_generate_blip2_basic(mock_model, mock_processor):
    examples = [
        {"image": "img1", "question": "What is this?"},
        {"image": "img2", "question": "Count items"},
        {"image": "img3", "question": "Describe"},
    ]

    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    with patch("med_vlm.training.generation.blip2_vqa_prompt", return_value="prompt"):
        result = generate_blip2(mock_model, mock_processor, examples)

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "A cat"
    assert result[1] == "5 items"
    assert result[2] == "No answer format text"


def test_generate_blip2_model_eval_called(mock_model, mock_processor):
    examples = [{"image": "img", "question": "Q?"}]
    mock_model.generate.return_value = torch.tensor([[1]])

    with patch("med_vlm.training.generation.blip2_vqa_prompt", return_value="prompt"):
        generate_blip2(mock_model, mock_processor, examples)

    mock_model.eval.assert_called_once()


def test_generate_blip2_max_new_tokens(mock_model, mock_processor):
    examples = [{"image": "img", "question": "Q?"}]
    mock_model.generate.return_value = torch.tensor([[1]])

    with patch("med_vlm.training.generation.blip2_vqa_prompt", return_value="prompt"):
        generate_blip2(mock_model, mock_processor, examples, max_new_tokens=32)

    mock_model.generate.assert_called_once()
    assert mock_model.generate.call_args[1]["max_new_tokens"] == 32


def test_generate_blip2_empty_examples(mock_model, mock_processor):
    mock_processor.tokenizer.batch_decode.return_value = []
    mock_model.generate.return_value = torch.tensor([])

    result = generate_blip2(mock_model, mock_processor, [])

    assert result == []


def test_generate_blip2_answer_extraction_with_colon(mock_model, mock_processor):
    mock_processor.tokenizer.batch_decode.return_value = ["Q: Test? Answer: Result"]
    mock_model.generate.return_value = torch.tensor([[1]])

    with patch("med_vlm.training.generation.blip2_vqa_prompt", return_value="prompt"):
        result = generate_blip2(
            mock_model, mock_processor, [{"image": "img", "question": "Q?"}]
        )

    assert result[0] == "Result"


def test_generate_blip2_multiple_colons(mock_model, mock_processor):
    mock_processor.tokenizer.batch_decode.return_value = ["Answer: First: Second"]
    mock_model.generate.return_value = torch.tensor([[1]])

    with patch("med_vlm.training.generation.blip2_vqa_prompt", return_value="prompt"):
        result = generate_blip2(
            mock_model, mock_processor, [{"image": "img", "question": "Q?"}]
        )

    assert result[0] == "First: Second"


@torch.no_grad()
def test_generate_blip2_no_grad_context(mock_model, mock_processor):
    examples = [{"image": "img", "question": "Q?"}]
    mock_model.generate.return_value = torch.tensor([[1]])

    with patch("med_vlm.training.generation.blip2_vqa_prompt", return_value="prompt"):
        result = generate_blip2(mock_model, mock_processor, examples)

    assert isinstance(result, list)


# ── generate_chat_template ────────────────────────────────────────────────────


def test_generate_chat_template_empty_examples():
    model = MagicMock()
    processor = MagicMock()
    result = generate_chat_template(model, processor, [])
    assert result == []
    model.eval.assert_called_once()


def test_generate_chat_template_basic(mock_model):
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt text"

    input_ids = torch.tensor([[1, 2, 3]])
    processor.return_value = {"input_ids": input_ids}

    out_ids = torch.tensor([[1, 2, 3, 6, 7]])
    mock_model.generate.return_value = out_ids
    processor.tokenizer.decode.return_value = " a cat "

    examples = [{"image": "img1", "question": "What is this?"}]
    result = generate_chat_template(mock_model, processor, examples)

    assert result == ["a cat"]
    mock_model.eval.assert_called_once()
    processor.apply_chat_template.assert_called_once()


def test_generate_chat_template_multiple_examples(mock_model):
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"

    input_ids = torch.tensor([[1, 2]])
    processor.return_value = {"input_ids": input_ids}

    mock_model.generate.return_value = torch.tensor([[1, 2, 5]])
    processor.tokenizer.decode.side_effect = [" yes ", " no "]

    examples = [
        {"image": "img1", "question": "Is this a cat?"},
        {"image": "img2", "question": "Is this a dog?"},
    ]
    result = generate_chat_template(mock_model, processor, examples)

    assert result == ["yes", "no"]
    assert processor.apply_chat_template.call_count == 2


def test_generate_chat_template_uses_processor_as_fallback_tokenizer(mock_model):
    """When processor has no .tokenizer attribute, falls back to processor.decode."""
    processor = MagicMock(spec=["apply_chat_template", "__call__", "decode"])
    processor.apply_chat_template.return_value = "prompt"

    input_ids = torch.tensor([[10, 20]])
    processor.return_value = {"input_ids": input_ids}
    mock_model.generate.return_value = torch.tensor([[10, 20, 30]])
    processor.decode.return_value = "result"

    examples = [{"image": "img", "question": "Q?"}]
    result = generate_chat_template(mock_model, processor, examples)

    assert result == ["result"]
    processor.decode.assert_called_once()


def test_generate_chat_template_max_new_tokens(mock_model):
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]])}
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    processor.tokenizer.decode.return_value = "ans"

    examples = [{"image": "img", "question": "Q?"}]
    generate_chat_template(mock_model, processor, examples, max_new_tokens=128)

    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs["max_new_tokens"] == 128
