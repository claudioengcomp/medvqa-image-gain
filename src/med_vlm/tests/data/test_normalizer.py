import pytest
from PIL import Image as PILImage

from med_vlm.data.normalizer import build_normalizer


def test_normalizer_with_valid_data():
    normalizer = build_normalizer("image", "question", "answer", "id")
    example = {
        "image": "image_data",
        "question": "What is this?",
        "answer": "A cat",
        "id": 1,
    }
    normalized = normalizer(example)
    assert normalized == {
        "image": "image_data",
        "question": "What is this?",
        "answers": ["A cat"],
        "id": 1,
    }


def test_normalizer_with_string_answers():
    normalizer = build_normalizer("image", "question", "answer", "id")
    example = {
        "image": "image_data",
        "question": "What is this?",
        "answer": "A dog",
        "id": 2,
    }
    normalized = normalizer(example)
    assert normalized["answers"] == ["A dog"]


def test_normalizer_with_multiple_answers():
    normalizer = build_normalizer("image", "question", "answer", "id")
    example = {
        "image": "image_data",
        "question": "What is this?",
        "answer": ["A cat", "A dog"],
        "id": 3,
    }
    normalized = normalizer(example)
    assert normalized["answers"] == ["A cat", "A dog"]


def test_normalizer_without_id():
    normalizer = build_normalizer("image", "question", "answer")
    example = {"image": "image_data", "question": "What is this?", "answer": "A bird"}
    normalized = normalizer(example)
    assert normalized["id"] is None


def test_normalizer_missing_required_keys():
    normalizer = build_normalizer("image", "question", "answer", "id")

    example = {}

    with pytest.raises(KeyError):
        normalizer(example)


def test_normalizer_converts_grayscale_to_rgb():
    normalizer = build_normalizer("image", "question", "answer")
    grayscale_image = PILImage.new("L", (32, 32))
    example = {"image": grayscale_image, "question": "What mode?", "answer": "gray"}
    normalized = normalizer(example)
    assert normalized["image"].mode == "RGB"


def test_normalizer_keeps_rgb_image_unchanged():
    normalizer = build_normalizer("image", "question", "answer")
    rgb_image = PILImage.new("RGB", (32, 32))
    example = {"image": rgb_image, "question": "What mode?", "answer": "rgb"}
    normalized = normalizer(example)
    assert normalized["image"].mode == "RGB"


def test_normalizer_converts_rgba_to_rgb():
    normalizer = build_normalizer("image", "question", "answer")
    rgba_image = PILImage.new("RGBA", (32, 32))
    example = {"image": rgba_image, "question": "What mode?", "answer": "rgba"}
    normalized = normalizer(example)
    assert normalized["image"].mode == "RGB"


def test_normalizer_applies_preprocessor():
    sentinel = object()
    calls = []

    def mock_preprocessor(img: PILImage.Image) -> PILImage.Image:
        calls.append(sentinel)
        return img

    normalizer = build_normalizer(
        "image", "question", "answer", preprocessor=mock_preprocessor
    )
    rgb_image = PILImage.new("RGB", (32, 32))
    example = {"image": rgb_image, "question": "?", "answer": "yes"}
    normalizer(example)
    assert len(calls) == 1


def test_normalizer_no_preprocessor_by_default():
    """Passing no preprocessor (default) should not break anything."""
    normalizer = build_normalizer("image", "question", "answer")
    rgb_image = PILImage.new("RGB", (32, 32))
    example = {"image": rgb_image, "question": "?", "answer": "yes"}
    result = normalizer(example)
    assert result["image"].mode == "RGB"
