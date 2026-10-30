import pytest

from med_vlm.metrics.accuracy import accuracy
from med_vlm.metrics.process_text import normalize_text


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello World", "hello world"),
        ("  Hello   World  ", "hello world"),
        ("HeLLo WoRLd", "hello world"),
        ("Pneumonia.", "pneumonia"),
        ("No!!!", "no"),
        ("  multiple   spaces\tand\nlines  ", "multiple spaces and lines"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_accuracy_perfect_match_single_refs():
    preds = ["Nothing", "Pneumonia"]
    refs = ["nothing", "pneumonia"]
    assert accuracy(preds, refs) == 1.0


def test_accuracy_partial_match_single_refs():
    preds = ["Nothing", "Bronchitis"]
    refs = ["nothing", "pneumonia"]
    assert accuracy(preds, refs) == 0.5


def test_accuracy_no_match_single_refs():
    preds = ["A", "B"]
    refs = ["C", "D"]
    assert accuracy(preds, refs) == 0.0


def test_accuracy_empty_list_returns_zero():
    assert accuracy([], []) == 0.0


def test_accuracy_single_element():
    assert accuracy(["hello"], ["hello"]) == 1.0


def test_accuracy_any_match_multiple_refs():
    preds = ["no", "pneumonia"]
    refs = [["no", "negative"], ["pneumonia", "lung infection"]]
    assert accuracy(preds, refs) == 1.0


def test_accuracy_multiple_refs_partial():
    preds = ["negative", "flu"]
    refs = [["no", "negative"], ["pneumonia", "lung infection"]]
    assert accuracy(preds, refs) == 0.5


def test_accuracy_length_mismatch_raises():
    with pytest.raises(AssertionError):
        accuracy(["a"], ["a", "b"])
