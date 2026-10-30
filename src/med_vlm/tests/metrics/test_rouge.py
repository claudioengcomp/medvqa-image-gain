import pytest

from med_vlm.metrics.rouge import rouge_scores


def test_rouge_empty_preds():
    result = rouge_scores([], [])
    assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_rouge_empty_refs():
    result = rouge_scores([], ["ref"])
    assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_rouge_mismatched_length_raises():
    with pytest.raises(ValueError, match="equal length"):
        rouge_scores(["a", "b"], ["ref"])


def test_rouge_exact_match():
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    result = rouge_scores(preds, refs)
    assert result["rouge1"] == pytest.approx(1.0, abs=1e-3)
    assert result["rougeL"] == pytest.approx(1.0, abs=1e-3)


def test_rouge_no_overlap():
    preds = ["apple orange banana"]
    refs = ["dog cat fish"]
    result = rouge_scores(preds, refs)
    assert result["rouge1"] == pytest.approx(0.0, abs=1e-3)


def test_rouge_with_list_refs():
    preds = ["hello world"]
    refs = [["hello world", "hi world"]]
    result = rouge_scores(preds, refs)
    assert result["rouge1"] > 0.0


def test_rouge_returns_all_keys():
    result = rouge_scores(["test"], ["test"])
    assert set(result.keys()) == {"rouge1", "rouge2", "rougeL"}
