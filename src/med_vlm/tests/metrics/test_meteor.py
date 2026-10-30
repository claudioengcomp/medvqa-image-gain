import pytest

from med_vlm.metrics.meteor import meteor_score


def test_meteor_empty_preds():
    result = meteor_score([], [])
    assert result == {"meteor": 0.0}


def test_meteor_empty_refs():
    result = meteor_score([], ["ref"])
    assert result == {"meteor": 0.0}


def test_meteor_mismatched_length_raises():
    with pytest.raises(ValueError, match="equal length"):
        meteor_score(["a", "b"], ["ref"])


def test_meteor_exact_match():
    preds = ["the quick brown fox"]
    refs = ["the quick brown fox"]
    result = meteor_score(preds, refs)
    assert result["meteor"] == pytest.approx(1.0, abs=0.05)


def test_meteor_no_overlap():
    preds = ["banana apple cherry"]
    refs = ["dog cat horse"]
    result = meteor_score(preds, refs)
    assert result["meteor"] < 0.1


def test_meteor_with_list_refs():
    preds = ["hello world"]
    refs = [["hello world", "hi world"]]
    result = meteor_score(preds, refs)
    assert result["meteor"] > 0.0


def test_meteor_returns_key():
    result = meteor_score(["test"], ["test"])
    assert "meteor" in result
    assert isinstance(result["meteor"], float)
