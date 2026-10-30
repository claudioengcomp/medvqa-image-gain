from unittest.mock import patch

import pytest

from med_vlm.metrics.bleu import bleu_score


def test_bleu_score_returns_dict_with_bleu_key():
    result = bleu_score(["this is a test"], ["this is a test"])
    assert isinstance(result, dict)
    assert "bleu" in result


def test_bleu_score_perfect_match_long_is_one():
    preds = ["this is a test", "one more test here"]
    refs = ["this is a test", "one more test here"]
    result = bleu_score(preds, refs)
    assert result["bleu"] == pytest.approx(1.0, abs=1e-12)


def test_bleu_score_empty_preds_returns_zero():
    result = bleu_score([], [])
    assert result["bleu"] == 0.0


def test_bleu_score_is_between_zero_and_one():
    preds = ["pneumonia found here", "something else entirely"]
    refs = ["pneumonia found here", "completely different words"]
    result = bleu_score(preds, refs)
    assert 0.0 <= result["bleu"] <= 1.0


def test_bleu_score_length_mismatch_raises():
    with pytest.raises(AssertionError):
        bleu_score(["a"], ["a", "b"])


def test_bleu_score_short_strings_is_reasonable():
    preds = ["nothing", "pneumonia"]
    refs = ["nothing", "pneumonia"]
    result = bleu_score(preds, refs)
    assert 0.0 <= result["bleu"] <= 1.0


def test_bleu_score_builds_ref_streams_for_multi_refs_with_padding():
    preds = ["a", "b"]
    refs = [
        ["first_a", "second_a"],
        ["only_b"],
    ]

    expected_ref_streams = [
        ["first_a", "only_b"],
        ["second_a", "only_b"],
    ]

    with patch("med_vlm.metrics.bleu.sacrebleu.corpus_bleu") as mock_corpus_bleu:
        mock_corpus_bleu.return_value.score = 0.0

        bleu_score(preds, refs)

        mock_corpus_bleu.assert_called_once()
        _, kwargs = mock_corpus_bleu.call_args

        assert mock_corpus_bleu.call_args.kwargs["smooth_method"] == "exp"

        called_preds_norm = mock_corpus_bleu.call_args[0][0]
        called_ref_streams = mock_corpus_bleu.call_args[0][1]

        assert called_preds_norm == ["a", "b"]
        assert called_ref_streams == expected_ref_streams
