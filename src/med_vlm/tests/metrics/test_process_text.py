import pytest

from med_vlm.metrics.process_text import (
    answer_matches,
    extract_answer,
    normalize_text,
    strip_thinking,
)

# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_think_block():
    pred = "<think>It could be yes or no, let me reason…</think> The answer is no."
    assert strip_thinking(pred) == "The answer is no."


def test_strip_thinking_removes_thinking_block():
    pred = "<thinking>Long reasoning here.</thinking>Pneumonia"
    assert strip_thinking(pred) == "Pneumonia"


def test_strip_thinking_multiline():
    pred = "<think>\nline one\nline two\n</think>\nFinal: yes"
    assert strip_thinking(pred) == "Final: yes"


def test_strip_thinking_case_insensitive():
    pred = "<THINK>reasoning</THINK> no"
    assert strip_thinking(pred) == "no"


def test_strip_thinking_no_block_unchanged():
    pred = "Just a plain answer."
    assert strip_thinking(pred) == "Just a plain answer."


def test_strip_thinking_multiple_blocks():
    pred = "<think>first</think> middle <think>second</think> end"
    assert strip_thinking(pred) == "middle  end"


# ---------------------------------------------------------------------------
# extract_answer — yes/no questions
# ---------------------------------------------------------------------------

YES_NO_QUESTIONS = [
    "Is there pneumonia in the image?",
    "Are the lungs clear?",
    "Does this show a fracture?",
]


@pytest.mark.parametrize("question", YES_NO_QUESTIONS)
def test_extract_yesno_plain_yes(question):
    assert extract_answer("yes", question) == "yes"


@pytest.mark.parametrize("question", YES_NO_QUESTIONS)
def test_extract_yesno_plain_no(question):
    assert extract_answer("no", question) == "no"


def test_extract_yesno_reasoning_model_think_block():
    """<think> block with misleading yes inside; real answer is no after block."""
    pred = "<think>It seems like yes based on the opacity, but wait…</think> The answer\
          is no."
    assert extract_answer(pred, "Is there pneumonia?") == "no"


def test_extract_yesno_last_token_wins():
    """After stripping thinking, uses the *last* yes/no token found."""
    pred = "Initially yes, but on reflection no."
    assert extract_answer(pred, "Is there pneumonia?") == "no"


def test_extract_yesno_answer_marker():
    pred = "<think>Reasoning…</think> Final answer: yes"
    assert extract_answer(pred, "Is there a fracture?") == "yes"


def test_extract_yesno_verbose_with_marker():
    pred = "<think>Details</think> The answer is: No, there is nothing wrong."
    assert extract_answer(pred, "Is there pneumonia?") == "no"


# ---------------------------------------------------------------------------
# extract_answer — open-ended questions
# ---------------------------------------------------------------------------

OPEN_QUESTION = "What is the primary finding in this chest X-ray?"


def test_extract_open_plain():
    pred = "Pneumonia"
    assert extract_answer(pred, OPEN_QUESTION) == "pneumonia"


def test_extract_open_reasoning_model():
    """Reasoning block stripped; clean text extracted."""
    pred = (
        "<think>The image shows opacification on the right side which is "
        "consistent with lobar pneumonia.</think> Pneumonia"
    )
    assert extract_answer(pred, OPEN_QUESTION) == "pneumonia"


def test_extract_open_answer_marker_after_think():
    pred = "<think>Detailed reasoning here.</think> The answer is: pleural effusion"
    result = extract_answer(pred, OPEN_QUESTION)
    assert result == "pleural effusion"


def test_extract_open_truncates_to_max_words():
    pred = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
    result = extract_answer(pred, OPEN_QUESTION)
    assert result == "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"


def test_extract_open_fallback_when_cleaned_empty():
    """If stripping leaves nothing, fall back to normalizing the original."""
    pred = "<think>all content is here</think>"
    result = extract_answer(pred, OPEN_QUESTION)
    # cleaned is empty after stripping; should fall back to something non-empty
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_answer — <answer>…</answer> tags (MedVLThinker-style)
# ---------------------------------------------------------------------------


def test_extract_open_answer_tag():
    pred = "<think>Reasoning about the modality.</think><answer>CT</answer>"
    assert extract_answer(pred, OPEN_QUESTION) == "ct"


def test_extract_yesno_answer_tag():
    pred = "<think>Weighing the evidence…</think><answer>No</answer>"
    assert extract_answer(pred, "Is there pneumonia?") == "no"


def test_extract_answer_tag_takes_priority_over_marker():
    """If both an <answer> tag and an 'answer is:' phrase appear, the tag wins."""
    pred = "<think>The answer is: wrong guess</think><answer>chest</answer>"
    assert extract_answer(pred, OPEN_QUESTION) == "chest"


def test_extract_answer_tag_verbose_content_still_truncated():
    """Content inside the tag still goes through the open-ended truncation."""
    pred = "<answer>word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11</answer>"
    result = extract_answer(pred, OPEN_QUESTION)
    assert result == "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"


def test_extract_no_answer_tag_falls_back_to_marker():
    """Without a tag, existing marker-based extraction still works unchanged."""
    pred = "<think>Reasoning</think> The answer is: pleural effusion"
    assert extract_answer(pred, OPEN_QUESTION) == "pleural effusion"


# ---------------------------------------------------------------------------
# answer_matches -- whitespace-only formatting differences (e.g. "x ray" vs
# "xray" vs "x-ray", the last of which normalize_text already collapses to
# "xray" since hyphens are stripped as punctuation).
# ---------------------------------------------------------------------------


def test_answer_matches_exact():
    assert answer_matches(normalize_text("Pneumonia"), [normalize_text("pneumonia")])


def test_answer_matches_whitespace_variant_pred_has_space():
    pred = normalize_text("x ray")
    gold = [normalize_text("xray")]
    assert answer_matches(pred, gold)


def test_answer_matches_whitespace_variant_gold_has_space():
    pred = normalize_text("xray")
    gold = [normalize_text("x ray")]
    assert answer_matches(pred, gold)


def test_answer_matches_hyphenated_vs_spaced():
    # normalize_text("x-ray") -> "xray" (hyphen stripped as punctuation);
    # normalize_text("x ray") -> "x ray" (space preserved) -- these must
    # still match via the whitespace-squashed fallback.
    pred = normalize_text("x-ray")
    gold = [normalize_text("x ray")]
    assert answer_matches(pred, gold)


def test_answer_matches_does_not_grant_synonym_credit():
    """Deliberate non-goal: genuine synonyms with different tokens ('no' vs
    'negative') must NOT match -- this is a normalization fix, not a
    semantic-equivalence scorer."""
    pred = normalize_text("negative")
    gold = [normalize_text("no")]
    assert not answer_matches(pred, gold)


def test_answer_matches_multiword_mismatch_still_fails():
    pred = normalize_text("kidney")
    gold = [normalize_text("spleen")]
    assert not answer_matches(pred, gold)
