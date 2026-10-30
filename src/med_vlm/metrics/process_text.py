import re

PENCTUATION_REGEX = re.compile(r"[^\w\s]")

_YESNO_QUESTION_WORDS = re.compile(
    r"^(is|are|does|do|has|have|was|were|will|can|could|should|would|did)\b",
    re.IGNORECASE,
)
_YESNO_TOKEN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

# Matches reasoning blocks produced by R1-style models, e.g. <think>…</think>
# or <thinking>…</thinking>.  DOTALL so newlines inside the block are captured.
_THINK_BLOCK_REGEX = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)

# Explicit answer-marker prefixes that reasoning models emit after their chain
# of thought, e.g. "The answer is: yes", "Answer: chest x-ray", "Final answer: no".
_ANSWER_MARKER_REGEX = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*)?[:\-]?\s*(.+)",
    re.IGNORECASE,
)

# Explicit <answer>…</answer> tags, used by some RL-trained reasoning models
# (e.g. MedVLThinker) as a structured alternative to the "answer is:" marker
# above -- takes priority over it when present, since it's unambiguous.
_ANSWER_TAG_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_MAX_OPEN_WORDS = 10


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = PENCTUATION_REGEX.sub("", s)
    s = " ".join(s.split())
    return s


def _squash_whitespace(s: str) -> str:
    """Remove ALL whitespace from an already-normalize_text()-ed string.

    Comparison-only helper, NOT part of normalize_text() itself --
    normalize_text() must keep word boundaries intact because
    metrics/{bleu,rouge,meteor}.py rely on it for n-gram tokenization.
    """
    return "".join(s.split())


# 14/10/2026 (candidato; revision do zero-shot): numeros por extenso de "zero" a "ten" viram digitos. Usada SO na etapa de rescoring
# (harness/v7_txt/rescore_numbers.py), depois da extracao; normalize_text nao muda (as predicoes em curso seguem identicas).
_NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_NUM_RE = re.compile(r"\b(" + "|".join(_NUM_WORDS) + r")\b")


def numbers_to_digits(text_norm: str) -> str:
    """Troca palavras inteiras de 'zero' a 'ten' pelos digitos (texto ja normalizado)."""
    return _NUM_RE.sub(lambda m: _NUM_WORDS[m.group(1)], text_norm)


def answer_matches(pred_norm: str, gold_norms: list[str]) -> bool:
    """True if `pred_norm` (already normalize_text()-ed) exact-matches any
    of `gold_norms`, OR matches after removing all whitespace.

    The whitespace-squashed fallback catches pure formatting differences
    that punctuation-stripping alone misses -- e.g. normalize_text("x-ray")
    == "xray" (hyphen removed, no space) but normalize_text("x ray") ==
    "x ray" (space preserved), so today's exact match treats them as
    different answers despite meaning the same thing.

    Deliberately does NOT grant credit for genuine synonyms with different
    tokens (e.g. "no" vs "negative", "chest" vs "thorax") -- that would be a
    scoring-semantics change, not a normalization fix. See
    docs/superpowers/specs/2026-08-25-medvlthinker-third-model-design.md §8.
    """
    if pred_norm in gold_norms:
        return True
    squashed_pred = _squash_whitespace(pred_norm)
    return any(squashed_pred == _squash_whitespace(g) for g in gold_norms)


def strip_thinking(pred: str) -> str:
    """Remove <think>…</think> / <thinking>…</thinking> blocks from *pred*.

    Safe to call on any model output: if no thinking tags are present the
    string is returned unchanged (after stripping leading/trailing whitespace).
    """
    return _THINK_BLOCK_REGEX.sub("", pred).strip()


def extract_answer(pred: str, question: str) -> str:
    """Extract the core answer from a verbose model prediction.

    First strips any reasoning-model thinking blocks (<think>…</think>), then
    looks for an explicit <answer>…</answer> tag or an "answer is: …" marker.
    Falls back to the original yes/no and open-ended truncation logic on the
    cleaned text.

    For yes/no questions: returns the last 'yes' or 'no' token found (the
    final answer is more reliable than the first occurrence, which may appear
    inside a reasoning chain that was not fully stripped by the regex).
    For open-ended questions: truncates to the first _MAX_OPEN_WORDS words.
    """
    # 1. Strip thinking blocks produced by reasoning models (MedVLM-R1, etc.)
    cleaned = strip_thinking(pred)

    # 2. Look for an explicit <answer> tag first (unambiguous, used by e.g.
    #    MedVLThinker); fall back to the "answer is: …" marker otherwise.
    tag_match = _ANSWER_TAG_REGEX.search(cleaned)
    if tag_match:
        cleaned = tag_match.group(1).strip()
    else:
        marker_match = _ANSWER_MARKER_REGEX.search(cleaned)
        if marker_match:
            cleaned = marker_match.group(1).strip()

    q = question.strip()
    is_yesno = bool(_YESNO_QUESTION_WORDS.match(q))

    if is_yesno:
        # Use the *last* yes/no token: reasoning models that were not fully
        # stripped tend to conclude with the definitive answer.
        matches = list(_YESNO_TOKEN.finditer(cleaned))
        if matches:
            return normalize_text(matches[-1].group(0))

    # Open-ended or yes/no fallback: keep first _MAX_OPEN_WORDS words
    words = cleaned.split()
    truncated = " ".join(words[:_MAX_OPEN_WORDS])
    return normalize_text(truncated) or normalize_text(cleaned) or normalize_text(pred)
