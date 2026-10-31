"""Builds the Kvasir-VQA evaluation .arrow in the M3AE schema
(harness/interventions/arrow_io.py), from SimulaMet/Kvasir-VQA-x1.

Decomposes each row of x1 (composite question, natural language
answer) back into the atomic questions that generated it, via the
`original` field (JSON with the list {"q":..., "a":...} of the base
Kvasir-VQA template questions that were combined). This is necessary because:
- M3AE needs a short, closed-form answer vocabulary for
  classification: x1's composite answers are long sentences.
- The condition evaluation (C0-C4, C5) reuses this same arrow
  for the generative models via harness/circle_data_bridge.py; they
  receive the atomic question just like the others, with no disadvantage.

It also separately captures the answer to the finding-category template
question ("Are there any abnormalities in the image? Check
all that are present.") as `pathology_label` per image; used by
condition C5 (harness/interventions/make_c5_matched_pathology_swap.py)
to restrict the swap to a different category. See docs/superpowers/
plans/2026-08-21-kvasir-targeted-pathology-swap-condition.md.

TRAINING of the generative models (LoRA) does NOT use this arrow; it uses
Kvasir-VQA-x1 directly via med_vlm.data.loader (composite questions),
because med_vlm is already configured natively for this format.

Usage:
    python -m harness.prepro.prepare_kvasir_arrow \
        --split test \
        --output data/finetune_arrows/vqa_kvasir_test.arrow

Real finding during execution (Colab, 21-22/08/2026), not covered by the 17
patterns mapped in the planning session (which only looked at the TEST
split): the TRAIN split has 2 extra patterns; "Does this image contain
any finding?" (2263/6212 train images, 36%, had ONLY this question and
were being discarded entirely; answer 100% "no" across the 2263+237
occurrences measured in train+test; a degenerate question, with no
variance, but legitimately closed) and "What color is the anatomical
landmark?..." (small vocabulary, 22 combinations in train; the same
categorical pattern already handled for its sibling "what color is the
abnormality"). Both also appear in the test split (237 and 55
occurrences): the test arrow had to be regenerated after this fix to
stay consistent with the train one.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Callable

import pandas as pd

from harness.interventions.arrow_io import write_arrow

# Each Kvasir-VQA base template-question pattern mapped to the type
# closed/open. Prefix comparison, case-insensitive. 17 confirmed via
# inspection of the TEST split on 21/08/2026 + 2 findings only in the split of
# TRAIN during a real run (21-22/08/2026, see the module docstring:
# "does this image contain any finding" recovers 36% of the train images;
# "what color is the anatomical landmark" has a small vocabulary,
# verified before including, same pattern as "abnormality" below).
_CLOSED_PATTERNS = [
    "is there", "are there", "have all polyps", "how many",
    "what type of", "what is the size", "where in the image",
    "does this image contain any finding",
    "what color is the anatomical landmark",
]

_PATHOLOGY_QUESTION = "are there any abnormalities in the image? check all that are present."


def classify_question(question_text: str) -> str:
    """Classifies an atomic Kvasir question as 'closed' or 'open'.
    Raises ValueError if the text does not match any of the known
    patterns: a sign that Kvasir-VQA-x1 has changed or was poorly decomposed,
    which should not be silenced."""
    q = question_text.strip().lower()
    for pat in _CLOSED_PATTERNS:
        if q.startswith(pat):
            return "closed"
    if q.startswith("what color is the abnormality"):
        return "closed"  # short vocabulary in practice, verified before including
    raise ValueError(f"Atomic question not recognized among the known patterns: {question_text!r}")


def decompose_original_field(original_json: str) -> list[dict]:
    """Parses the `original` field (JSON string) of a Kvasir-VQA-x1 row,
    returning the list of atomic {"question":..., "answer":...}
    dicts. Returns [] if the field is empty or malformed (it does not
    raise an exception; third-party data, treat as absence of data)."""
    if not original_json:
        return []
    try:
        items = json.loads(original_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return [{"question": it["q"], "answer": it["a"]} for it in items if "q" in it and "a" in it]


def build_image_records(rows: list[dict]) -> list[dict]:
    """`rows` is the list of raw Kvasir-VQA-x1 rows (dict with
    img_id/image/original). Decomposes each row and groups by img_id,
    deduplicating identical atomic questions that appear repeated in
    multiple composite rows of the same image, and separately capturing
    the finding-category question.

    Returns a list of dicts, one per image:
        {"img_id": str, "image_url": str,
         "questions": [str, ...], "answers": [str, ...],
         "pathology_label": str | None}
    """
    by_img: dict[str, dict] = {}
    for row in rows:
        img_id = row["img_id"]
        if img_id not in by_img:
            by_img[img_id] = {
                "img_id": img_id,
                "image_url": row["image"],
                "seen_questions": {},  # question_text -> answer (dedup)
                "pathology_label": None,
            }
        for atom in decompose_original_field(row["original"]):
            q, a = atom["question"], atom["answer"]
            if q.strip().lower() == _PATHOLOGY_QUESTION:
                by_img[img_id]["pathology_label"] = a.strip().lower()
            try:
                if classify_question(q) != "closed":
                    continue
            except ValueError:
                continue  # unknown pattern; skip, do not halt the entire build
            by_img[img_id]["seen_questions"][q] = a

    records = []
    for img_id, data in by_img.items():
        if not data["seen_questions"]:
            continue  # image with no recognized closed question; out of scope
        records.append({
            "img_id": img_id,
            "image_url": data["image_url"],
            "questions": list(data["seen_questions"].keys()),
            "answers": list(data["seen_questions"].values()),
            "pathology_label": data["pathology_label"],
        })
    return records


def _download_image_bytes(url: str, max_retries: int = 5, backoff_base_s: float = 1.0) -> bytes:
    """Downloads the bytes of `url`, with exponential retry. Real finding in a
    run (21/08/2026, Colab): a transient `ConnectionResetError`/`URLError`
    happened in the middle of ~4,000 sequential downloads via
    urllib: without retry, a single network failure would bring down
    the entire script after having already downloaded most of it successfully, with no
    progress saved (`build_arrow_dataframe` only writes the arrow at the end).
    5 attempts, backoff 1s/2s/4s/8s/16s; generous enough to
    absorb transient instability of the image host without masking a
    real failure (broken URL, 404, etc., which `URLError` also covers)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base_s * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


_PLACEHOLDER_JPEG_BYTES: bytes | None = None


def _placeholder_image_bytes(url: str) -> bytes:
    """Always returns the same bytes of a black 1x1 JPEG, ignoring `url`;
    used in place of `_download_image_bytes` when the pixels will never
    actually be decoded (real finding, 22/08/2026: `train`/`val` only
    enter the M3AE pipeline to build the answer vocabulary; 
    `VQAKVASIRDataModule.setup()`, confirmed by reading m3ae/datamodules/
    base_datamodule.py, only reads the `answers`/`answer_labels` columns, never
    calls `train_dataloader()`/`val_dataloader()`; which would be the only
    points that decode the `image` column). Saves downloading thousands of
    whole training images when all that is needed is to rebuild the vocabulary
    to run inference from an already-ready checkpoint (e.g.: resuming
    after a Colab session crash, without re-training)."""
    global _PLACEHOLDER_JPEG_BYTES
    if _PLACEHOLDER_JPEG_BYTES is None:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, format="JPEG")
        _PLACEHOLDER_JPEG_BYTES = buf.getvalue()
    return _PLACEHOLDER_JPEG_BYTES


def build_arrow_dataframe(records: list[dict], split: str, qid_offset: int = 0,
                           download_fn: Callable[[str], bytes] = _download_image_bytes) -> pd.DataFrame:
    """Converts `records` (output of build_image_records) into a DataFrame in
    the ARROW_COLUMNS schema of arrow_io.py. question_id is generated
    deterministically by iteration order + qid_offset (document the
    offset used per split so it does not collide between train/test).
    `download_fn` is injectable so tests do not do a real download."""
    rows: dict[str, list] = {col: [] for col in [
        "image", "questions", "answers", "answer_labels", "answer_scores",
        "image_id", "question_id", "answer_type", "split",
    ]}
    next_qid = qid_offset
    for i, rec in enumerate(records):
        if i > 0 and i % 200 == 0:
            print(f"  ... {i}/{len(records)} imagens baixadas")
        image_bytes = download_fn(rec["image_url"])
        n_q = len(rec["questions"])
        qids = list(range(next_qid, next_qid + n_q))
        next_qid += n_q

        rows["image"].append(image_bytes)
        rows["questions"].append(rec["questions"])
        rows["answers"].append([[a] for a in rec["answers"]])  # 1 reference answer per question
        rows["answer_labels"].append([[0]] * n_q)  # placeholder: real vocabulary in build_kvasir_answer_vocab.py
        rows["answer_scores"].append([[1.0]] * n_q)
        rows["image_id"].append(f"kvasir/{rec['img_id']}")
        rows["question_id"].append(qids)
        rows["answer_type"].append([0] * n_q)  # 0 = closed, all questions here are already closed by construction
        rows["split"].append(split)
    return pd.DataFrame(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--qid-offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-images", action="store_true",
                         help="Does not download the actual pixels; writes a 1x1 placeholder JPEG in "
                              "place of each image. Only use for --split train when the goal is "
                              "to rebuild the answer vocabulary to run inference from "
                              "an already-ready checkpoint (without actually training); see "
                              "_placeholder_image_bytes(). NUNCA use pro --split test, cujas imagens "
                              "are actually evaluated.")
    return parser.parse_args()


if __name__ == "__main__":
    from datasets import load_dataset

    args = _parse_args()
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1", split=args.split)
    records = build_image_records(list(ds))
    download_fn = _placeholder_image_bytes if args.skip_images else _download_image_bytes
    df = build_arrow_dataframe(records, split="test" if args.split == "test" else "train",
                                qid_offset=args.qid_offset, download_fn=download_fn)
    write_arrow(df, args.output)
    n_questions = sum(len(q) for q in df["questions"])
    print(f"{len(df)} imagens, {n_questions} perguntas -> {args.output}")

    pathology_labels = {f"kvasir/{r['img_id']}": r["pathology_label"] for r in records if r["pathology_label"]}
    labels_path = args.output.replace(".arrow", "_pathology_labels.json")
    with open(labels_path, "w", encoding="utf-8") as fp:
        json.dump(pathology_labels, fp, indent=2, ensure_ascii=False)
    print(f"{len(pathology_labels)} pathology labels -> {labels_path}")
