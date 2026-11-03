"""E5 (15/09, D1-D2): HEAL TRAINING and VALIDATION arrows in M3AE format (the same one as the
test arrows `vqa_heal_occlusion_study_*.arrow` and the clean Kvasir arrows that `med_vlm` loads via
`load_condition_dataset`): one row per image, lists of questions/answers/labels.

D1 (like Kvasir): closed-form and open, answers reduced to atomic facts without the <seg> grounding:
  closed_yes/closed_no -> "yes"/"no" (same rule as the inventory);
  open_normal          -> "none";
  open_abnormal        -> finding(s) in lowercase, without "it suffers from" and without the final period.
D2: 10 % of the training images for validation, draw per image stratified by
(predominant anatomy, presence of finding), seed 20260913; disjoint from the test (ids checked).
Vocabulary (M3AE): "no" = 0 and "yes" = 1 fixed (match the test arrows), the rest by
decreasing frequency in train + val + test; written to heal_answer_vocab.json.
question_id starting from 3_100_000 (the test uses 3_000_000..3_002_850).

Usage (Mac, CPU):
    python -m scripts_credentialed.heal.make_heal_train_arrows \
        --annotations data/isbi_occlusion/heal/heal_annotations_train.parquet \
        --png-dir <png/train> --png-manifest data/isbi_occlusion/heal/png_manifest_train.json \
        --test-annotations data/isbi_occlusion/heal/heal_annotations_test.parquet \
        --out-dir data/isbi_occlusion --manifest results/isbi_occlusion/manifest_heal_train.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from harness.interventions.arrow_io import write_arrow
from scripts_credentialed.heal.inventory_heal import normalize_answer

QID_BASE = 3_100_000
SPLIT_SEED = 20260913
VAL_FRAC = 0.10
_SUFFERS = re.compile(r"^\s*it suffers from\s*", re.I)
_TAIL = re.compile(r"<seg>\.?", re.I)


def atomic_answer(answer: str, qtype: str) -> str:
    """Atomic fact without <seg>, in the spirit of the clean Kvasir (short, lowercase answers)."""
    if qtype in ("closed_yes", "closed_no"):
        return normalize_answer(answer, qtype)
    tail = _TAIL.split(str(answer))[-1].strip()
    if qtype == "open_normal":
        return "none"
    tail = _SUFFERS.sub("", tail).strip().rstrip(".").strip().lower()
    tail = re.sub(r"\s+", " ", tail)
    if tail in ("", "no abnormalities", "it has no abnormalities"):
        return "none"
    return tail


def split_images(ann: pd.DataFrame, val_frac: float = VAL_FRAC, seed: int = SPLIT_SEED) -> dict[str, str]:
    """image_id -> 'train'|'val'; stratum = (predominant anatomy, presence of a finding)."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[str]] = collections.defaultdict(list)
    for img, g in ann.groupby("image_id", sort=True):
        anat = g["anatomy"].value_counts().idxmax()
        abnormal = bool((g["question_type"] == "open_abnormal").any() or (g["question_type"] == "closed_yes").any())
        strata[(anat, abnormal)].append(img)
    out = {}
    for key in sorted(strata):
        imgs = sorted(strata[key]); rng.shuffle(imgs)
        n_val = int(round(val_frac * len(imgs)))
        for i, img in enumerate(imgs):
            out[img] = "val" if i < n_val else "train"
    return out


def build_vocab(answer_lists: list[list[str]]) -> dict[str, int]:
    cnt = collections.Counter(a for lst in answer_lists for a in lst)
    vocab = {"no": 0, "yes": 1}
    for a, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        if a not in vocab:
            vocab[a] = len(vocab)
    return vocab


def build(annotations: str, png_dir: str, png_manifest: str, out_dir: str, manifest_out: str,
          test_annotations: str | None = None, seed: int = SPLIT_SEED, val_frac: float = VAL_FRAC) -> dict:
    ann = pd.read_parquet(annotations)
    pm = json.loads(Path(png_manifest).read_text())["items"]
    test_ids = set()
    test_answers: list[list[str]] = []
    if test_annotations:
        te = pd.read_parquet(test_annotations)
        test_ids = set(te["image_id"])
        test_answers = [[atomic_answer(a, t)] for a, t in zip(te["answer"], te["question_type"])]
    leak = sorted(set(ann["image_id"]) & test_ids)
    if leak:
        raise ValueError(f"{len(leak)} training images also in the test: {leak[:5]}")
    missing = sorted(set(ann["image_id"]) - set(pm))
    if missing:
        raise ValueError(f"{len(missing)} training images without PNG in the manifest: {missing[:5]}")
    ann = ann.copy()
    ann["atomic"] = [atomic_answer(a, t) for a, t in zip(ann["answer"], ann["question_type"])]
    assignment = split_images(ann, val_frac, seed)
    vocab = build_vocab([[a] for a in ann["atomic"]] + test_answers)
    rows = {"train": [], "val": []}
    qid = QID_BASE
    per_split_q = collections.Counter()
    for img, g in ann.groupby("image_id", sort=True):
        split = assignment[img]
        p = Path(png_dir, pm[img]["png"])
        png_bytes = p.read_bytes()
        qs, ans, labels, scores, qids, atypes = [], [], [], [], [], []
        for r in g.itertuples(index=False):
            qs.append(str(r.question)); ans.append([r.atomic]); labels.append([vocab[r.atomic]]); scores.append([1.0])
            qids.append(qid); qid += 1; atypes.append(0 if r.question_type.startswith("closed") else 1)
        rows[split].append({"image": png_bytes, "questions": qs, "answers": ans, "answer_labels": labels, "answer_scores": scores,
                            "image_id": f"heal/{img}", "question_id": qids, "answer_type": atypes, "split": split})
        per_split_q[split] += len(qs)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files = {}
    for split, name in (("train", "vqa_heal_train.arrow"), ("val", "vqa_heal_val.arrow")):
        path = Path(out_dir, name); write_arrow(pd.DataFrame(rows[split]), str(path))
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                       "n_images": len(rows[split]), "n_questions": per_split_q[split]}
    vocab_path = Path(out_dir, "heal_answer_vocab.json"); vocab_path.write_text(json.dumps(vocab, indent=1, ensure_ascii=False) + "\n")
    man = {"dataset": "heal_vindr_train", "annotations": annotations, "png_manifest": png_manifest, "seed": seed, "val_frac": val_frac,
           "qid_base": QID_BASE, "n_images": int(ann["image_id"].nunique()), "n_questions": int(len(ann)),
           "by_type": {k: int(v) for k, v in ann["question_type"].value_counts().items()},
           "answers_by_type": {t: dict(collections.Counter(ann.loc[ann["question_type"] == t, "atomic"]).most_common(5)) for t in sorted(ann["question_type"].unique())},
           "vocab_size": len(vocab), "vocab_file": str(vocab_path), "files": files,
           "leakage_train_test_by_id": 0, "val_images": sorted(i for i, s in assignment.items() if s == "val")}
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_out).write_text(json.dumps(man, indent=1) + "\n")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True); ap.add_argument("--png-dir", required=True); ap.add_argument("--png-manifest", required=True)
    ap.add_argument("--test-annotations", default=None); ap.add_argument("--out-dir", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED); ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    a = ap.parse_args(argv)
    man = build(a.annotations, a.png_dir, a.png_manifest, a.out_dir, a.manifest, a.test_annotations, a.seed, a.val_frac)
    print(json.dumps({k: man[k] for k in ("n_images", "n_questions", "by_type", "answers_by_type", "vocab_size", "files")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
