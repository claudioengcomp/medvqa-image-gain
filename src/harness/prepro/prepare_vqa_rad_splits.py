"""Splits the raw VQA-RAD trainset.json (MEVF/data_RAD format, with no
own valset.json; confirmed in this session by downloading the official package from
`https://huggingface.co/datasets/aiozai/MEVF/resolve/main/data_RAD.zip`) into
train/val, since M3AE's `prepro_vqa_vqa_rad()` expects the three files
(trainset.json/valset.json/testset.json).

Split by QUESTION (not by image), with a fixed seed; consistent with
VQA-RAD's own official train/test split, which already has image
overlap between train and test (finding confirmed in this session: the 203 images
in `testset.json` all also appear in `trainset.json`; that is how the
official benchmark works; it is not a leak introduced by us, it is a
known limitation of the dataset to record in the paper, not to "fix").

Usage:
    python -m harness.prepro.prepare_vqa_rad_splits \
        --trainset data_RAD/trainset.json \
        --train-output data_RAD/trainset_split.json \
        --val-output data_RAD/valset.json \
        --val-fraction 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random


def split_train_val(records: list[dict], val_fraction: float = 0.1,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Splits `records` (each with a unique 'qid' key) into
    (train, val), shuffled deterministically by `seed`. `val_fraction`
    of the total (rounded) goes to val; the rest to train. The union of the two
    is exactly `records`, with no overlap."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction={val_fraction} deve estar em (0, 1)")
    records_sorted = sorted(records, key=lambda r: r["qid"])  # deterministic order before shuffling
    rng = random.Random(seed)
    shuffled = records_sorted.copy()
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]
    return train_records, val_records


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainset", required=True, help="Caminho do trainset.json bruto (3064 registros)")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.trainset, "r", encoding="utf-8") as fp:
        records = json.load(fp)
    train_records, val_records = split_train_val(records, val_fraction=args.val_fraction, seed=args.seed)
    with open(args.train_output, "w", encoding="utf-8") as fp:
        json.dump(train_records, fp, indent=2, ensure_ascii=False)
    with open(args.val_output, "w", encoding="utf-8") as fp:
        json.dump(val_records, fp, indent=2, ensure_ascii=False)
    print(f"train: {len(train_records)} -> {args.train_output}")
    print(f"val:   {len(val_records)} -> {args.val_output}")
