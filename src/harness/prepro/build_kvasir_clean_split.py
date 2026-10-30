"""Build the 3 leakage-free arrows (train/val/test) for Kvasir-VQA-x1 --
an image-disjoint split (harness.prepro.kvasir_split), reusing the atomic
decomposition and the image download already exercised in
harness/prepro/prepare_kvasir_arrow.py.

Two properties define it. Generative training and evaluation use the same
terse atomic facts, not the composite verbose answer. And the split is
image-disjoint, which the official HuggingFace split is not: 3,821 of its
4,058 test images (94%) also appear in training, verified by image
identifier, because the dataset assigns each image-question PAIR to a
partition rather than each image. An automated zero-overlap test guards the
property (tests/prepro/test_kvasir_split.py).

The official partitions are merged into the 6,449 images that carry at
least one question, composite natural-language answers are decomposed
into atomic (question, answer) facts, duplicates are removed per image
and question, and a 70/10/20 split is drawn BY IMAGE, stratified by
finding label, with a fixed seed. The resulting test set has 1,291 images
and 10,779 questions.

Usage (from the repository root):
    python -m harness.prepro.build_kvasir_clean_split \
        --output-dir data/finetune_arrows \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Callable

from harness.interventions.arrow_io import write_arrow
from harness.prepro.kvasir_split import split_records_by_image
from harness.prepro.prepare_kvasir_arrow import (
    _download_image_bytes,
    build_arrow_dataframe,
    build_image_records,
)

_QID_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def build_clean_arrows(
    train_rows: list[dict],
    test_rows: list[dict],
    output_dir: str,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
    download_fn: Callable[[str], bytes] = _download_image_bytes,
) -> dict:
    """Build the 3 leakage-free arrows from the raw HuggingFace `train`
    and `test` rows (already loaded -- injectable so tests can run without
    network access).
    Returns a manifest {"counts": {"train": N, "val": N, "test": N}}."""
    # Made an explicit responsibility of this function -- do not rely on
    # the side effect of write_arrow() creating the directory on the first
    # iteration of the loop (that would break as soon as the loop or
    # split_assignment is reordered; see the write_arrow docstring in
    # arrow_io.py for the real FileNotFoundError incident that motivated
    # the side effect in the first place).
    os.makedirs(output_dir, exist_ok=True)

    all_rows = list(train_rows) + list(test_rows)
    records = build_image_records(all_rows)
    split_assignment = split_records_by_image(records, ratios=ratios, seed=seed)

    counts = {}
    all_pathology_labels: dict[str, str] = {}
    split_map_rows = []
    for split_name, split_records in split_assignment.items():
        df = build_arrow_dataframe(
            split_records,
            split=split_name,
            qid_offset=_QID_OFFSET[split_name],
            download_fn=download_fn,
        )
        out_path = f"{output_dir}/vqa_kvasir_{split_name}_clean.arrow"
        write_arrow(df, out_path)
        counts[split_name] = len(df)

        for rec in split_records:
            split_map_rows.append({"img_id": rec["img_id"], "split": split_name})
            if rec["pathology_label"]:
                all_pathology_labels[f"kvasir/{rec['img_id']}"] = rec["pathology_label"]

    split_map_path = f"{output_dir}/kvasir_clean_split_map.csv"
    with open(split_map_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["img_id", "split"])
        writer.writeheader()
        writer.writerows(split_map_rows)

    labels_path = f"{output_dir}/vqa_kvasir_clean_pathology_labels.json"
    with open(labels_path, "w", encoding="utf-8") as fp:
        json.dump(all_pathology_labels, fp, indent=2, ensure_ascii=False)

    return {"counts": counts}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    return parser.parse_args()


if __name__ == "__main__":
    from datasets import load_dataset

    args = _parse_args()
    train_ds = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train")
    test_ds = load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
    manifest = build_clean_arrows(
        train_rows=list(train_ds),
        test_rows=list(test_ds),
        output_dir=args.output_dir,
        ratios=tuple(args.ratios),
        seed=args.seed,
    )
    print(f"Split limpo escrito em {args.output_dir}: {manifest['counts']}")
