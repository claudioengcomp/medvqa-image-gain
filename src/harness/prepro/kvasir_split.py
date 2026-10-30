"""Deterministic, image-stratified split of Kvasir-VQA-x1 --
pure logic, no network/GPU (separate from the CLI script in
build_kvasir_clean_split.py, which does the actual download).

Motivation: the official HuggingFace split (train/test) has 94% image
overlap between the two sides -- see docs/superpowers/specs/
2026-08-30-kvasir-clean-split-design.md §1.B. This module groups ALL the
unique images (from train+test combined) and draws a new split, disjoint
by img_id, stratified by pathology_label so as not to leave an entire
rare clinical category on just one side.
"""
from __future__ import annotations

import random
from collections import defaultdict


def split_records_by_image(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Splits `records` (output of
    harness.prepro.prepare_kvasir_arrow.build_image_records -- one dict per
    unique image, with `pathology_label`) into train/val/test, disjoint by
    `img_id`, stratified by `pathology_label` (images without a
    pathology_label fall into the `None` stratum, treated like any other).

    `ratios` = (train, val, test), must sum to 1.0 (tolerance 1e-9).
    Deterministic: the same `seed` always produces the same assignment.
    """
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios devem somar 1.0, somaram {total}")

    rng = random.Random(seed)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[rec["pathology_label"]].append(rec)

    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train:n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val:])

    return result
