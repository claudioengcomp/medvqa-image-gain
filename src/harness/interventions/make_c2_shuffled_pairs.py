"""C2: generates a copy of the test .arrow file with the image<->question
pairing shuffled (each row receives the image of ANOTHER row,
never its own — a derangement), with a fixed and documented seed.

Usage:
    python -m harness.interventions.make_c2_shuffled_pairs \
        --input data/finetune_arrows/vqa_slack_test.arrow \
        --output data/finetune_arrows/vqa_slack_test_c2.arrow \
        --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np

from harness.interventions.arrow_io import read_arrow, write_arrow


def derangement(n: int, seed: int) -> np.ndarray:
    """Returns a permutation of range(n) with no fixed points (perm[i] != i for
    every i). Deterministic for fixed (n, seed)."""
    if n < 2:
        raise ValueError("derangement requer n >= 2")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fixed = [i for i in range(n) if perm[i] == i]
    while fixed:
        if len(fixed) >= 2:
            i, j = fixed[0], fixed[1]
            perm[i], perm[j] = perm[j], perm[i]
        else:
            i = fixed[0]
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
        fixed = [k for k in range(n) if perm[k] == k]
    return perm


def shuffle_image_pairs(input_path: str, output_path: str, seed: int = 42) -> None:
    df = read_arrow(input_path)
    n = len(df)
    perm = derangement(n, seed)

    df = df.copy()
    original_image = df["image"].tolist()
    original_image_id = df["image_id"].tolist()
    df["image"] = [original_image[p] for p in perm]
    df["image_id"] = [original_image_id[p] for p in perm]

    write_arrow(df, output_path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    shuffle_image_pairs(args.input, args.output, seed=args.seed)
    print(f"C2 escrito em {args.output}")
