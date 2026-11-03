"""Splits the Kvasir training arrow (harness/prepro/prepare_kvasir_arrow.py
--split train) into train/val; `SimulaMet/Kvasir-VQA-x1` only publishes
`train`/`test` splits, and M3AE needs all three.

Real finding (not an assumption), confirmed by reading the source code of the
`zhjohnchan/M3AE` repository, commit `14da377255280eed547e451f368cf71a43f3ca1c` (pinned
by `environment/colab_setup_m3ae.sh`), file
`m3ae/datamodules/vqa_vqa_rad_datamodule.py`: `VQAVQARADDataModule.setup()`
builds `self.answer2id`/`self.id2answer` (used by
`harness/extract_predictions_m3ae.py` to decode the prediction) by reading the
`answers`/`answer_labels` columns from `self.train_dataset` + `self.val_dataset`;
NOT from `test_dataset`. Without a `vqa_kvasir_val.arrow`, the equivalent
`VQAKVASIRDataModule` setup() would break due to the missing split.

Unlike `harness/prepro/prepare_vqa_rad_splits.py` (which splits the raw
VQA-RAD JSON *by question*, before the arrow exists): here the Kvasir arrow
already groups all the questions of an image into a single row (schema
`harness/interventions/arrow_io.py`): splitting by question would require
duplicating the image between train and val. We split by ROW (image) instead:
same spirit (deterministic, fixed seed, documented; it is not "the correct
split" clinically, just a stable choice).

Usage:
    python -m harness.prepro.split_kvasir_train_val \
        --input data/finetune_arrows/vqa_kvasir_train_full.arrow \
        --train-output data/finetune_arrows/vqa_kvasir_train.arrow \
        --val-output data/finetune_arrows/vqa_kvasir_val.arrow \
        --val-fraction 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from harness.interventions.arrow_io import read_arrow, write_arrow


def split_train_val_df(df: pd.DataFrame, val_fraction: float = 0.1,
                        seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits `df` (one row per image) into (train_df, val_df), shuffled
    deterministically by `seed`. `val_fraction` of the total (rounded) goes
    to val; the rest to train. The union of the two is exactly `df`, with no
    overlap. Rewrites the `split` column ('train'/'val') in each part."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction={val_fraction} must be in (0, 1)")

    order = sorted(range(len(df)), key=lambda i: df.iloc[i]["image_id"])  # deterministic order before shuffling
    rng = random.Random(seed)
    shuffled = order.copy()
    rng.shuffle(shuffled)

    n_val = round(len(shuffled) * val_fraction)
    val_positions = shuffled[:n_val]
    train_positions = shuffled[n_val:]

    val_df = df.iloc[val_positions].reset_index(drop=True).copy()
    train_df = df.iloc[train_positions].reset_index(drop=True).copy()
    val_df["split"] = "val"
    train_df["split"] = "train"
    return train_df, val_df


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Full training arrow (prepare_kvasir_arrow.py --split train)")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = read_arrow(args.input)
    train_df, val_df = split_train_val_df(df, val_fraction=args.val_fraction, seed=args.seed)
    write_arrow(train_df, args.train_output)
    write_arrow(val_df, args.val_output)
    n_train_q = int(sum(len(qs) for qs in train_df["questions"]))
    n_val_q = int(sum(len(qs) for qs in val_df["questions"]))
    print(f"train: {len(train_df)} images, {n_train_q} questions -> {args.train_output}")
    print(f"val:   {len(val_df)} images, {n_val_q} questions -> {args.val_output}")
