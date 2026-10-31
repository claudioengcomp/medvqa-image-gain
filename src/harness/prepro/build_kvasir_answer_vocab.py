"""Builds the Kvasir closed-form answer vocabulary (same logic
that prepro_vqa_vqa_rad() uses for VQA-RAD: one index per unique answer,
ordered by descending frequency), and rewrites the
answer_labels field of the train+val+test arrows with the real indices.

`--val-arrow` is required: M3AE's real vocabulary (`answer2id`,
built in `VQAVQARADDataModule.setup()`, confirmed by reading the source code
of `zhjohnchan/M3AE` commit `14da377255280eed547e451f368cf71a43f3ca1c`) is
built from train+val, not train+test; without the val split (see
`harness/prepro/split_kvasir_train_val.py`), the val `answer_labels` would
be left with the `[[0]]` placeholder from `prepare_kvasir_arrow.py`, breaking
training. We also include test (not just train+val) to cover any
answer that only appears there; broader than the minimum required by M3AE at
runtime, at no cost (the same spirit documented for VQA-RAD:
"own vocabulary, built... from train+val+test").

Usage:
    python -m harness.prepro.build_kvasir_answer_vocab \
        --train-arrow data/finetune_arrows/vqa_kvasir_train.arrow \
        --val-arrow data/finetune_arrows/vqa_kvasir_val.arrow \
        --test-arrow data/finetune_arrows/vqa_kvasir_test.arrow \
        --vocab-output data/finetune_arrows/kvasir_answer_vocab.json
"""
from __future__ import annotations

import argparse
import collections
import json

from harness.interventions.arrow_io import read_arrow, write_arrow


def build_vocab(answer_lists: list[list[str]]) -> dict[str, int]:
    """answer_lists: list of [unique_answer] per question (1 ref each,
    see build_arrow_dataframe). Returns {answer: index}, ordered by
    descending frequency (ties broken alphabetically, deterministic)."""
    counter = collections.Counter(a[0] for a in answer_lists)
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return {answer: i for i, (answer, _count) in enumerate(ordered)}


def apply_vocab(answer_lists: list[list[str]], vocab: dict[str, int]) -> list[list[int]]:
    return [[vocab.get(a[0], -1)] for a in answer_lists]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", required=True)
    parser.add_argument("--val-arrow", required=True)
    parser.add_argument("--test-arrow", required=True)
    parser.add_argument("--vocab-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df_train = read_arrow(args.train_arrow)
    df_val = read_arrow(args.val_arrow)
    df_test = read_arrow(args.test_arrow)

    all_answers = []
    for df in (df_train, df_val, df_test):
        for row_answers in df["answers"]:
            all_answers.extend(row_answers)
    vocab = build_vocab(all_answers)

    with open(args.vocab_output, "w", encoding="utf-8") as fp:
        json.dump(vocab, fp, indent=2, ensure_ascii=False)

    for path, df in [(args.train_arrow, df_train), (args.val_arrow, df_val), (args.test_arrow, df_test)]:
        df["answer_labels"] = [apply_vocab(row_answers, vocab) for row_answers in df["answers"]]
        write_arrow(df, path)

    print(f"vocabulary: {len(vocab)} unique answers -> {args.vocab_output}")
    print(f"arrows rewritten with real answer_labels: {args.train_arrow}, {args.val_arrow}, {args.test_arrow}")
