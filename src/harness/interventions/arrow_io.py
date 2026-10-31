"""Shared utilities to read/write .arrow files in the same
format used by prepro/make_arrow.py::make_arrow_vqa of the LaPA/M3AE repository.

Schema (column order, confirmed by reading the make_arrow_vqa source code):
    image          bytes      - binary content of the image file (jpg)
    questions      list[str]
    answers        list[list[str]]
    answer_labels  list[list[int]]
    answer_scores  list[list[float]]
    image_id       str        - path of the image file (used as key)
    question_id    list[int]
    answer_type    list[int]  - 0=closed, 1=open
    split          str        - "train" | "val" | "test"
"""
from __future__ import annotations

import os

import pyarrow as pa
import pandas as pd

ARROW_COLUMNS = [
    "image",
    "questions",
    "answers",
    "answer_labels",
    "answer_scores",
    "image_id",
    "question_id",
    "answer_type",
    "split",
]


def read_arrow(path: str) -> pd.DataFrame:
    """Reads a .arrow file written in the make_arrow_vqa format and returns a DataFrame."""
    with pa.OSFile(path, "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    df = table.to_pandas()
    missing = [c for c in ARROW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Arquivo {path} does not have the expected columns: {missing}")
    return df


def write_arrow(df: pd.DataFrame, path: str) -> None:
    """Writes a DataFrame to .arrow in the same format used by make_arrow_vqa
    (same column order, same RecordBatchFileWriter).

    Creates the directory of `path` if it does not exist (real finding in a
    run, Colab, 21/08/2026: `data/finetune_arrows/` is gitignored and never
    existed in a fresh clone of the harness; pa.OSFile raises a
    low-level `FileNotFoundError`, with no clear message, after
    minutes of work already done, e.g. downloading ~4,000 Kvasir images)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = df[ARROW_COLUMNS]
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def count_questions(df: pd.DataFrame) -> int:
    """Counts the real number of evaluated questions in an arrow DataFrame.

    Each ROW of the arrow is an IMAGE, which carries a LIST of questions in
    the 'questions' column; counting rows (len(df)) counts images, not
    questions. This was exactly the bug that produced a wrong
    CI95%(bootstrap) in the first M3AE result (n=96 images instead of n=1061
    questions); always count questions with this function, never len(df)."""
    return int(sum(len(qs) for qs in df["questions"]))
