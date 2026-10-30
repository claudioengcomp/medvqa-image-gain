"""Incremental inference checkpointing -- persists per-item predictions
during a long run of the C0-C6 battery and allows resuming where it stopped,
without re-running what is already done.

Kvasir has 10,779 questions in the leakage-free test split, and running the
11 conditions (C0, C1, C2, C4_OCC_25/50/75, C4_NOISE_25/50/75, C5, C6) with
real text generation takes on the order of hours per condition. Checkpointing
turns the cost of a connection drop or an OOM midway from the entire condition
into a few minutes.

Pure logic -- no torch/GPU -- so it can be unit-tested locally, following the
same convention as circle_data_bridge.py and circle_predictions.py (which
isolate the pure logic out of the extract_predictions_* scripts that require
a GPU and cannot run under the local pytest suite).

Checkpoint format: JSONL, one results-schema prediction per line
({"qid", "predicted", "gold", "correct"}) -- written in append mode, never
rewriting the whole file, so that persisting frequently does not become an
O(total items) cost on every batch.
"""
from __future__ import annotations

import json
import os


def load_checkpoint(checkpoint_path: str | None) -> tuple[list[dict], set]:
    """Load already-persisted predictions. checkpoint_path=None, or a file
    that does not exist, gives ([], set()) -- a run from scratch, behaving
    exactly as it did before this module existed (backward-compatible by
    design: callers that pass no checkpoint_path see no change)."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return [], set()
    predictions = []
    with open(checkpoint_path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions, {p["qid"] for p in predictions}


def append_checkpoint(checkpoint_path: str | None, new_predictions: list[dict]) -> None:
    """Append `new_predictions` to the checkpoint in append mode. No-op if
    checkpoint_path is None or new_predictions is empty (so no empty file is
    created for nothing)."""
    if checkpoint_path is None or not new_predictions:
        return
    with open(checkpoint_path, "a") as fp:
        for p in new_predictions:
            fp.write(json.dumps(p) + "\n")


def filter_pending(examples: list[dict], done_qids: set, qid_key: str = "id") -> list[dict]:
    """Return only the items of `examples` whose qid is not yet in
    `done_qids`. Comparison is always done as int (the same type used by the
    results schema) -- qid_key is coerced even when the dataset yields it as
    a str or a numpy integer."""
    return [ex for ex in examples if int(ex[qid_key]) not in done_qids]
