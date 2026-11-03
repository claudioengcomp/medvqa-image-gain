"""Applies HEAL dataset support (E5, 15/09/2026) to the cloned M3AE repository, via the same
mechanism as `harness/patch_m3ae_kvasir.py`: the two generated files (`vqa_heal_dataset.py`,
`vqa_heal_datamodule.py`) are the Kvasir templates with `kvasir`→`heal` and `KVASIR`→`HEAL`; the
edits to `__init__` and the named config `task_finetune_vqa_heal` follow the same pattern
(batch 64, 15 epochs, lr 5e-6, like Kvasir; `vqa_label_size` from the real vocabulary,
`data/isbi_occlusion/heal_answer_vocab.json`, no=0/yes=1 fixed).

Arrows expected at `<M3AE>/data/finetune_arrows/`: vqa_heal_train.arrow, vqa_heal_val.arrow,
vqa_heal_test.arrow (= vqa_heal_occlusion_study_C0.arrow). Idempotent.

Usage (gpu1):
    python -m harness.patch_m3ae_heal --m3ae-dir ${MEDVQA_ROOT}/doutorado/M3AE \
        --vocab-json data/isbi_occlusion/heal_answer_vocab.json
"""
from __future__ import annotations

import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse
import json
import os

from harness import patch_m3ae_kvasir as K

_DATASET_FILE = K._DATASET_FILE.replace("kvasir", "heal").replace("KVASIR", "HEAL")
_DATAMODULE_FILE = K._DATAMODULE_FILE.replace("kvasir", "heal").replace("KVASIR", "HEAL")
_NAMED_CONFIG_TEMPLATE = K._NAMED_CONFIG_TEMPLATE.replace("kvasir", "heal")
_DATASETS_IMPORT_LINE = "from .vqa_heal_dataset import VQAHEALDataset\n"
_DATAMODULES_IMPORT_LINE = "from .vqa_heal_datamodule import VQAHEALDataModule\n"
_DATAMODULES_NEW_ENTRY = '"vqa_heal": VQAHEALDataModule,'


def write_dataset_file(m3ae_dir: str) -> str:
    path = os.path.join(m3ae_dir, "m3ae", "datasets", "vqa_heal_dataset.py")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_DATASET_FILE)
    return path


def write_datamodule_file(m3ae_dir: str) -> str:
    path = os.path.join(m3ae_dir, "m3ae", "datamodules", "vqa_heal_datamodule.py")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_DATAMODULE_FILE)
    return path


def patch_datasets_init(m3ae_dir: str) -> None:
    path = os.path.join(m3ae_dir, "m3ae", "datasets", "__init__.py")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read()
    if "VQAHEALDataset" in content:
        return
    if content and not content.endswith("\n"):
        content += "\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content + _DATASETS_IMPORT_LINE)


def patch_datamodules_init(m3ae_dir: str) -> None:
    path = os.path.join(m3ae_dir, "m3ae", "datamodules", "__init__.py")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read()
    if "VQAHEALDataModule" in content:
        return
    if K._DATAMODULES_MARKER not in content:
        raise ValueError(f"{path}: marker '_datamodules = {{' not found; file format changed?")
    if K._DATAMODULES_ANCHOR_ENTRY not in content:
        raise ValueError(f"{path}: entry 'vqa_vqa_rad' not found in the dictionary; file format changed?")
    content = content.replace(K._DATAMODULES_MARKER, f"\n{_DATAMODULES_IMPORT_LINE}{K._DATAMODULES_MARKER}", 1)
    content = content.replace(K._DATAMODULES_ANCHOR_ENTRY, f"{K._DATAMODULES_ANCHOR_ENTRY}\n    {_DATAMODULES_NEW_ENTRY}", 1)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)


def patch_config(m3ae_dir: str, vqa_label_size: int) -> None:
    path = os.path.join(m3ae_dir, "m3ae", "config.py")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read()
    if "def task_finetune_vqa_heal" in content:
        return
    if content and not content.endswith("\n"):
        content += "\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content + "\n\n" + _NAMED_CONFIG_TEMPLATE.format(vqa_label_size=vqa_label_size))


def apply_patch(m3ae_dir: str, vocab_json_path: str) -> int:
    with open(vocab_json_path, "r", encoding="utf-8") as fp:
        vocab = json.load(fp)
    if vocab.get("no") != 0 or vocab.get("yes") != 1:
        raise ValueError("HEAL vocabulary must have no=0 and yes=1 (match the test arrows)")
    vqa_label_size = len(vocab)
    write_dataset_file(m3ae_dir)
    write_datamodule_file(m3ae_dir)
    patch_datasets_init(m3ae_dir)
    patch_datamodules_init(m3ae_dir)
    patch_config(m3ae_dir, vqa_label_size)
    return vqa_label_size


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3ae-dir", required=True)
    parser.add_argument("--vocab-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n = apply_patch(args.m3ae_dir, args.vocab_json)
    print(f"Patch applied to {args.m3ae_dir}: VQAHEALDataset/DataModule created and registered, "
          f"task_finetune_vqa_heal with vqa_label_size={n}.")
