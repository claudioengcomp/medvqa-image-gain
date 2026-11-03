"""Checks the sha256 of M3AE arrows against a manifest of known
checksums BEFORE running real inference -- explicit request from the
candidate after the seed 2 finding (`kill -9` temporarily left
`vqa_nih_pa_test.arrow` with validation content; the
`.orig_backup` saved it that time, but "if the backup had not existed you
would have discovered this with the test arrow swapped, silently").

Detects silent file corruption/swap -- does NOT REPLACE
`swap_test_arrow`/`relabel_arrow_with_vocab` (which already take care of
correct label/vocabulary); checks that the file's BYTES are the same
as what was recorded as canonical the last time someone confirmed
it was correct.

Usage:
    python -m harness.prepro.verify_arrow_integrity \
        --manifest data/isbi/nih_test_arrow_checksums.json \
        --arrow-dir ${MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows
"""
from __future__ import annotations

import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse
import hashlib
import json
import os


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_arrows(manifest: dict[str, str], arrow_dir: str) -> dict[str, dict]:
    """`manifest`: {filename: expected_sha256}. Returns
    {filename: {"ok": bool, "expected": str, "actual": str|None,
    "error": str|None}} -- NEVER raises on its own (the caller
    decides what to do with a mismatch), but clearly marks a missing
    file vs. a different hash, which are distinct findings."""
    result = {}
    for filename, expected_sha256 in manifest.items():
        path = os.path.join(arrow_dir, filename)
        if not os.path.isfile(path):
            result[filename] = {"ok": False, "expected": expected_sha256, "actual": None,
                                 "error": f"file does not exist: {path}"}
            continue
        actual = sha256_of_file(path)
        result[filename] = {"ok": actual == expected_sha256, "expected": expected_sha256,
                             "actual": actual, "error": None}
    return result


def load_manifest_checksums(manifest_path: str) -> dict[str, str]:
    """Reads the manifest file ({"comment": ..., "checksums": {file:
    sha256}}) and returns only the `checksums` dict -- separate from the
    `comment` key so that `verify_arrows` never tries to treat the comment as
    a file name."""
    with open(manifest_path, encoding="utf-8") as fp:
        manifest_doc = json.load(fp)
    return manifest_doc["checksums"]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arrow-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest = load_manifest_checksums(args.manifest)

    results = verify_arrows(manifest, args.arrow_dir)
    any_failed = False
    for filename, r in results.items():
        if r["ok"]:
            print(f"OK   {filename}")
        else:
            any_failed = True
            if r["error"]:
                print(f"FAIL {filename}: {r['error']}")
            else:
                print(f"FAIL {filename}: expected {r['expected']}, found {r['actual']}")

    if any_failed:
        raise SystemExit(
            "arrow integrity check FAILED -- do not proceed to Gate 1 without investigating "
            "(check motivated by an earlier arrow mismatch found during training)"
        )
    print(f"\n{len(results)} arrow(s) checked, all match the manifest.")
