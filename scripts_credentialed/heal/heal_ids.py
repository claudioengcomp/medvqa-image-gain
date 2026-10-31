"""Step 1 of the selective download of VinDr-CXR (HEAL-MedVQA, VinDr part): ids of the
images used by HEAL, read from the Hugging Face parquets by column
projection (duckdb reads by range; the `image` column is never read or downloaded).

Canonical source: the configuration with image (train-*-of-00033 and test-*-of-00006:
16.380 and 2.851 rows). The `data/` configuration (23.091 /
3.933 rows) is reported separately for checking.

Outputs in data/isbi_occlusion/heal/: heal_items.parquet (heal_split, image_id,
question_id, anatomy, question_type, mask_h, mask_w) and heal_image_ids.txt
(image_id<TAB>HEAL splits, one id per line, header). No image data.

    python -m scripts_credentialed.heal.heal_ids --out data/isbi_occlusion/heal
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import duckdb

HF = "hf://datasets/MM-Hallu/HEAL-MedVQA"
DEFAULT_SOURCES = {"train": [f"{HF}/train-*-of-00033.parquet"], "test": [f"{HF}/test-*-of-00006.parquet"]}
COLS = "image_id, question_id, anatomy, question_type, mask_h, mask_w"


def build_heal_items(sources: dict[str, list[str]], out_dir: str) -> dict:
    con = duckdb.connect()
    if any(p.startswith("hf://") or p.startswith("http") for ps in sources.values() for p in ps):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    frames = []
    for split, paths in sources.items():
        lst = ", ".join(f"'{p}'" for p in paths)
        frames.append(con.execute(f"SELECT '{split}' AS heal_split, {COLS} FROM read_parquet([{lst}])").df())
    import pandas as pd
    items = pd.concat(frames, ignore_index=True)
    items.to_parquet(Path(out_dir) / "heal_items.parquet", index=False)
    rows = {s: int((items.heal_split == s).sum()) for s in sources}
    uniq = {s: int(items[items.heal_split == s].image_id.nunique()) for s in sources}
    splits_of = collections.defaultdict(set)
    for s, i in zip(items.heal_split, items.image_id):
        splits_of[i].add(s)
    both = sorted(i for i, ss in splits_of.items() if len(ss) > 1)
    with open(Path(out_dir) / "heal_image_ids.txt", "w") as fp:
        fp.write("image_id\theal_split\n")
        for i in sorted(splits_of):
            fp.write(f"{i}\t{','.join(sorted(splits_of[i]))}\n")
    summary = {"sources": sources, "rows": rows, "unique_ids": uniq, "unique_ids_total": len(splits_of), "ids_in_both_splits": both,
               "n_ids_in_both_splits": len(both)}
    Path(out_dir, "heal_ids_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/isbi_occlusion/heal")
    a = ap.parse_args(argv)
    s = build_heal_items(DEFAULT_SOURCES, a.out)
    print(json.dumps({k: v for k, v in s.items() if k != "ids_in_both_splits"}, indent=1))


if __name__ == "__main__":
    main()
