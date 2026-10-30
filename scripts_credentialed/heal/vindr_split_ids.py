"""Step 2 of the selective download: split of VinDr-CXR (train/ or test/) for each id of
HEAL, cross-referencing heal_image_ids.txt with image_labels_train.csv and
image_labels_test.csv from PhysioNet. Generates ids_train.txt and ids_test.txt with
relative paths train/<id>.dicom and test/<id>.dicom, and lists the ids missing
from both CSVs (expected: zero; if any, stop).

    python -m scripts_credentialed.heal.vindr_split_ids --ids data/isbi_occlusion/heal/heal_image_ids.txt \
        --labels-train <base>/annotations/image_labels_train.csv --labels-test <base>/annotations/image_labels_test.csv --out <base>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _ids_from_csv(path: str) -> set[str]:
    with open(path, newline="") as fp:
        return {row["image_id"] for row in csv.DictReader(fp)}


def split_ids(heal_ids: str, labels_train: str, labels_test: str, out_dir: str) -> dict:
    ids = [l.split("\t")[0] for l in Path(heal_ids).read_text().splitlines()[1:] if l.strip()]
    tr, te = _ids_from_csv(labels_train), _ids_from_csv(labels_test)
    train = sorted(i for i in ids if i in tr); test = sorted(i for i in ids if i in te and i not in tr)
    missing = sorted(i for i in ids if i not in tr and i not in te)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "ids_train.txt").write_text("".join(f"train/{i}.dicom\n" for i in train))
    Path(out_dir, "ids_test.txt").write_text("".join(f"test/{i}.dicom\n" for i in test))
    out = {"n_ids": len(ids), "n_train": len(train), "n_test": len(test), "missing": missing, "n_missing": len(missing)}
    Path(out_dir, "vindr_split_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True); ap.add_argument("--labels-train", required=True); ap.add_argument("--labels-test", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(split_ids(a.ids, a.labels_train, a.labels_test, a.out), indent=1))


if __name__ == "__main__":
    main()
