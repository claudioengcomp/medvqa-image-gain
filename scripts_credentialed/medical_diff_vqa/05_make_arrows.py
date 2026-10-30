"""v7_mimic/scripts/05_make_arrows.py -- 2026-10-10 -- lasid-gpu1 (CPU only)

Arrows in M3AE/med_vlm format (one row per image; lists of questions/answers/labels), replicating
harness/isbi_occlusion/heal/make_heal_train_arrows.py (occlusion_study repo @2b7e731): same write_arrow
(harness/interventions/arrow_io.py), same vocabulary rule ("no"=0, "yes"=1, rest by decreasing
frequency in train+val+test, alphabetical tiebreak), answer_type 0=closed 1=open, image_id "mimic/<dicom_id>".
Input: data/items_{train,val,test}.parquet (qid already assigned) + derived PNGs (manifest per split).
Output: data/vqa_mimic_{train,val,test}.arrow, data/manifest_mimic_{split}.json, data/answer_vocab.json.

  python 05_make_arrows.py --items-dir data --png-root <derived>/mimic_png --out-dir data
"""
from __future__ import annotations
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, collections, hashlib, json, socket, sys, time
from pathlib import Path
import pandas as pd

OCCLUSION_STUDY = f"{_MEDVQA_ROOT}/doutorado/medvqa-occlusion-study"
sys.path.insert(0, OCCLUSION_STUDY)
from harness.interventions.arrow_io import write_arrow, read_arrow, count_questions  # noqa: E402

SPLITS = ("train", "val", "test")


def build_vocab(answers):
    cnt = collections.Counter(answers)
    vocab = {"no": 0, "yes": 1}
    for a, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        if a not in vocab:
            vocab[a] = len(vocab)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", required=True); ap.add_argument("--png-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), help="splits a escrever agora (vocabulario sempre de train+val+test)")
    a = ap.parse_args()
    items = {s: pd.read_parquet(Path(a.items_dir, f"items_{s}.parquet")) for s in SPLITS}
    for s in SPLITS:
        assert items[s].qid.is_unique, s
    allq = pd.concat(items.values())
    assert allq.qid.is_unique, "qid repetido entre splits"
    # disjunction of patients and images between splits
    for x in SPLITS:
        for y in SPLITS:
            if x < y:
                assert not (set(items[x].subject_id) & set(items[y].subject_id)), f"subject_id em {x} e {y}"
                assert not (set(items[x].dicom_id) & set(items[y].dicom_id)), f"dicom_id em {x} e {y}"
    vocab = build_vocab(list(allq.answer))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    vp = Path(a.out_dir, "answer_vocab.json"); vp.write_text(json.dumps(vocab, indent=1, ensure_ascii=False) + "\n")
    out = {}
    for s in a.splits:
        df = items[s]
        pm = json.loads(Path(a.png_root, f"png_manifest_{s}.json").read_text())["items"]
        missing = sorted(set(df.dicom_id) - set(pm))
        if missing:
            raise SystemExit(f"{len(missing)} imagens de {s} sem PNG no manifesto: {missing[:5]}")
        rows = []
        for d, g in df.groupby("dicom_id", sort=True):
            p = Path(a.png_root, s, pm[d]["png"]); b = p.read_bytes()
            if hashlib.sha256(b).hexdigest() != pm[d]["sha256"]:
                raise SystemExit(f"sha256 do PNG difere do manifesto: {p}")
            g = g.sort_values("qid")
            rows.append({"image": b, "questions": [str(q) for q in g.question], "answers": [[x] for x in g.answer],
                         "answer_labels": [[vocab[x]] for x in g.answer], "answer_scores": [[1.0]] * len(g),
                         "image_id": f"mimic/{d}", "question_id": [int(q) for q in g.qid],
                         "answer_type": [0 if t == "closed" else 1 for t in g.answer_type], "split": s})
        path = Path(a.out_dir, f"vqa_mimic_{s}.arrow"); write_arrow(pd.DataFrame(rows), str(path))
        back = read_arrow(str(path)); nq = count_questions(back)
        assert nq == len(df) and len(back) == df.dicom_id.nunique()
        man = {"dataset": "mimic_diffvqa_single_image", "split": s, "items": str(Path(a.items_dir, f"items_{s}.parquet")),
               "png_manifest": str(Path(a.png_root, f"png_manifest_{s}.json")), "n_images": len(back), "n_questions": nq,
               "n_subjects": int(df.subject_id.nunique()), "n_studies": int(df.study_id.nunique()),
               "by_type": {k: int(v) for k, v in df.question_type.value_counts().items()},
               "by_answer_type": {k: int(v) for k, v in df.answer_type.value_counts().items()},
               "answers_top10": {t: dict(collections.Counter(df.loc[df.question_type == t, "answer"]).most_common(10)) for t in sorted(df.question_type.unique())},
               "vocab_size": len(vocab), "vocab_file": str(vp),
               "file": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size},
               "provenance": {"script": "v7_mimic/scripts/05_make_arrows.py", "reused": f"{OCCLUSION_STUDY}/harness/interventions/arrow_io.py @2b7e731; regra de vocabulario de make_heal_train_arrows.py",
                              "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname()}}
        Path(a.out_dir, f"manifest_mimic_{s}.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
        out[s] = {k: man[k] for k in ("n_images", "n_questions", "file")}
    print(json.dumps({"vocab_size": len(vocab), **out}, indent=1))


if __name__ == "__main__":
    main()
