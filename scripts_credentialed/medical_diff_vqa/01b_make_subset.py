"""v7_mimic/scripts/01b_make_subset.py -- 2026-10-10 -- lasid-gpu1 (CPU only)

Single-image subset of Medical-Diff-VQA 1.0.1 over MIMIC-CXR-JPG 2.1.0.
Selection order (the candidate's prompt from 10/10):
 1. single image: excludes question_type == difference; no other question mentions comparison (checked in schema_stats).
 2. types: closed-form = presence (yes/no) and view with yes/no answer ("is this AP/PA view?"); short open = location, level,
    type with answer in the fixed vocabulary (frequency >= MIN_FREQ in the whole table and <= MAX_WORDS words). Excludes abnormality
    and "which view is this image taken?" (open view).
 3. one image per study: dicom_id from mimic_all.csv; ViewPosition in {PA, AP} in the MIMIC-CXR-JPG metadata.
 4. split: mimic-cxr-2.0.0-split.csv.gz by subject_id (checks disjointness; a cross patient goes to the split with more studies).
 5. sizes: test ~3000 questions / ~1500 images (stratified by template x answer, <= 2 per image per type);
    val ~1500 / ~750 from validate; train ~15000 from train, <= 4 per study, quotas by template x answer. Seed 20260913.
Template = question with the entity (31 names from all_diseases.json) masked by <e> and, in presence, the location by <loc>.
"""
from __future__ import annotations
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, json, re, socket, time
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260913
MIN_FREQ, MAX_WORDS = 100, 4
RAW_D = Path(_os.environ.get("MEDDIFF_RAW", f"{_MEDVQA_ROOT}/doutorado/data/raw/medical-diff-vqa/1.0.1"))
RAW_M = Path(_os.environ.get("MIMICCXR_RAW", f"{_MEDVQA_ROOT}/doutorado/data/raw/mimic-cxr-jpg/2.1.0"))
V7 = Path(_os.environ.get("MIMIC_OUT", f"{_MEDVQA_ROOT}/v7_mimic")); DATA = V7 / "data"; DOCS = V7 / "docs"; AN = V7 / "analysis"
DATA.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
TARGET = {"test": (3000, 1500, 2), "val": (1500, 750, 2), "train": (15000, None, 4)}  # (questions, images, cap per image[/type])
QID_BASE = {"test": 3_200_000, "val": 3_210_000, "train": 3_220_000}  # HEAL uses 3_000_000 (test) and 3_100_000 (training); the candidate's request on 10/10: starting from 3_200_000
SPLIT_NAME = {"test": "test", "validate": "val", "train": "train"}
rep = {"provenance": {"script": "v7_mimic/scripts/01b_make_subset.py", "seed": SEED, "min_freq": MIN_FREQ, "max_words": MAX_WORDS,
                      "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname()}}

q = pd.read_csv(RAW_D / "mimic_pair_questions.csv", dtype={"study_id": "int64", "subject_id": "int64", "ref_id": "int64"}, keep_default_na=False)
m = pd.read_csv(RAW_D / "mimic_all.csv", dtype={"study_id": "int64", "subject_id": "int64"}, keep_default_na=False)[["study_id", "subject_id", "dicom_id", "view", "split"]]
meta = pd.read_csv(RAW_M / "mimic-cxr-2.0.0-metadata.csv.gz", keep_default_na=False, dtype=str)
sp = pd.read_csv(RAW_M / "mimic-cxr-2.0.0-split.csv.gz", dtype={"subject_id": "int64", "study_id": "int64"})
rep["inputs"] = {"pair_questions_rows": len(q), "mimic_all_rows": len(m), "metadata_rows": len(meta), "split_rows": len(sp)}

# 1. single image
rep["step1"] = {"difference_removed": int((q.question_type == "difference").sum())}
q = q[q.question_type != "difference"].copy()
cmp_re = re.compile(r"\b(?:compar|previous|prior|reference|change|earlier|interval|since)\w*", re.I)
rep["step1"]["comparison_mentions_removed"] = int(q.question.str.contains(cmp_re).sum())
q = q[~q.question.str.contains(cmp_re)]

# 2. types and vocabulary
vocab = {}
rep["step2"] = {}
for t in ("location", "level", "type"):
    vc = q.loc[q.question_type == t, "answer"].value_counts()
    keep = vc[(vc >= MIN_FREQ) & (vc.index.str.split().str.len() <= MAX_WORDS)]
    vocab[t] = sorted(keep.index)
    rep["step2"][t] = {"n_distinct_all": int(len(vc)), "n_vocab": int(len(keep)), "coverage_rows": float(keep.sum() / vc.sum())}
json.dump(vocab, open(DATA / "open_answer_vocab_fixed.json", "w"), indent=1)
closed = (q.question_type == "presence") | ((q.question_type == "view") & q.answer.isin(["yes", "no"]))
openok = q.question_type.isin(["location", "level", "type"]) & np.array([a in vocab.get(t, ()) for a, t in zip(q.answer, q.question_type)])
rep["step2"]["removed_abnormality"] = int((q.question_type == "abnormality").sum())
rep["step2"]["removed_view_open"] = int(((q.question_type == "view") & ~q.answer.isin(["yes", "no"])).sum())
rep["step2"]["removed_open_out_of_vocab"] = int((q.question_type.isin(["location", "level", "type"]) & ~openok).sum())
assert (q.loc[closed, "answer"].isin(["yes", "no"])).all()
q = q[closed | openok].copy()
q["answer_type"] = np.where(q.question_type.isin(["presence", "view"]), "closed", "open")
rep["step2"]["kept"] = int(len(q))

# templates
ents = sorted((e.strip() for e in json.load(open(AN / "entities.json"))["entities"]), key=len, reverse=True)
ent_re = re.compile(r"\b(" + "|".join(re.escape(e) for e in ents) + r")\b")


def template(s: str) -> str:
    t, n = ent_re.subn("<e>", s)
    if n != 1:
        return "UNMASKED::" + s
    t = re.sub(r" in the .+? area\?$", " in the <loc> area?", t)
    if " in the " in t and "<loc>" not in t and "this image" not in t:
        t = re.sub(r"<e> in the .+?\?$", "<e> in the <loc>?", t)
    t = re.sub(r"\s+", " ", t).replace(" ?", "?").strip()  # entities with trailing space generated variants of the same template
    return t


q["template"] = [qq if ty == "view" else template(qq) for qq, ty in zip(q.question, q.question_type)]  # view: AP/PA is not a disease entity; each question is its own template
unm = q.loc[q.template.str.startswith("UNMASKED"), "question"].value_counts()
rep["templates"] = {"unmasked": int(unm.sum()), "unmasked_examples": unm.head(20).to_dict(), "n_templates": int(q.template.nunique()),
                    "by_template": q.groupby(["question_type", "template"]).size().reset_index().values.tolist()}
assert rep["templates"]["unmasked"] == 0, unm.head(20)
tmpl_ids = {t: f"T{i:02d}" for i, t in enumerate(sorted(q.template.unique()))}
q["template_id"] = q.template.map(tmpl_ids)

# 3. image per study + ViewPosition
assert m.study_id.is_unique
q = q.merge(m[["study_id", "dicom_id", "view"]], on="study_id", how="left", validate="m:1")
rep["step3"] = {"questions_without_dicom": int(q.dicom_id.isna().sum())}
meta = meta[["dicom_id", "subject_id", "study_id", "ViewPosition"]].copy()
q = q.merge(meta[["dicom_id", "ViewPosition"]], on="dicom_id", how="left", validate="m:1")
rep["step3"]["ViewPosition_counts_before"] = q.ViewPosition.fillna("NA").replace("", "EMPTY").value_counts().to_dict()
q = q[q.ViewPosition.isin(["PA", "AP"])].copy()
rep["step3"]["kept"] = int(len(q))

# 4. split by subject_id (MIMIC-CXR-JPG)
subj = sp.groupby("subject_id").split.agg(lambda s: sorted(set(s))).reset_index()
cross = subj[subj.split.str.len() > 1]
rep["step4"] = {"split_csv_subjects": int(len(subj)), "split_csv_subjects_in_two_splits": int(len(cross)),
                "split_csv_images_by_split": sp.split.value_counts().to_dict(), "split_csv_studies_by_split": sp.drop_duplicates("study_id").split.value_counts().to_dict()}
subj_split = sp.drop_duplicates("subject_id").set_index("subject_id").split.to_dict()
if len(cross):
    fix = sp[sp.subject_id.isin(cross.subject_id)].drop_duplicates(["subject_id", "study_id"]).groupby(["subject_id", "split"]).size().reset_index(name="n")
    best = fix.sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id").set_index("subject_id").split
    rep["step4"]["moved_subjects"] = int(len(best)); rep["step4"]["moved_examples"] = {str(k): v for k, v in best.head(10).items()}
    subj_split.update(best.to_dict())
q["split_mimic"] = q.subject_id.map(subj_split)
rep["step4"]["questions_without_split"] = int(q.split_mimic.isna().sum())
q = q[q.split_mimic.notna()].copy()
q["split"] = q.split_mimic.map(SPLIT_NAME)
rep["step4"]["mimic_all_split_agrees_with_split_csv_frac"] = float((q.split_mimic == q.merge(m[["study_id", "split"]].rename(columns={"split": "s2"}), on="study_id", how="left").s2.values).mean())
rep["step4"]["pool_by_split"] = q.split.value_counts().to_dict()
rep["step4"]["pool_images_by_split"] = q.groupby("split").dicom_id.nunique().to_dict()
rep["step4"]["pool_subjects_by_split"] = q.groupby("split").subject_id.nunique().to_dict()
rep["step4"]["pool_by_split_type"] = {s: g.question_type.value_counts().to_dict() for s, g in q.groupby("split")}
q["image_path"] = ["files/p" + str(s)[:2] + f"/p{s}/s{st}/{d}.jpg" for s, st, d in zip(q.subject_id, q.study_id, q.dicom_id)]
q = q.drop(columns=["split_mimic"]).reset_index(drop=True)
pool_cols = ["subject_id", "study_id", "dicom_id", "image_path", "question_type", "template_id", "template", "question", "answer", "answer_type", "split", "ViewPosition", "ref_id"]
q[pool_cols].to_parquet(DATA / "pool_single_image_all_splits.parquet", index=False)

# 5. sampling
rng = np.random.default_rng(SEED)


def sample_split(pool: pd.DataFrame, n_q: int, n_img, cap: int, per_type: bool) -> pd.DataFrame:
    pool = pool.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    if n_img is not None:
        imgs = pool.dicom_id.drop_duplicates().sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).head(n_img)
        pool = pool[pool.dicom_id.isin(set(imgs))].reset_index(drop=True)
    strata = pool.groupby(["template_id", "answer"]).size()
    quota = {k: max(1, int(round(n_q * v / len(pool)))) for k, v in strata.items()}
    used = collections.Counter(); per_img = collections.Counter(); chosen = []
    for r in pool.itertuples(index=False):
        k = (r.template_id, r.answer); key = (r.dicom_id, r.question_type) if per_type else r.dicom_id
        if used[k] < quota[k] and per_img[key] < cap and len(chosen) < n_q:
            used[k] += 1; per_img[key] += 1; chosen.append(r)
    return pd.DataFrame(chosen, columns=pool.columns)


items = {}
for s, (n_q, n_img, cap) in TARGET.items():
    pool = q[q.split == s]
    if s == "train":
        per_study = pool.groupby("dicom_id").size().clip(upper=cap)
        order = per_study.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
        cum = order.cumsum(); n_img_tr = int((cum <= int(n_q * 1.15)).sum()) + 1
        pool = pool[pool.dicom_id.isin(set(order.index[:n_img_tr]))]
        df = sample_split(pool, n_q, None, cap, per_type=False)
    else:
        df = sample_split(pool, n_q, n_img, cap, per_type=True)
    df = df.sort_values(["dicom_id", "question_type", "question"]).reset_index(drop=True)
    df.insert(0, "qid", QID_BASE[s] + np.arange(len(df)))
    items[s] = df[["qid"] + pool_cols]
    items[s].to_parquet(DATA / f"items_{s}.parquet", index=False)

# checks
for a in items:
    for b in items:
        if a < b:
            assert not (set(items[a].subject_id) & set(items[b].subject_id)), (a, b)
            assert not (set(items[a].dicom_id) & set(items[b].dicom_id)), (a, b)
allimg = pd.concat([items[s][["dicom_id", "image_path", "split"]] for s in items]).drop_duplicates("dicom_id")
(DATA / "image_list.txt").write_text("\n".join(sorted(allimg.image_path)) + "\n")
rep["step5"] = {s: {"n_questions": int(len(d)), "n_images": int(d.dicom_id.nunique()), "n_studies": int(d.study_id.nunique()), "n_subjects": int(d.subject_id.nunique()),
                    "by_type": d.question_type.value_counts().to_dict(), "by_answer_type": d.answer_type.value_counts().to_dict(),
                    "max_q_per_image": int(d.groupby("dicom_id").size().max()), "ViewPosition": d.drop_duplicates("dicom_id").ViewPosition.value_counts().to_dict()} for s, d in items.items()}
rep["step5"]["total_images"] = int(len(allimg)); rep["step5"]["est_gb_at_1.5MB"] = round(len(allimg) * 1.5 / 1024, 2)
rep["template_ids"] = {v: k for k, v in tmpl_ids.items()}
json.dump(rep, open(AN / "subset_report.json", "w"), indent=1, ensure_ascii=False)

# docs/subset.md
L = ["# Single-image MIMIC/Diff-VQA subset (v7_mimic)", "",
     f"Generated on {rep['provenance']['date']} on {rep['provenance']['host']} by `scripts/01b_make_subset.py`, seed {SEED}. Raw numbers in `analysis/subset_report.json`.",
     f"Fixed vocabulary of open questions (frequency ≥ {MIN_FREQ} in the whole table and ≤ {MAX_WORDS} words): "
     + "; ".join(f"{t}: {rep['step2'][t]['n_vocab']} answers out of {rep['step2'][t]['n_distinct_all']} ({100*rep['step2'][t]['coverage_rows']:.1f} % of rows)" for t in ("location", "level", "type")) + ". List in `data/open_answer_vocab_fixed.json`.", "",
     "## Funnel",
     f"- Rows: {rep['inputs']['pair_questions_rows']}; `difference` removed: {rep['step1']['difference_removed']}; mentions of comparison outside difference: {rep['step1']['comparison_mentions_removed']}.",
     f"- Removed `abnormality`: {rep['step2']['removed_abnormality']}; open `view` (which view): {rep['step2']['removed_view_open']}; open questions outside the vocabulary: {rep['step2']['removed_open_out_of_vocab']}. Kept: {rep['step2']['kept']}.",
     f"- Without dicom in mimic_all: {rep['step3']['questions_without_dicom']}; ViewPosition before the filter: {rep['step3']['ViewPosition_counts_before']}; kept PA/AP: {rep['step3']['kept']}.",
     f"- Split CSV of MIMIC-CXR-JPG: {rep['step4']['split_csv_subjects']} patients, {rep['step4']['split_csv_subjects_in_two_splits']} in two splits" + (f" (moved to the split with more studies: {rep['step4'].get('moved_subjects')})" if len(cross) else "") + f"; questions without split: {rep['step4']['questions_without_split']}; agreement with the `split` of mimic_all: {100*rep['step4']['mimic_all_split_agrees_with_split_csv_frac']:.1f} %.",
     f"- Eligible pool by split (questions): {rep['step4']['pool_by_split']}; images: {rep['step4']['pool_images_by_split']}; patients: {rep['step4']['pool_subjects_by_split']}.",
     f"- Pool by split × type: {rep['step4']['pool_by_split_type']}", "",
     "## Subset sizes", "| split | questions | images | studies | patients | closed | open | max q/image | PA / AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
for s in ("train", "val", "test"):
    r = rep["step5"][s]
    L.append(f"| {s} | {r['n_questions']} | {r['n_images']} | {r['n_studies']} | {r['n_subjects']} | {r['by_answer_type'].get('closed',0)} | {r['by_answer_type'].get('open',0)} | {r['max_q_per_image']} | {r['ViewPosition'].get('PA',0)} / {r['ViewPosition'].get('AP',0)} |")
L += ["", "By type: " + "; ".join(f"{s}: {rep['step5'][s]['by_type']}" for s in ("train", "val", "test")),
      "", f"Distinct images to download: {rep['step5']['total_images']} (≈ {rep['step5']['est_gb_at_1.5MB']} GB at 1,5 MB/image). List: `data/image_list.txt`.",
      "Disjointness: no subject_id and no dicom_id in two splits (assert in the script).", "",
      "## Templates", "| id | type | template | train | val | test |", "|---|---|---|---:|---:|---:|"]
allit = pd.concat(items.values())
tt = allit.groupby(["template_id", "question_type", "template", "split"]).size().unstack("split").fillna(0).astype(int)
for (tid, ty, tp), r in tt.iterrows():
    L.append(f"| {tid} | {ty} | `{tp}` | {r.get('train',0)} | {r.get('val',0)} | {r.get('test',0)} |")
L += ["", "## Template × answer × split (up to 12 answers per template)", "| template | answer | train | val | test |", "|---|---|---:|---:|---:|"]
ta = allit.groupby(["template_id", "answer", "split"]).size().unstack("split").fillna(0).astype(int)
for tid in sorted(allit.template_id.unique()):
    sub = ta.loc[tid].copy(); sub["tot"] = sub.sum(axis=1); sub = sub.sort_values("tot", ascending=False).head(12)
    for ans, r in sub.iterrows():
        L.append(f"| {tid} | {ans} | {r.get('train',0)} | {r.get('val',0)} | {r.get('test',0)} |")
L += ["", "## Prior by template (majority answer)", "| template | train: majority (fraction, n) | test: majority (fraction, n) |", "|---|---|---|"]
for tid in sorted(allit.template_id.unique()):
    row = []
    for s in ("train", "test"):
        d = items[s][items[s].template_id == tid]
        row.append(f"{d.answer.value_counts().idxmax()} ({d.answer.value_counts().iloc[0]/len(d):.3f}, n={len(d)})" if len(d) else "-")
    L.append(f"| {tid} | {row[0]} | {row[1]} |")
(DOCS / "subset.md").write_text("\n".join(L) + "\n")
print(json.dumps({k: rep[k] for k in ("step1", "step2", "step3", "step4", "step5")}, indent=1, ensure_ascii=False, default=str))
