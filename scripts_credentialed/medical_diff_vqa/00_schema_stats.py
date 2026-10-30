"""v7_mimic/scripts/00_schema_stats.py -- 2026-10-10 -- statistics of the Medical-Diff-VQA 1.0.1 tables (CPU only)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, json, re, socket, time
from pathlib import Path
import pandas as pd
RAW = Path(f"{_MEDVQA_ROOT}/doutorado/data/raw/medical-diff-vqa/1.0.1")
OUT = Path(f"{_MEDVQA_ROOT}/v7_mimic/analysis"); OUT.mkdir(exist_ok=True)
q = pd.read_csv(RAW / "mimic_pair_questions.csv", dtype={"study_id": "int64", "subject_id": "int64", "ref_id": "int64"}, keep_default_na=False)
m = pd.read_csv(RAW / "mimic_all.csv", dtype={"study_id": "int64", "subject_id": "int64"}, keep_default_na=False)
res = {"pair_questions": {"n_rows": len(q), "columns": {c: str(t) for c, t in q.dtypes.items()},
                          "by_type": q.question_type.value_counts().to_dict(),
                          "by_type_split": {t: g.split.value_counts().to_dict() for t, g in q.groupby("question_type")},
                          "n_studies": int(q.study_id.nunique()), "n_subjects": int(q.subject_id.nunique()),
                          "ref_eq_study": int((q.ref_id == q.study_id).sum()),
                          "examples": {t: g.head(3)[["study_id", "subject_id", "ref_id", "question", "answer", "split"]].astype(str).values.tolist() for t, g in q.groupby("question_type")},
                          "answers": {}, "questions_distinct": {}, "comparison_words_non_difference": {}},
       "mimic_all": {"n_rows": len(m), "columns": {c: str(t) for c, t in m.dtypes.items()}, "n_studies": int(m.study_id.nunique()), "n_subjects": int(m.subject_id.nunique()),
                     "n_dicom": int(m.dicom_id.nunique()), "rows_per_study_max": int(m.groupby("study_id").size().max()),
                     "view": m.view.value_counts().to_dict(), "split": m.split.value_counts().to_dict(),
                     "subject_in_two_splits": int((m.groupby("subject_id").split.nunique() > 1).sum()),
                     "examples": m.head(3).astype(str).values.tolist()}}
cmp_re = re.compile(r"\b(compar|previous|prior|reference|change|earlier|last|interval|since)\w*", re.I)
for t, g in q.groupby("question_type"):
    vc = g.answer.value_counts()
    res["pair_questions"]["answers"][t] = {"n_distinct": int(len(vc)), "top30": vc.head(30).to_dict(),
                                           "yes_no_frac": float(g.answer.isin(["yes", "no"]).mean()),
                                           "n_words_answer_median": float(g.answer.str.split().str.len().median()),
                                           "n_words_answer_max": int(g.answer.str.split().str.len().max())}
    qvc = g.question.value_counts()
    res["pair_questions"]["questions_distinct"][t] = {"n_distinct": int(len(qvc)), "top15": qvc.head(15).to_dict()}
    if t != "difference":
        hit = g.question.str.contains(cmp_re)
        res["pair_questions"]["comparison_words_non_difference"][t] = {"n": int(hit.sum()), "examples": g.loc[hit, "question"].head(5).tolist()}
# entidades (nomes) do all_diseases.json
ent = collections.Counter(); noent = collections.Counter()
for rec in json.load(open(RAW / "all_diseases.json")):
    for e in rec.get("entity", {}): ent[e] += 1
    for e in rec.get("no_entity", []) or []: noent[e] += 1
res["all_diseases"] = {"n_records": sum(1 for _ in ent) and None, "n_entities_distinct": len(ent), "entities_top40": dict(ent.most_common(40)), "no_entity_top20": dict(noent.most_common(20))}
json.dump({"entities": dict(ent.most_common()), "no_entity": dict(noent.most_common())}, open(OUT / "entities.json", "w"), indent=1)
# juncao questoes x mimic_all por study_id
j = q.merge(m[["study_id", "dicom_id", "view", "split"]].rename(columns={"split": "split_mimic_all"}), on="study_id", how="left")
res["join"] = {"questions_without_mimic_all_row": int(j.dicom_id.isna().sum()), "split_agree_frac": float((j.split == j.split_mimic_all).mean()),
               "view_by_type": {t: g.view.value_counts().to_dict() for t, g in j.groupby("question_type")}}
res["provenance"] = {"script": "v7_mimic/scripts/00_schema_stats.py", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname()}
json.dump(res, open(OUT / "schema_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: res["pair_questions"][k] for k in ("n_rows", "by_type", "n_studies", "n_subjects", "ref_eq_study")}))
print(json.dumps({k: res["mimic_all"][k] for k in ("n_rows", "n_studies", "n_dicom", "rows_per_study_max", "view", "split", "subject_in_two_splits")}))
print(json.dumps(res["join"]["questions_without_mimic_all_row"]), res["join"]["split_agree_frac"])
