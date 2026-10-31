"""TFIDF question-only of the ALIGNED HEAL (2026-10-10, requested by the candidate): same classifier as harness/isbi_occlusion/heal/tfidf_heal.py
(TF-IDF 1-2-grams, min_df 1, lowercase; LogisticRegression C=1, max_iter 2000, random_state 0) and the same training set
(the 7 641 closed-form yes/no questions from the VinDr training of HEAL). The parquet heal_annotations_train.parquet is no longer on disk
nor in the Drive; the same 7 641 rows (question, normalized answer) are read from the training + validation arrows
(vqa_heal_train.arrow + vqa_heal_val.arrow = split D2 of the same annotations; count and prior checked against the
published tfidf_by_type_heal.json: 7 641 rows, prior no 0,5112 / yes 0,4888). Predicts exactly the 1 446 evaluable qids from the
current inventory; output in the format of the _C0.json files + aggregates + provenance."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, platform, time
from harness.interventions.arrow_io import read_arrow
from harness.results_schema import validate_results_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn
P = f"{_MEDVQA_ROOT}/doutorado/medvqa-occlusion-study"; ARR = [f"{P}/data/finetune_arrows/vqa_heal_train.arrow", f"{P}/data/finetune_arrows/vqa_heal_val.arrow"]
INV = f"{P}/results/isbi_occlusion/inventory_heal_test.json"; PUB = f"{P}/results/isbi_occlusion/tfidf_by_type_heal.json"; OUT = f"{_MEDVQA_ROOT}/v7_txt/results/TFIDF_question_only_HEAL_aligned.json"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
qs, ys = [], []
for a in ARR:
    df = read_arrow(a)
    for _, r in df.iterrows():
        for q, ans in zip(r["questions"], r["answers"]):
            g = str(list(ans)[0]).strip().lower()
            if g in ("yes", "no"): qs.append(q); ys.append(g)
pub = json.load(open(PUB)); prior = collections.Counter(ys); assert len(qs) == pub["train_rows"], (len(qs), pub["train_rows"])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True); X = vec.fit_transform(qs); clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(X, ys)
inv = json.load(open(INV)); ev = [q for q in inv["questions"] if q["evaluable"]]; pred = clf.predict(vec.transform([q["question"] for q in ev]))
preds = [{"qid": int(q["qid"]), "predicted": str(p), "gold": q["answer"], "correct": bool(p == q["answer"])} for q, p in zip(ev, pred)]
byc = collections.defaultdict(list); bya = collections.defaultdict(list)
for q, p in zip(ev, preds): byc[q["question_class"]].append(p["correct"]); bya[q["anatomy"]].append(p["correct"])
agg = lambda v: {"n": len(v), "n_correct": int(sum(v)), "accuracy": sum(v) / len(v)}
rec = {"model": "TFIDF_question_only", "dataset": "HEAL", "condition": "C0", "accuracy": sum(p["correct"] for p in preds) / len(preds), "n": len(preds), "n_correct": sum(p["correct"] for p in preds), "predictions": preds,
       "aggregates": {"train_rows": len(qs), "train_prior": {k: v / len(qs) for k, v in prior.items()}, "by_class": {k: agg(v) for k, v in sorted(byc.items())}, "by_anatomy": {k: agg(v) for k, v in sorted(bya.items())}, "published_1512": {"train_rows": pub["train_rows"], "train_prior": pub["train_prior"], "by_class": pub["by_class"]["evaluable"]}},
       "provenance": {"recipe": "igual a scripts_credentialed.heal.tfidf_heal.train_and_eval (TF-IDF 1-2-gramas + LogisticRegression C=1, random_state 0)", "train_source": ARR, "train_source_sha256": [sha(a) for a in ARR], "train_rows": len(qs), "note": "parquet heal_annotations_train.parquet missing from disk and from Drive; the same 7641 closed-form questions read from the train+val arrows (D2 split of the same annotations); prediction on the 1446 evaluable qids of the current inventory",
                      "inventory": INV, "inventory_sha256": sha(INV), "sklearn": sklearn.__version__, "python": platform.python_version(), "host": platform.node(), "stamped": time.strftime("%Y-%m-%d %H:%M:%S")}}
validate_results_dict(rec); json.dump(rec, open(OUT, "w"), indent=1)
print("train rows", len(qs), "prior", rec["aggregates"]["train_prior"], "| n", rec["n"], "acc %.4f" % rec["accuracy"], {k: round(v["accuracy"], 4) for k, v in rec["aggregates"]["by_class"].items()}, "| published 1512:", {k: round(v["accuracy"], 4) for k, v in pub["by_class"]["evaluable"].items()})
