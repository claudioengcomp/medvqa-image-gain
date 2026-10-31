"""TFIDF_question_only_VQARAD_orig (15/10/2026, the candidate): question-only classifier retrained on the 2 758 training set of the original VQA-RAD
(M3AE/data/finetune_data/vqa_rad/trainset.json, the same list as the v7_txt/data/vqarad_orig arrows), on CPU.
Recipe = harness/baselines/question_only.py (standard TfidfVectorizer + LogisticRegression max_iter 1000; answer str().strip().lower();
correct = exact match), in two variants:
  principal: training on all 2 758 (open and closed-form), prediction on the 451 of the test set; also aggregated on the 272 closed-form;
  so_fechadas: training on the closed-form training items (the published recipe), prediction on the 272 closed-form -- checked item by item against
               p0/results/TFIDF_question_only_VQARAD_C0.json.
Output: v7_txt/results/TFIDF_question_only_VQARAD_orig.json (format of the _C0.json files + aggregates + provenance). Does not overwrite."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, platform, time
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from harness.results_schema import validate_results_dict
D = f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_data/vqa_rad"; PUB = f"{_MEDVQA_ROOT}/doutorado/medvqa-main/results/TFIDF_question_only_VQARAD_C0.json"
OUT = f"{_MEDVQA_ROOT}/v7_txt/results/TFIDF_question_only_VQARAD_orig.json"
assert not os.path.exists(OUT), "already exists, I will not overwrite"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def load(split):
    return [{"qid": s["qid"], "question": s["question"], "answer": str(s["answer"]).strip().lower(), "closed": str(s["answer_type"]).strip().upper() == "CLOSED"} for s in json.load(open(f"{D}/{split}set.json"))]
tr, te = load("train"), load("test"); assert len(tr) == 2758 and len(te) == 451
def fit(rows): return Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]).fit([r["question"] for r in rows], [r["answer"] for r in rows])
def run(model, rows):
    pr = model.predict([r["question"] for r in rows])
    return [{"qid": r["qid"], "predicted": str(p), "gold": r["answer"], "correct": bool(p == r["answer"]), "closed": r["closed"]} for r, p in zip(rows, pr)]
agg = lambda v: {"n": len(v), "n_correct": int(sum(x["correct"] for x in v)), "accuracy": sum(x["correct"] for x in v) / len(v)}
main = run(fit(tr), te)
trc = [r for r in tr if r["closed"]]; tec = [r for r in te if r["closed"]]; clo = run(fit(trc), tec)
pub = {p["qid"]: p for p in json.load(open(PUB))["predictions"]}
conf = {"n_publicado": len(pub), "mesmos_qids": sorted(pub) == sorted(p["qid"] for p in clo), "predicoes_diferentes_do_publicado": sum(1 for p in clo if pub.get(p["qid"], {}).get("predicted") != p["predicted"]),
        "acc_publicada": json.load(open(PUB))["accuracy"]}
rec = {"model": "TFIDF_question_only", "dataset": "VQA-RAD", "condition": "C0", "accuracy": agg(main)["accuracy"], "n": len(main), "n_correct": agg(main)["n_correct"],
       "predictions": [{k: v for k, v in p.items() if k != "closed"} for p in main],
       "aggregates": {"todas_451": agg(main), "fechadas_272_do_classificador_principal": agg([p for p in main if p["closed"]]), "abertas_179_do_classificador_principal": agg([p for p in main if not p["closed"]]),
                      "so_fechadas_receita_publicada": {**agg(clo), "treino_n": len(trc), "conferencia_publicado": conf, "predictions": [{k: v for k, v in p.items() if k != "closed"} for p in clo]},
                      "treino_n": len(tr), "treino_fechadas": len(trc), "classes_de_resposta_no_treino": len({r["answer"] for r in tr})},
       "provenance": {"line": "TFIDF_question_only_VQARAD_orig", "recipe": "harness/baselines/question_only.py (TfidfVectorizer() + LogisticRegression(max_iter=1000)); answer strip().lower(); correctness by exact match",
                      "train_json": f"{D}/trainset.json", "train_json_sha256": sha(f"{D}/trainset.json"), "test_json": f"{D}/testset.json", "test_json_sha256": sha(f"{D}/testset.json"),
                      "sklearn": sklearn.__version__, "python": platform.python_version(), "host": platform.node(), "stamped": time.strftime("%Y-%m-%d %H:%M:%S")}}
validate_results_dict(rec); json.dump(rec, open(OUT, "w"), indent=1)
a = rec["aggregates"]; print("451: %.4f | 272 closed-form (main): %.4f | 179 open: %.4f | closed-only: %.4f (published %.4f; %d different predictions; same qids %s) | classes %d" % (
    a["todas_451"]["accuracy"], a["fechadas_272_do_classificador_principal"]["accuracy"], a["abertas_179_do_classificador_principal"]["accuracy"], a["so_fechadas_receita_publicada"]["accuracy"], conf["acc_publicada"], conf["predicoes_diferentes_do_publicado"], conf["mesmos_qids"], a["classes_de_resposta_no_treino"]))
