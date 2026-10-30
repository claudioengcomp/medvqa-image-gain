"""19/10/2026 (heal_classifiers_train_only): TF-IDF so-pergunta do HEAL treinado SO no treino (6 882 fechadas sim/nao de vqa_heal_train.arrow), mesma receita de
harness/v7_txt/tfidf_heal_aligned.py (TF-IDF 1-2-gramas, min_df 1, lowercase; LogisticRegression C=1, max_iter 2000, random_state 0); avaliado nos 1 446 avaliaveis.
Sem selecao de checkpoint (modelo convexo). Saida: v7_txt/article_runs/results/TFIDF_question_only_HEAL_TREINO.json."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, sys
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from med_vlm.metrics.process_text import answer_matches, normalize_text
D = json.load(open(f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/texto/HEAL_TREINO.json")); tr, te = D["train"], D["test"]; assert len(tr) == 6882
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True); X = vec.fit_transform([x["question"] for x in tr]); clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(X, [x["answer"] for x in tr])
pr = clf.predict(vec.transform([x["question"] for x in te])); P = [{"qid": x["qid"], "predicted": str(p), "gold": x["refs"][0], "correct": bool(answer_matches(normalize_text(str(p)), [normalize_text(r) for r in x["refs"]]))} for p, x in zip(pr, te)]
k = sum(p["correct"] for p in P); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"model": "TFIDF_question_only", "dataset": "HEAL", "condition": "C0", "accuracy": k / len(P), "n": len(P), "n_correct": k, "train_rows": len(tr), "protocolo": "so treino (heal_classifiers_train_only)",
           "provenance": {"script": __file__, "script_sha256": sha(__file__), "dados": "v7_txt/article_runs/data/texto/HEAL_TREINO.json", "receita": "igual a tfidf_heal_aligned.py"}, "predictions": P},
          open(f"{_MEDVQA_ROOT}/v7_txt/article_runs/results/TFIDF_question_only_HEAL_TREINO.json", "w"), indent=1); print("acc %.4f n %d treino %d" % (k / len(P), len(P), len(tr)))
