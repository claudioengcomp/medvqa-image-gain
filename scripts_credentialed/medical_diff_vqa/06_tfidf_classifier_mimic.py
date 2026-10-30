"""v7_mimic/scripts/06_m0_mimic.py -- 2026-10-10 -- lasid-gpu1 (so CPU)

Baseline so-pergunta (TFIDF) do MIMIC/Diff-VQA de imagem unica, como harness/isbi_occlusion/heal/tfidf_heal.py (repo
occlusion_study @2b7e731): TF-IDF de palavras (1-2-gramas, min_df=1, lowercase) + LogisticRegression(C=1, max_iter=2000),
sem imagem, treinado nas perguntas FECHADAS (yes/no) e avaliado em todas as perguntas do teste (fechadas pela
predicao; para as abertas a predicao yes/no nunca acerta e o baseline e reportado tambem como resposta majoritaria
por template no treino, "prior"). Agregacao por tipo, por template_id e por answer_type; IC 95 % por bootstrap
POR IMAGEM (dicom_id), 2 000 replicas, semente 20260913.

  python 06_m0_mimic.py --train <parquet com question,answer,template_id,question_type,answer_type> \
      --test data/items_test.parquet --tag subset --out results/TFIDF_question_only_MIMIC_subset.json
"""
from __future__ import annotations
import argparse, collections, json, socket, time
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260913
B = 2000


def _acc(c):
    n = len(c); k = int(sum(c))
    return {"n": n, "n_correct": k, "accuracy": k / n if n else None}


def boot_ci(correct: np.ndarray, groups: np.ndarray, b=B, seed=SEED):
    """IC 95 % por bootstrap de imagens (grupos): reamostra imagens com reposicao, soma acertos/itens."""
    uniq, inv = np.unique(groups, return_inverse=True)
    n_img = len(uniq)
    sum_c = np.bincount(inv, weights=correct, minlength=n_img)
    cnt = np.bincount(inv, minlength=n_img).astype(float)
    rng = np.random.default_rng(seed)
    accs = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n_img, n_img)
        accs[i] = sum_c[idx].sum() / cnt[idx].sum()
    return {"ci95_low": float(np.percentile(accs, 2.5)), "ci95_high": float(np.percentile(accs, 97.5)), "n_images": int(n_img), "B": b, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = pd.read_parquet(a.train); te = pd.read_parquet(a.test)
    trc = tr[tr.answer_type == "closed"]
    assert set(trc.answer) <= {"yes", "no"}, set(trc.answer)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    X = vec.fit_transform(trc.question)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(X, trc.answer)
    pred = clf.predict(vec.transform(te.question))
    correct = (pred == te.answer.values).astype(int)
    # prior: resposta majoritaria por template no TREINO (todas as perguntas, fechadas e abertas)
    maj = tr.groupby("template_id").answer.agg(lambda s: s.value_counts().idxmax()).to_dict()
    glob = tr.answer.value_counts().idxmax()
    prior_pred = np.array([maj.get(t, glob) for t in te.template_id])
    prior_correct = (prior_pred == te.answer.values).astype(int)
    groups = te.dicom_id.values
    def agg(mask_col):
        out = {}
        for k, g in te.groupby(mask_col, sort=True):
            idx = g.index.values
            c = correct[te.index.get_indexer(idx)]; pc = prior_correct[te.index.get_indexer(idx)]
            out[str(k)] = {"tfidf": {**_acc(c), **boot_ci(c, groups[te.index.get_indexer(idx)])},
                           "prior_majority": {**_acc(pc), **boot_ci(pc, groups[te.index.get_indexer(idx)])},
                           "n_images": int(g.dicom_id.nunique())}
        return out
    closed = te.answer_type.values == "closed"
    res = {"model": "TFIDF_TFIDF_LR", "dataset": f"MIMIC-Diff-VQA single-image ({a.tag})", "train_file": a.train, "test_file": a.test,
           "train_rows_closed": int(len(trc)), "train_prior_closed": {k: float(v) for k, v in (trc.answer.value_counts() / len(trc)).items()},
           "n_test": int(len(te)), "n_test_images": int(te.dicom_id.nunique()),
           "overall_closed": {**_acc(correct[closed]), **boot_ci(correct[closed], groups[closed])},
           "overall_all_tfidf": {**_acc(correct), **boot_ci(correct, groups)},
           "overall_all_prior": {**_acc(prior_correct), **boot_ci(prior_correct, groups)},
           "by_answer_type": agg("answer_type"), "by_question_type": agg("question_type"), "by_template": agg("template_id"),
           "template_text": tr.drop_duplicates("template_id").set_index("template_id").template.to_dict() if "template" in tr.columns else {},
           "predictions": {str(q): p for q, p in zip(te.qid, pred)}, "prior_predictions": {str(q): p for q, p in zip(te.qid, prior_pred)},
           "provenance": {"script": "v7_mimic/scripts/06_m0_mimic.py", "like": "harness/isbi_occlusion/heal/tfidf_heal.py @2b7e731", "seed_bootstrap": SEED, "B": B,
                          "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("train_rows_closed", "train_prior_closed", "n_test", "overall_closed", "overall_all_tfidf", "overall_all_prior")}, indent=1))


if __name__ == "__main__":
    main()
