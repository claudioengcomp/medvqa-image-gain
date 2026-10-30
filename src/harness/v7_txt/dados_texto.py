"""17/10/2026 (article_runs, question_only_sets): question-only sets per benchmark, IDENTICAL to those of the TF-IDF classifier (TFIDF) of each benchmark, and the baseline
of the majority answer (2a). Per benchmark: closed training set (the TF-IDF one: Kvasir = all 38 211 of the cleaned training set; SLAKE = closed-form in
English from train.json; HEAL = closed yes/no from training + validation; MIMIC = closed-form from items_train; VQA-RAD = closed-form from
trainset.json via harness.baselines.question_only.load_closed_qa_pairs), closed validation where it exists (HEAL: there is none, the validation
is within the training set) and closed test with the golds. Asserts: training n = the one recorded in the TFIDF; test qids and golds = those of the TFIDF.
Rule: answer_matches(normalize_text(...)) (med_vlm.metrics.process_text), checked against the correct field of each TFIDF.
Majority: most frequent answer in the closed-form TRAINING set (HEAL: training only; the training+validation variant goes along, for checking).
Outputs: <out>/data/texto/<BASE>.json, <out>/results/MAJ_question_only_<BASE>[_trainval].json. No interpretation."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, collections, hashlib, json, os, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree"); sys.path.insert(0, f"{_MEDVQA_ROOT}/doutorado/medvqa-main")
import pandas as pd, pyarrow as pa
from med_vlm.metrics.process_text import answer_matches, normalize_text
from harness.baselines.question_only import load_closed_qa_pairs
W = f"{_MEDVQA_ROOT}"; P0 = f"{W}/doutorado/medvqa-main"; PL = f"{W}/doutorado/medvqa-occlusion-study"; M3 = f"{W}/doutorado/M3AE/data/finetune_data"; MI = f"{W}/v7_mimic"
TFIDF = {"KVASIR": f"{PL}/results/isbi_occlusion/TFIDF_question_only_KVASIR_C0.json", "SLAKE": f"{W}/v6_medida/design/tfidf_by_type/SLAKE_closed_recomputed.json", "HEAL": f"{W}/v7_txt/results/TFIDF_question_only_HEAL_aligned.json",
      "MIMIC": f"{MI}/results/TFIDF_question_only_MIMIC_subset.json", "VQARAD": f"{P0}/results/TFIDF_question_only_VQARAD_C0.json"}
NTRAIN = {"KVASIR": 38211, "SLAKE": 1943, "HEAL": 7641, "MIMIC": 7800, "VQARAD": 1643}
QD = f"{W}/v7_txt/article_runs/data/vqarad_qlevel"; TFIDF["VQARAD_QLEVEL"] = f"{W}/v7_txt/article_runs/results/TFIDF_question_only_VQARAD_QLEVEL.json"; NTRAIN["VQARAD_QLEVEL"] = 1474   # 19/10 (vqarad_question_split)
NTRAIN["HEAL_TREINO"] = 6882; TFIDF["HEAL_TREINO"] = TFIDF["HEAL"]   # 19/10 (heal_classifiers_train_only): HEAL training only, validation = val
RD = f"{W}/v7_txt/article_runs/data/vqarad_rediv"; TFIDF["VQARAD_REDIV"] = f"{W}/v7_txt/article_runs/results/TFIDF_question_only_VQARAD_REDIV.json"; NTRAIN["VQARAD_REDIV"] = 1469   # 17/10 (vqarad_image_split): VQA-RAD redivided by image
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def arrow_rows(p):
    with pa.memory_map(p, "rb") as s: t = pa.ipc.open_file(s).read_all()
    return t.to_pylist()
def flat(p, keep=lambda g: True):
    out = []
    for r in arrow_rows(p):
        for q, a, qid in zip(r["questions"], r["answers"], r["question_id"]):
            refs = [str(x).strip().lower() for x in (a if isinstance(a, list) else [a])]
            if keep(refs[0]): out.append({"qid": int(qid), "question": str(q), "answer": refs[0], "refs": refs})
    return out
def slake_vqarad(path): return [{"qid": int(x["qid"]), "question": x["question"], "answer": x["answer"], "refs": [x["answer"]]} for x in load_closed_qa_pairs(path)]
def mimic(split):
    d = pd.read_parquet(f"{MI}/data/items_{split}.parquet"); d = d[d.answer_type == "closed"]
    return [{"qid": int(r.qid), "question": str(r.question), "answer": str(r.answer).strip().lower(), "refs": [str(r.answer).strip().lower()]} for r in d.itertuples()]
def build(b):
    src = {}
    if b == "KVASIR":
        A = f"{P0}/data/finetune_arrows"; tr = flat(f"{A}/vqa_kvasir_train_clean.arrow"); va = flat(f"{A}/vqa_kvasir_val_clean.arrow"); te = flat(f"{A}/vqa_kvasir_test_clean.arrow"); src = {"train": f"{A}/vqa_kvasir_train_clean.arrow", "val": f"{A}/vqa_kvasir_val_clean.arrow", "test": f"{A}/vqa_kvasir_test_clean.arrow"}; tr_only = tr
    elif b in ("SLAKE", "VQARAD"):
        f = {"SLAKE": ("slack/train.json", "slack/validate.json", "slack/test.json"), "VQARAD": ("vqa_rad/trainset.json", "vqa_rad/valset.json", "vqa_rad/testset.json")}[b]
        tr, va, te = (slake_vqarad(f"{M3}/{x}") for x in f); src = dict(zip(("train", "val", "test"), (f"{M3}/{x}" for x in f))); tr_only = tr
    elif b == "VQARAD_QLEVEL":
        tr, va, te = (slake_vqarad(f"{QD}/{x}set.json") for x in ("train", "val", "test")); src = {k: f"{QD}/{k}set.json" for k in ("train", "val", "test")}; tr_only = tr
    elif b == "VQARAD_REDIV":
        tr, va, te = (slake_vqarad(f"{RD}/{x}set.json") for x in ("train", "val", "test")); src = {k: f"{RD}/{k}set.json" for k in ("train", "val", "test")}; tr_only = tr
    elif b == "HEAL_TREINO":
        A = f"{PL}/data/finetune_arrows"; yn = lambda g: g in ("yes", "no"); tr = flat(f"{A}/vqa_heal_train.arrow", yn); tr_only = tr; va = flat(f"{A}/vqa_heal_val.arrow", yn)
        inv = json.load(open(f"{PL}/results/isbi_occlusion/inventory_heal_test.json")); te = [{"qid": int(q["qid"]), "question": q["question"], "answer": str(q["answer"]).strip().lower(), "refs": [str(q["answer"]).strip().lower()]} for q in inv["questions"] if q["evaluable"]]
        src = {"train": f"{A}/vqa_heal_train.arrow", "val": f"{A}/vqa_heal_val.arrow", "test": f"{PL}/results/isbi_occlusion/inventory_heal_test.json"}
    elif b == "HEAL":
        A = f"{PL}/data/finetune_arrows"; yn = lambda g: g in ("yes", "no"); tr_only = flat(f"{A}/vqa_heal_train.arrow", yn); tr = tr_only + flat(f"{A}/vqa_heal_val.arrow", yn); va = []
        inv = json.load(open(f"{PL}/results/isbi_occlusion/inventory_heal_test.json")); te = [{"qid": int(q["qid"]), "question": q["question"], "answer": str(q["answer"]).strip().lower(), "refs": [str(q["answer"]).strip().lower()]} for q in inv["questions"] if q["evaluable"]]
        src = {"train": [f"{A}/vqa_heal_train.arrow", f"{A}/vqa_heal_val.arrow"], "val": None, "test": f"{PL}/results/isbi_occlusion/inventory_heal_test.json"}
    elif b == "MIMIC":
        tr, va, te = mimic("train"), mimic("val"), mimic("test"); src = {k: f"{MI}/data/items_{s}.parquet" for k, s in (("train", "train"), ("val", "val"), ("test", "test"))}; tr_only = tr
    assert len(tr) == NTRAIN[b], (b, len(tr), NTRAIN[b])
    tfidf = json.load(open(TFIDF[b])); P = tfidf["predictions"]
    if isinstance(P, dict):   # MIMIC: qid -> predicao; gold do items_test
        g = {x["qid"]: x for x in te}; P = [{"qid": int(q), "predicted": p, "gold": g[int(q)]["answer"], "correct": None} for q, p in P.items() if int(q) in g]
        ok_ref = {int(q): bool(answer_matches(normalize_text(str(p["predicted"])), [normalize_text(g[int(q)]["answer"])])) for q, p in ((p["qid"], p) for p in P)}
    else: ok_ref = {int(p["qid"]): bool(p["correct"]) for p in P}
    T = {x["qid"]: x for x in te}; mq = {int(p["qid"]) for p in P}
    assert set(T) == mq, (b, len(set(T)), len(mq), len(set(T) - mq), len(mq - set(T)))
    same = sum(bool(answer_matches(normalize_text(str(p["predicted"])), [normalize_text(r) for r in T[int(p["qid"])]["refs"]])) == ok_ref[int(p["qid"])] for p in P)
    return {"base": b, "fontes": src, "fontes_sha256": {k: ([sha(x) for x in v] if isinstance(v, list) else (sha(v) if v else None)) for k, v in src.items()}, "tfidf": TFIDF[b], "tfidf_sha256": sha(TFIDF[b]),
            "n": {"train": len(tr), "train_so_treino": len(tr_only), "val": len(va), "test": len(te)}, "regua_confere_tfidf": {"iguais": same, "n": len(P)}, "train": tr, "train_so_treino": tr_only, "val": va, "test": te}
def majority(D, which):
    c = collections.Counter(x["answer"] for x in D[which]); top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]   # empate: ordem alfabetica
    preds = [{"qid": x["qid"], "predicted": top, "gold": x["refs"][0] if len(x["refs"]) == 1 else x["refs"], "correct": bool(answer_matches(normalize_text(top), [normalize_text(r) for r in x["refs"]]))} for x in D["test"]]
    return top, c, preds
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--bases", default="KVASIR,SLAKE,HEAL,MIMIC,VQARAD"); a = ap.parse_args()
    os.makedirs(f"{a.out}/data/texto", exist_ok=True); os.makedirs(f"{a.out}/results", exist_ok=True)
    for b in a.bases.split(","):
        t0 = time.time(); D = build(b); json.dump(D, open(f"{a.out}/data/texto/{b}.json", "w"))
        for which, suf in (("train_so_treino", ""),) + ((("train", "_trainval"),) if b == "HEAL" else ()):
            top, c, preds = majority(D, which); n = len(preds); k = sum(p["correct"] for p in preds)
            rec = {"model": "MAJ_question_only", "dataset": b, "condition": "C0", "accuracy": k / n, "n": n, "n_correct": k, "majoritaria": top, "contagem_top5_treino": c.most_common(5), "n_treino": len(D[which]),
                   "provenance": {"script": os.path.abspath(__file__), "script_sha256": sha(__file__), "treino": which, "fontes": D["fontes"], "fontes_sha256": D["fontes_sha256"], "regra": "resposta mais frequente nas fechadas do treino; empate por ordem alfabetica; regua answer_matches(normalize_text)", "t": time.strftime("%Y-%m-%d %H:%M:%S")}, "predictions": preds}
            json.dump(rec, open(f"{a.out}/results/MAJ_question_only_{b}{suf}.json", "w"), indent=1)
            print(b, which, "n", D["n"], "regua x TFIDF", D["regua_confere_tfidf"], "| majoritaria", repr(top), "acc %.4f" % (k / n), "| %.1f s" % (time.time() - t0), flush=True)
