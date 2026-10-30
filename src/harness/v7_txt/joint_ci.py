"""19/10/2026 (joint_ci): IC condicional (media das 3 sementes por replica) e IC conjunto (replica de imagens + 3 sementes sorteadas com reposicao, numpy RandomState(20260913))
para G_enc, G_ft, G, A - A_swap, A_swap - A_text, A_text - PubMedBERT, A_text - TF-IDF e A_text - baseline por texto exato de pergunta, nas cinco bases do artigo (fechadas).
HEAL: classificadores no protocolo so-treino (heal_classifiers_train_only); troca: HEAL e Medical-Diff-VQA refeitas (image_swap_v2). Bootstrap por imagem 2 000, seed 20260913. Saida: analysis/joint_ci.json."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, os, sys, time
os.environ["ARTICLE2_RUNS"] = "1"; os.environ["TROCA_V2"] = "1"
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, RESULTS_DIR, V7, TXT_BENCH_M3AE, cells, classes, preds, tfidf_fechadas
from harness.v7_txt.analise_troca import swap_file
from harness.stats.bootstrap import ImageBootstrap
from harness.interventions.arrow_io import read_arrow
from med_vlm.metrics.process_text import answer_matches, normalize_text as NT
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; PJR = f"{PJ}/results"; FAM = ["QWEN2VL", "SMOLVLM", "M3AE"]; ART = ["KVASIR", "SLAKE", "HEAL", "MIMIC", "VQARAD_REDIV", "VQARAD_QLEVEL", "VQARAD_RADORIG"]   # 19/10: tres divisoes do VQA-RAD
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); SEEDS = np.random.RandomState(20260913).randint(0, 3, size=(2000, 3))
PAT = ["is there", "are there", "have all polyps", "how many", "what type of", "what is the size", "where in the image", "does this image contain any finding", "what color is the anatomical landmark", "what color is the abnormality"]
def baseline_string(b):
    D = json.load(open(f"{PJ}/data/texto/{ {'HEAL': 'HEAL_TREINO', 'VQARAD_RADORIG': 'VQARAD'}.get(b, b) }.json")); tr, te = D["train_so_treino"], D["test"]
    tpl = lambda q: next((p for p in PAT if q.strip().lower().startswith(p)), None); byq = collections.defaultdict(collections.Counter); byt = collections.defaultdict(collections.Counter); g = collections.Counter()
    for x in tr: byq[NT(x["question"])][x["answer"]] += 1; g[x["answer"]] += 1; (byt[tpl(x["question"])].update([x["answer"]]) if b == "KVASIR" else None)
    top = lambda c: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]; out = {}; fonte = collections.Counter()
    for x in te:
        k = NT(x["question"])
        if k in byq: p = top(byq[k]); fonte["texto_exato"] += 1
        elif b == "KVASIR" and tpl(x["question"]) in byt: p = top(byt[tpl(x["question"])]); fonte["template"] += 1
        else: p = top(g); fonte["global"] += 1
        out[x["qid"]] = bool(answer_matches(NT(p), [NT(r) for r in x["refs"]]))
    return out, {"n_treino": len(tr), "fonte": dict(fonte), "cobertura_texto_exato_pct": 100 * fonte["texto_exato"] / len(te), "n_teste": len(te)}
def bvec(bs, img, qs, d):
    idx = np.array([bs.index[img[q]] for q in qs]); n = len(bs.ids); return (bs.W @ np.bincount(idx, weights=d, minlength=n)) / np.maximum(bs.W @ np.bincount(idx, minlength=n).astype(float), 1e-9)
def stat(ds):   # ds: lista (por semente) de (diferencas por item, qids)
    vs = np.vstack([v for _, v in ds]).T; pts = [d.mean() for d, _ in ds]; cond = vs.mean(axis=1); joint = np.take_along_axis(vs, SEEDS, axis=1).mean(axis=1)
    return {"media": float(np.mean(pts)), "dp_entre_sementes": float(np.std(pts, ddof=1)), "ic_condicional": [float(np.percentile(cond, 2.5)), float(np.percentile(cond, 97.5))], "ic_conjunto": [float(np.percentile(joint, 2.5)), float(np.percentile(joint, 97.5))]}
out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "bases": {}, "baseline_string": {}}
for b in ART:
    df = read_arrow(ARROWS[b]); cls, img = classes(b, df); allq = sorted(q for q in img if cls[q].get("answer_type") == "closed"); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
    TF = preds(f"{PJR}/TFIDF_question_only_HEAL_TREINO.json") if b == "HEAL" else tfidf_fechadas(b)[0]; PS, info = baseline_string(b); out["baseline_string"][b] = info
    B = {}
    for f in FAM:
        acc = collections.defaultdict(list)
        for s in range(3):
            c = cells(b, f, s); tl = f"{f}_LORA_TXT_{b}_seed{s}" if f != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(b, b)}_seed{s}"; T = preds(f"{RESULTS_DIR.get(b, V7 + '/results')}/{tl}_TXT.json")
            F = preds(c["frozen"]); A = preds(c["adapted"]) if c["adapted"] else F; _sp = swap_file(b, f, s, c["adapted_line"] or c["frozen_line"]); S = preds(_sp) if os.path.exists(_sp) else None
            BT = preds(f"{PJR}/BERT2_question_only_HEAL_TREINO_seed{s}.json") if b == "HEAL" else (preds(f"{PJR}/BERT2_question_only_{b}_seed{s}.json") if b == "VQARAD_QLEVEL" else preds(f"{PJR}/BERT_question_only_{ {'VQARAD_RADORIG': 'VQARAD'}.get(b, b) }_seed{s}.json"))
            qs = [q for q in allq if q in T and q in F and q in A]; v = lambda X: np.array([X[q] for q in qs], float)
            pares = {"G": (v(A) - v(T)), "A_text_menos_PubMedBERT": v(T) - v(BT), "A_text_menos_TFIDF": v(T) - v(TF), "A_text_menos_exact_text_baseline": v(T) - v(PS)}
            if f != "M3AE": pares["G_enc"] = v(F) - v(T); pares["G_ft"] = v(A) - v(F)
            for k, d in pares.items(): acc[k].append((d, bvec(bs, img, qs, d)))
            if S is None: continue   # 19/10: sem troca (VLM da divisao oficial)
            qw = [q for q in qs if q in S]; vw = lambda X: np.array([X[q] for q in qw], float)
            acc["A_menos_A_swap"].append((vw(A) - vw(S), bvec(bs, img, qw, vw(A) - vw(S)))); acc["A_swap_menos_A_text"].append((vw(S) - vw(T), bvec(bs, img, qw, vw(S) - vw(T))))
        B[f] = {k: stat(v) for k, v in acc.items() if len(v) == 3}
    out["bases"][b] = B; print(b, "ok", flush=True)
out["script"] = os.path.abspath(__file__); out["script_sha256"] = sha(__file__)
json.dump(out, open(f"{PJ}/analysis/joint_ci.json", "w"), indent=1); print("FIM")
