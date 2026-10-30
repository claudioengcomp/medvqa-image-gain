"""20/10/2026 (m3ae_frozen_vit): M3AE com o encoder visual congelado -- decomposicao com paridade das familias VLM, nas fechadas.
A_text = M3AE so-texto publicado (cinza no treino/val/teste); A_frozen = M3AE_FROZEN C0; A_ft = M3AE publicado (fine-tune completo) C0.
G_enc = A_frozen - A_text; G_ft = A_ft - A_frozen; G = A_ft - A_text. Cinza = M3AE_FROZEN com imagem cinza so no teste (C1). Troca = M3AE_FROZEN no arrow
de troca das linhas publicadas (C2; HEAL/MIMIC C2v2 do image_swap_v2). Baselines: PubMedBERT, TF-IDF, majoritaria, majoritaria por texto exato de pergunta (HEAL: so-treino).
IC: bootstrap por imagem 2 000, seed 20260913 (as mesmas replicas); IC condicional (media das 3 sementes por replica) e IC conjunto (replica + 3 sementes com
reposicao, RandomState(20260913)), como em joint_ci.py; IC por execucao (percentis da replica da semente). Base com menos de 3 sementes prontas: so numeros
por semente (parcial). Colunas VLM ao lado: joint_ci.json (G_enc / G_ft / G / troca, IC conjunto) e acuracias recalculadas nas mesmas fechadas. Saida: analysis/m3ae_frozen_vit.json."""
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
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; PJR = f"{PJ}/results"; FR = os.environ.get("M3AE_FROZEN_VIT_FR", f"{PJ}/m3ae_frozen/results")
FL = f"{PJ}/m3ae_frozen/logs"
OUTJ = os.environ.get("M3AE_FROZEN_VIT_OUT", f"{PJ}/analysis/m3ae_frozen_vit.json")   # 20/10: FR/OUT sobrescreviveis so para o teste de caminho
BASES = os.environ.get("M3AE_FROZEN_VIT_BASES", "SLAKE,VQARAD_REDIV,HEAL,MIMIC,KVASIR,VQARAD_QLEVEL").split(",")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); SEEDS = np.random.RandomState(20260913).randint(0, 3, size=(2000, 3))
PAT = ["is there", "are there", "have all polyps", "how many", "what type of", "what is the size", "where in the image", "does this image contain any finding", "what color is the anatomical landmark", "what color is the abnormality"]
def baseline_string(b):   # identico a joint_ci.py
    D = json.load(open(f"{PJ}/data/texto/{ {'HEAL': 'HEAL_TREINO', 'VQARAD_RADORIG': 'VQARAD'}.get(b, b) }.json")); tr, te = D["train_so_treino"], D["test"]
    tpl = lambda q: next((p for p in PAT if q.strip().lower().startswith(p)), None); byq = collections.defaultdict(collections.Counter); byt = collections.defaultdict(collections.Counter); g = collections.Counter()
    for x in tr: byq[NT(x["question"])][x["answer"]] += 1; g[x["answer"]] += 1; (byt[tpl(x["question"])].update([x["answer"]]) if b == "KVASIR" else None)
    top = lambda c: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]; out = {}
    for x in te:
        k = NT(x["question"])
        if k in byq: p = top(byq[k])
        elif b == "KVASIR" and tpl(x["question"]) in byt: p = top(byt[tpl(x["question"])])
        else: p = top(g)
        out[x["qid"]] = bool(answer_matches(NT(p), [NT(r) for r in x["refs"]]))
    return out
def bvec(bs, img, qs, d):
    idx = np.array([bs.index[img[q]] for q in qs]); n = len(bs.ids); return (bs.W @ np.bincount(idx, weights=d, minlength=n)) / np.maximum(bs.W @ np.bincount(idx, minlength=n).astype(float), 1e-9)
def stat(ds):
    vs = np.vstack([v for _, v in ds]).T; pts = [d.mean() for d, _ in ds]; cond = vs.mean(axis=1); joint = np.take_along_axis(vs, SEEDS, axis=1).mean(axis=1)
    return {"media": float(np.mean(pts)), "dp_entre_sementes": float(np.std(pts, ddof=1)), "ic_condicional": [float(np.percentile(cond, 2.5)), float(np.percentile(cond, 97.5))], "ic_conjunto": [float(np.percentile(joint, 2.5)), float(np.percentile(joint, 97.5))]}
def frozen_file(b, s, cond):
    if cond == "C2": cond = "C2v2" if b in ("HEAL", "MIMIC") else "C2"
    return f"{FR}/M3AE_FROZEN_{b}_seed{s}_{cond}.json"
def maj_file(b): return f"{PJR}/MAJ_question_only_{'HEAL_TREINO' if b == 'HEAL' else b}.json"
def bert_file(b, s): return f"{PJR}/BERT2_question_only_HEAL_TREINO_seed{s}.json" if b == "HEAL" else (f"{PJR}/BERT2_question_only_{b}_seed{s}.json" if b == "VQARAD_QLEVEL" else f"{PJR}/BERT_question_only_{b}_seed{s}.json")
I15 = json.load(open(f"{PJ}/analysis/joint_ci.json")); out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "bases": {}, "fontes": {}}
for b in BASES:
    seeds = [s for s in range(3) if os.path.exists(frozen_file(b, s, "C0"))]
    if not seeds: out["bases"][b] = {"estado": "sem predicoes"}; continue
    df = read_arrow(ARROWS[b]); cls, img = classes(b, df); allq = sorted(q for q in img if cls[q].get("answer_type") == "closed"); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
    TF = preds(f"{PJR}/TFIDF_question_only_HEAL_TREINO.json") if b == "HEAL" else tfidf_fechadas(b)[0]; PS = baseline_string(b); MJ = preds(maj_file(b))
    acc = collections.defaultdict(list); por_semente = {}; ic_exec = {}; accs = collections.defaultdict(list)
    for s in seeds:
        c = cells(b, "M3AE", s); T = preds(f"{RESULTS_DIR.get(b, V7 + '/results')}/M3AE_TXT_{TXT_BENCH_M3AE.get(b, b)}_seed{s}_TXT.json"); A = preds(c["frozen"]); BT = preds(bert_file(b, s))
        F = preds(frozen_file(b, s, "C0")); FG = preds(frozen_file(b, s, "C1")) if os.path.exists(frozen_file(b, s, "C1")) else None; FS = preds(frozen_file(b, s, "C2")) if os.path.exists(frozen_file(b, s, "C2")) else None
        AS_p = swap_file(b, "M3AE", s, c["frozen_line"]); AS = preds(AS_p) if os.path.exists(AS_p) else None
        for p in [frozen_file(b, s, k) for k in ("C0", "C1", "C2")] + [c["frozen"], AS_p, bert_file(b, s), maj_file(b)]:
            if os.path.exists(p): out["fontes"][p] = sha(p)
        qs = [q for q in allq if q in T and q in F and q in A]; v = lambda X: np.array([X[q] for q in qs], float)
        ps = {"n": len(qs), "A_text": v(T).mean(), "A_frozen": v(F).mean(), "A_ft": v(A).mean()}
        pares = {"G_enc": v(F) - v(T), "G_ft": v(A) - v(F), "G": v(A) - v(T), "A_frozen_menos_PubMedBERT": v(F) - v(BT), "A_frozen_menos_TFIDF": v(F) - v(TF),
                 "A_frozen_menos_majoritaria": v(F) - v(MJ), "A_frozen_menos_majoritaria_por_pergunta": v(F) - v(PS)}
        if FG is not None:
            qg = [q for q in qs if q in FG]; vg = lambda X: np.array([X[q] for q in qg], float); ps["A_cinza"] = vg(FG).mean()
            pares_g = {"A_text_menos_cinza": vg(T) - vg(FG), "A_frozen_menos_cinza": vg(F) - vg(FG)}
            for k, d in pares_g.items(): acc[k].append((d, bvec(bs, img, qg, d)))
        if FS is not None:
            qw = [q for q in qs if q in FS]; vw = lambda X: np.array([X[q] for q in qw], float); ps["A_swap_frozen"] = vw(FS).mean()
            for k, d in {"A_frozen_menos_A_swap": vw(F) - vw(FS), "A_swap_menos_A_text": vw(FS) - vw(T)}.items(): acc[k].append((d, bvec(bs, img, qw, d)))
        if AS is not None:
            qa = [q for q in qs if q in AS]; va = lambda X: np.array([X[q] for q in qa], float); ps["A_swap_ft"] = va(AS).mean()
        for k, d in pares.items():
            bv = bvec(bs, img, qs, d); acc[k].append((d, bv))
            if k in ("G_enc", "G_ft", "G"): ic_exec.setdefault(k, {})[s] = [float(d.mean()), float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))]
        por_semente[s] = {k: float(x) for k, x in ps.items()}
        for k, x in ps.items(): accs[k].append(float(x))
    B = {"sementes_prontas": seeds, "estado": "completa" if len(seeds) == 3 and all(os.path.exists(frozen_file(b, s, k)) for s in range(3) for k in ("C0", "C1", "C2")) else "parcial",
         "n_fechadas": len(allq), "por_semente": por_semente, "ic_por_execucao": ic_exec,
         "acuracias_media_dp": {k: [float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else None] for k, x in accs.items()},
         "contagem_ic_exclui_zero": {k: "%d de %d" % (sum(1 for m, lo, hi in d.values() if lo > 0 or hi < 0), len(d)) for k, d in ic_exec.items()}}
    if len(seeds) == 3: B["diferencas"] = {k: stat(v) for k, v in acc.items() if len(v) == 3}
    vlm = {}
    for f in ("QWEN2VL", "SMOLVLM"):
        r = {}
        for k in ("G_enc", "G_ft", "G", "A_menos_A_swap"):
            x = I15["bases"].get(b, {}).get(f, {}).get(k)
            if x: r[k] = {"media": x["media"], "dp_entre_sementes": x["dp_entre_sementes"], "ic_conjunto": x["ic_conjunto"]}
        ac = collections.defaultdict(list)
        for s in range(3):
            c = cells(b, f, s); T = preds(f"{RESULTS_DIR.get(b, V7 + '/results')}/{f}_LORA_TXT_{b}_seed{s}_TXT.json"); F = preds(c["frozen"]); A = preds(c["adapted"])
            qs = [q for q in allq if q in T and q in F and q in A]; v = lambda X: np.array([X[q] for q in qs], float)
            ac["A_text"].append(v(T).mean()); ac["A_frozen"].append(v(F).mean()); ac["A_ft"].append(v(A).mean())
            if c.get("c1") and os.path.exists(c["c1"]): G1 = preds(c["c1"]); qg = [q for q in qs if q in G1]; ac["A_cinza"].append(np.mean([G1[q] for q in qg]))
        r["acuracias_media_dp"] = {k: [float(np.mean(x)), float(np.std(x, ddof=1))] for k, x in ac.items() if len(x) == 3}; vlm[f] = r
    x = I15["bases"].get(b, {}).get("M3AE", {}).get("A_menos_A_swap")
    B["vlm_ao_lado"] = vlm; B["m3ae_ft_troca_joint_ci"] = {"media": x["media"], "dp_entre_sementes": x["dp_entre_sementes"], "ic_conjunto": x["ic_conjunto"]} if x else None
    out["bases"][b] = B; print(b, B["estado"], "sementes", seeds, flush=True)
out["script"] = os.path.abspath(__file__); out["script_sha256"] = sha(__file__)
json.dump(out, open(OUTJ, "w"), indent=1); print("FIM")
