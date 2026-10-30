"""22/10/2026 (vqarad_cauldron, etapa 2; so CPU): acuracias nas fechadas de teste do VQA-RAD (divisao por imagem e por pergunta) separadas por presenca da
pergunta na The Cauldron (niveis a, b, c de cauldron_vqarad/pareamento.json), com as predicoes ja publicadas (nada regerado). Familias: Qwen2-VL-2B e
SmolVLM2-2.2B (A_text, A_frozen, A_ft das celulas da decomposicao), M3AE (A_text = so-texto, A_frozen = encoder congelado do m3ae_frozen_vit, A_ft = fine-tune
completo); G = A_ft - A_text; PubMedBERT (extra_baselines, o mesmo da Tabela 2) e TF-IDF (tfidf_fechadas). Media (DP) das 3 sementes; IC conjunto (imagens e
sementes; bootstrap por imagem 2 000, seed 20260913, sementes RandomState(20260913)), como joint_ci/m3ae_frozen_vit; dentro - fora por replica. Por categoria
(question_type do teste: PRES, ABN, ATTRIB, qualquer posicao): so contagens e medias. Saida: analysis/cauldron_overlap.json"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, os, sys
os.environ["ARTICLE2_RUNS"] = "1"
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, cells, classes, preds, tfidf_fechadas, extra_baselines
from harness.stats.bootstrap import ImageBootstrap
from harness.interventions.arrow_io import read_arrow
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; PJR = PJ + "/results"; PAR = PJ + "/cauldron_vqarad/pareamento.json"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); SEEDS = np.random.RandomState(20260913).randint(0, 3, size=(2000, 3))
def bvec(bs, img, qs, d):
    idx = np.array([bs.index[img[q]] for q in qs]); n = len(bs.ids); return (bs.W @ np.bincount(idx, weights=d, minlength=n)) / np.maximum(bs.W @ np.bincount(idx, minlength=n).astype(float), 1e-9)
def stat(pts, vs):
    vs = np.vstack(vs).T; joint = np.take_along_axis(vs, SEEDS, axis=1).mean(axis=1)
    return {"media": float(np.mean(pts)), "dp": float(np.std(pts, ddof=1)) if len(set(pts)) > 1 else 0.0, "ic_conjunto": [float(np.percentile(joint, 2.5)), float(np.percentile(joint, 97.5))]}
PA = json.load(open(PAR)); out = {"pareamento_sha256": sha(PAR), "divisoes": {}, "fontes": {}}
for b in ("VQARAD_REDIV", "VQARAD_QLEVEL"):
    df = read_arrow(ARROWS[b]); cls, img = classes(b, df); allq = sorted(q for q in img if cls[q].get("answer_type") == "closed"); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
    Q = {int(k): v for k, v in PA["divisoes"][b]["perguntas"].items()}; assert set(Q) == set(allq), (b, len(Q), len(allq))
    EP = extra_baselines(b); TF = tfidf_fechadas(b)[0]
    def arms(f, s):
        if f == "M3AE": return {"A_text": f"{PJR}/M3AE_TXT_{b}_seed{s}_TXT.json", "A_frozen": f"{PJ}/m3ae_frozen/results/M3AE_FROZEN_{b}_seed{s}_C0.json", "A_ft": cells(b, "M3AE", s)["frozen"]}
        c = cells(b, f, s); return {"A_text": f"{PJR}/{f}_LORA_TXT_{b}_seed{s}_TXT.json", "A_frozen": c["frozen"], "A_ft": c["adapted"]}
    P = {}
    for f in ("QWEN2VL", "SMOLVLM", "M3AE"):
        for s in range(3):
            for k, p in arms(f, s).items(): P[(f, k, s)] = preds(p); out["fontes"][p] = sha(p)
    for s in range(3): P[("PubMedBERT", "acc", s)] = EP["bert"][s]; P[("TF-IDF", "acc", s)] = TF
    out["fontes"][EP["bert_src"][0]] = sha(EP["bert_src"][0])
    D = {"n_fechadas": len(allq), "niveis": {}}
    for nv in ("a", "b", "c"):
        dentro = [q for q in allq if Q[q][nv]]; fora = [q for q in allq if not Q[q][nv]]; R = {"n_dentro": len(dentro), "n_fora": len(fora), "familias": {}, "categorias": {}}
        for f, ks in (("QWEN2VL", ("A_text", "A_frozen", "A_ft", "G")), ("SMOLVLM", ("A_text", "A_frozen", "A_ft", "G")), ("M3AE", ("A_text", "A_frozen", "A_ft", "G")), ("PubMedBERT", ("acc",)), ("TF-IDF", ("acc",))):
            R["familias"][f] = {}
            for k in ks:
                cel = {}
                for lado, qs in (("dentro", dentro), ("fora", fora), ("todas", allq)):
                    if not qs: continue
                    vec = (lambda s: np.array([P[(f, "A_ft", s)][q] for q in qs], float) - np.array([P[(f, "A_text", s)][q] for q in qs], float)) if k == "G" else (lambda s: np.array([P[(f, k, s)][q] for q in qs], float))
                    cel[lado] = (([float(vec(s).mean()) for s in range(3)], [bvec(bs, img, qs, vec(s)) for s in range(3)]))
                res = {lado: stat(*v) for lado, v in cel.items()}
                if "dentro" in cel and "fora" in cel: res["dentro_menos_fora"] = stat([a - b_ for a, b_ in zip(cel["dentro"][0], cel["fora"][0])], [a - b_ for a, b_ in zip(cel["dentro"][1], cel["fora"][1])])
                R["familias"][f][k] = res
        for cat in ("PRES", "ABN", "ATTRIB"):
            qc = [q for q in allq if cat in Q[q]["question_type"].upper().replace(" ", "").split(",")]; C2 = {}
            for lado, qs in (("dentro", [q for q in qc if Q[q][nv]]), ("fora", [q for q in qc if not Q[q][nv]])):
                C2[lado] = {"n": len(qs)}
                for f, k in (("QWEN2VL", "A_ft"), ("SMOLVLM", "A_ft"), ("M3AE", "A_ft"), ("QWEN2VL", "A_text"), ("SMOLVLM", "A_text"), ("M3AE", "A_text"), ("PubMedBERT", "acc"), ("TF-IDF", "acc")):
                    if qs: C2[lado][f"{f}|{k}"] = float(np.mean([np.mean([P[(f, k, s)][q] for q in qs]) for s in range(3)]))
            R["categorias"][cat] = C2
        D["niveis"][nv] = R
    out["divisoes"][b] = D; print(b, "ok", {nv: (D["niveis"][nv]["n_dentro"], D["niveis"][nv]["n_fora"]) for nv in "abc"}, flush=True)
out["script"] = os.path.abspath(__file__); out["script_sha256"] = sha(__file__)
json.dump(out, open(PJ + "/analysis/cauldron_overlap.json", "w"), indent=1); print("FIM")
