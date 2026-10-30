"""19/10/2026 (itens 10 e 12): analises da segunda rodada, nas fechadas, com os qids, o bootstrap por imagem (2 000, seed 20260913) e o IC da media das 3 sementes
(media por replica) da decomposicao. heal_classifiers_train_only: PubMedBERT melhor epoca x ultima epoca (5 bases) e HEAL com protocolo so-treino (PubMedBERT e TF-IDF);
encoder_regimes: regimes do encoder (attn64, attnmlp) em SLAKE e VQA-RAD redividido. Uso: ARTICLE2_RUNS=1 python -m harness.v7_txt.analises_rodada2 heal_classifiers_train_only|encoder_regimes"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, sys, time
os.environ["ARTICLE2_RUNS"] = "1"
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, RESULTS_DIR, V7, TXT_BENCH_M3AE, cells, classes, preds, tfidf_fechadas
from harness.stats.bootstrap import ImageBootstrap
from harness.interventions.arrow_io import read_arrow
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; PJR = f"{PJ}/results"; FAM = ["QWEN2VL", "SMOLVLM", "M3AE"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def setup(b):
    df = read_arrow(ARROWS[b]); cls, img = classes(b, df); allq = sorted(q for q in img if cls[q].get("answer_type") == "closed"); return img, allq, ImageBootstrap([img[q] for q in allq], 2000, 20260913)
def bvec(bs, img, qs, d):
    idx = np.array([bs.index[img[q]] for q in qs]); n = len(bs.ids); num = np.bincount(idx, weights=d, minlength=n); den = np.bincount(idx, minlength=n).astype(float); return (bs.W @ num) / np.maximum(bs.W @ den, 1e-9)
def dif(A, B, qs, img, bs):
    d = np.array([A[q] for q in qs], float) - np.array([B[q] for q in qs], float); v = bvec(bs, img, qs, d)
    return {"diff": float(d.mean()), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n": len(qs), "_v": v}
def fam_mean(rs, k):
    vs = [r[k] for r in rs if r.get(k)]
    if not vs: return None
    m = np.mean([x["_v"] for x in vs], axis=0); p = [x["diff"] for x in vs]
    return {"mean": float(np.mean(p)), "sd": float(np.std(p, ddof=1)) if len(p) > 1 else None, "ic_media_sementes": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n_sementes": len(p)}
def txt_of(b, f, s):
    tl = f"{f}_LORA_TXT_{b}_seed{s}" if f != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(b, b)}_seed{s}"; return f"{RESULTS_DIR.get(b, V7 + '/results')}/{tl}_TXT.json"
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k != "_v"}
    if isinstance(o, list): return [strip(x) for x in o]
    return o
def heal_classifiers_train_only():
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "melhor_x_ultima_epoca": {}, "heal_so_treino": {}}
    for base in ("KVASIR", "SLAKE", "MIMIC", "VQARAD_REDIV", "HEAL_TREINO", "VQARAD_QLEVEL"):
        rows = []
        for s in range(3):
            p = f"{PJR}/BERT2_question_only_{base}_seed{s}.json"
            if not os.path.exists(p): rows.append({"seed": s, "ausente": p}); continue
            d = json.load(open(p)); t = d["treino"]; rows.append({"seed": s, "acc_melhor": d["accuracy"], "epoca_escolhida": t["epoca_escolhida"], "acc_ultima_epoca": t["ultima_epoca"]["accuracy"], "epocas": t["epocas"], "val_acc_escolhida": t["val_acc_escolhida"], "arquivo": p, "sha256": sha(p)})
        ok = [r for r in rows if "acc_melhor" in r]
        out["melhor_x_ultima_epoca"][base] = {"sementes": rows, **({"media_melhor": float(np.mean([r["acc_melhor"] for r in ok])), "media_ultima": float(np.mean([r["acc_ultima_epoca"] for r in ok]))} if ok else {})}
    b = "HEAL"; img, allq, bs = setup(b); TF = preds(f"{PJR}/TFIDF_question_only_HEAL_TREINO.json"); TFant, _ = tfidf_fechadas(b); L = []
    for f in FAM:
        for s in range(3):
            c = cells(b, f, s); T = preds(txt_of(b, f, s)); F = preds(c["frozen"]); A = preds(c["adapted"]) if c["adapted"] else F
            BT = preds(f"{PJR}/BERT2_question_only_HEAL_TREINO_seed{s}.json"); BTant = preds(f"{PJR}/BERT_question_only_HEAL_seed{s}.json")
            qs = [q for q in allq if q in T and q in F and q in A and q in BT and q in TF]
            L.append({"familia": f, "seed": s, "n": len(qs), "txt_menos_bert": dif(T, BT, qs, img, bs), "txt_menos_tfidf": dif(T, TF, qs, img, bs), "ft_menos_bert": dif(A, BT, qs, img, bs), "ft_menos_tfidf": dif(A, TF, qs, img, bs),
                      "acc_bert": float(np.mean([BT[q] for q in qs])), "acc_tfidf": float(np.mean([TF[q] for q in qs])), "anterior_txt_menos_bert": dif(T, BTant, qs, img, bs), "anterior_txt_menos_tfidf": dif(T, TFant, qs, img, bs)})
    out["heal_so_treino"] = {"linhas": L, "por_familia": {f: {k: fam_mean([r for r in L if r["familia"] == f], k) for k in ("txt_menos_bert", "txt_menos_tfidf", "ft_menos_bert", "ft_menos_tfidf", "anterior_txt_menos_bert", "anterior_txt_menos_tfidf")} | {"acc_bert": float(np.mean([r["acc_bert"] for r in L if r["familia"] == f])), "acc_tfidf": L[0]["acc_tfidf"]} for f in FAM}}
    return out
def encoder_regimes():
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "bases": {}}
    st = json.load(open(f"{PJ}/logs/train_state_encoder_regimes.json"))["jobs"]; tt = {j["name"]: j for j in st if j["kind"] == "train"}
    for b, bl in (("SLAKE", "slake"), ("VQARAD_REDIV", "vqarad_rediv")):
        img, allq, bs = setup(b); B = {}
        for f, fl, pre in (("QWEN2VL", "qwen2vl", "QWEN2VL_LORA_VT"), ("SMOLVLM", "smolvlm2", "SMOLVLM_LORA")):
            for v in ("publicada", "attn64", "attnmlp"):
                rs = []
                for s in range(3):
                    c = cells(b, f, s); T = preds(txt_of(b, f, s)); F = preds(c["frozen"])
                    pA = c["adapted"] if v == "publicada" else f"{PJR}/{pre}_ENC{v.upper()}_{b}_seed{s}_C0.json"
                    if not os.path.exists(pA): rs.append({"seed": s, "ausente": pA}); continue
                    A = preds(pA); qs = [q for q in allq if q in T and q in F and q in A]; j = tt.get(f"{fl}_lora_enc{v}_{bl}_seed{s}", {})
                    r = {"seed": s, "n": len(qs), "acc_ft": float(np.mean([A[q] for q in qs])), "G_ft": dif(A, F, qs, img, bs), "G": dif(A, T, qs, img, bs), "arquivo": pA, "horas_treino": (j.get("duration_s") or 0) / 3600 or None, "pico_MiB": j.get("peak_mib")}
                    r["G_ft_exclui_zero"] = r["G_ft"]["ci"][0] > 0 or r["G_ft"]["ci"][1] < 0; r["G_exclui_zero"] = r["G"]["ci"][0] > 0 or r["G"]["ci"][1] < 0; rs.append(r)
                ok = [r for r in rs if "acc_ft" in r]
                B[f"{f}|{v}"] = {"sementes": rs, "acc_ft_media": float(np.mean([r["acc_ft"] for r in ok])) if ok else None, "G_ft": fam_mean(ok, "G_ft"), "G": fam_mean(ok, "G"), "G_ft_exclui_zero": sum(r["G_ft_exclui_zero"] for r in ok), "G_exclui_zero": sum(r["G_exclui_zero"] for r in ok), "n_sementes": len(ok)}
        out["bases"][b] = B
    pr = json.load(open(f"{PJ}/analysis/encoder_regimes_parametros.json")); out["parametros_treinaveis"] = pr
    return out
if __name__ == "__main__":
    k = sys.argv[1]; o = {"heal_classifiers_train_only": heal_classifiers_train_only, "encoder_regimes": encoder_regimes}[k](); o["script"] = os.path.abspath(__file__); o["script_sha256"] = sha(__file__)
    json.dump(strip(o), open(f"{PJ}/analysis/{k}.json", "w"), indent=1); print("FIM", k)
