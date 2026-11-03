"""19/10/2026 (the candidate, text_analyses; CPU, read-only): analyses for the text numbers. Paper benchmarks (closed-form): KVASIR, SLAKE, HEAL, MIMIC, VQARAD_REDIV.
Uses the same cells (txt_floors.cells, article_runs mode), the same qids from the decomposition and the same per-image bootstrap (2 000, seed 20260913).
Output: v7_txt/article_runs/analysis/text_analyses.json. No interpretation."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, glob, hashlib, json, os, sys, time
os.environ["ARTICLE2_RUNS"] = "1"
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, RESULTS_DIR, V7, TXT_BENCH_M3AE, cells, classes, preds, tfidf_fechadas, extra_baselines
from harness.stats.bootstrap import ImageBootstrap
from harness.interventions.arrow_io import read_arrow
from med_vlm.metrics.process_text import answer_matches, normalize_text as NT
W = f"{_MEDVQA_ROOT}"; PJ = f"{W}/v7_txt/article_runs"; ART = ["KVASIR", "SLAKE", "HEAL", "MIMIC", "VQARAD_REDIV"]; FAM = ["QWEN2VL", "SMOLVLM", "M3AE"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
DEC = json.load(open(f"{PJ}/analysis/decomposicao_fechadas.json"))["benchmarks"]; REO = json.load(open(f"{PJ}/analysis/reordenacao_fechadas.json"))["bases"]
out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": os.path.abspath(__file__), "script_sha256": sha(__file__)}
def bvec(bs, img, qs, d):
    idx = np.array([bs.index[img[q]] for q in qs]); n = len(bs.ids); num = np.bincount(idx, weights=np.asarray(d, float), minlength=n); den = np.bincount(idx, minlength=n).astype(float)
    return (bs.W @ num) / np.maximum(bs.W @ den, 1e-9)
# ---- 13.1 G_ft per run
L = []
for b in ART:
    for r in DEC[b]["lines"]:
        if r["family"] == "M3AE": continue
        g = r["overall"]["ganho_adaptacao"]; L.append({"base": b, "linha": r["line"], "G_ft": g["diff"], "ic": g["ci"], "exclui_zero": g["ci"][0] > 0 or g["ci"][1] < 0})
out["13_1"] = {"execucoes": L, "por_base": {b: {"excluem": sum(x["exclui_zero"] for x in L if x["base"] == b), "max_G_ft_entre_as_que_excluem": max([x["G_ft"] for x in L if x["base"] == b and x["exclui_zero"]], default=None)} for b in ART},
               "total_excluem": sum(x["exclui_zero"] for x in L), "max_global_entre_as_que_excluem": max(x["G_ft"] for x in L if x["exclui_zero"])}
# ---- 13.2 majoritaria
out["13_2"] = {b: json.load(open(f"{PJ}/results/MAJ_question_only_{b}.json"))["accuracy"] for b in ART}
# ---- 13.3 majority by exact question text
PAT = ["is there", "are there", "have all polyps", "how many", "what type of", "what is the size", "where in the image", "does this image contain any finding", "what color is the anatomical landmark", "what color is the abnormality"]
def tpl(q):
    q = q.strip().lower()
    return next((p for p in PAT if q.startswith(p)), None)
r133 = {}
for b in ART:
    D = json.load(open(f"{PJ}/data/texto/{b}.json")); tr, te = D["train_so_treino"], D["test"]
    byq = collections.defaultdict(collections.Counter); byt = collections.defaultdict(collections.Counter); glob_ = collections.Counter()
    for x in tr: byq[NT(x["question"])][x["answer"]] += 1; glob_[x["answer"]] += 1; (byt[tpl(x["question"])].__setitem__(x["answer"], byt[tpl(x["question"])][x["answer"]] + 1) if b == "KVASIR" else None)
    top = lambda c: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    ok = []; fonte = collections.Counter()
    for x in te:
        k = NT(x["question"])
        if k in byq: p = top(byq[k]); fonte["texto_exato"] += 1
        elif b == "KVASIR" and tpl(x["question"]) in byt: p = top(byt[tpl(x["question"])]); fonte["template"] += 1
        else: p = top(glob_); fonte["global"] += 1
        ok.append(bool(answer_matches(NT(p), [NT(r) for r in x["refs"]])))
    r133[b] = {"n": len(te), "acc": float(np.mean(ok)), "fonte_da_predicao": dict(fonte), "treino": "train_so_treino from dados_texto (HEAL: train only)", "normalizacao_da_chave": "normalize_text of the question"}
out["13_3"] = r133
# ---- 13.4 / 13.7: predictors per benchmark
r134, r137 = {}, {}
rng_seeds = np.random.RandomState(20260913).randint(0, 3, size=(2000, 3))
for b in ART:
    df = read_arrow(ARROWS[b]); cls, img = classes(b, df); allq = sorted(q for q in img if cls[q].get("answer_type") == "closed"); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
    tfidf, _ = tfidf_fechadas(b); XP = extra_baselines(b); P = {}
    for f in FAM:
        for s in range(3):
            c = cells(b, f, s); tl = f"{f}_LORA_TXT_{b}_seed{s}" if f != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(b, b)}_seed{s}"
            T = preds(f"{RESULTS_DIR.get(b, V7 + '/results')}/{tl}_TXT.json"); F = preds(c["frozen"]); A = preds(c["adapted"]) if c["adapted"] else F; C1 = preds(c["c1"]) if c["c1"] and os.path.exists(c["c1"]) else None
            qs = [q for q in allq if q in T and q in F and q in A]; P[(f, s)] = (qs, T, F, A, C1, XP["bert"][s])
    base = {}; 
    for f in FAM:
        v = {"A_text": [], "TF-IDF": [], "PubMedBERT": [], "majoritaria": [], "cinza": []}
        for s in range(3):
            qs, T, F, A, C1, BT = P[(f, s)]; a = np.mean([A[q] for q in qs])
            v["A_text"].append(a - np.mean([T[q] for q in qs])); v["TF-IDF"].append(a - np.mean([tfidf[q] for q in qs])); v["PubMedBERT"].append(a - np.mean([BT[q] for q in qs])); v["majoritaria"].append(a - np.mean([XP["maj"][q] for q in qs]))
            if C1 and f != "M3AE": q1 = [q for q in qs if q in C1]; v["cinza"].append(np.mean([A[q] for q in q1]) - np.mean([C1[q] for q in q1]))
        base[f] = {k: float(np.mean(x)) for k, x in v.items() if x}
    ordens = {k: sorted([f for f in FAM if k in base[f]], key=lambda f: -base[f][k]) for k in ("A_text", "TF-IDF", "PubMedBERT", "majoritaria", "cinza")}
    pares = REO[b]["artigo_9_ft"]["pares_que_trocam"]; fam_of = lambda n: "M3AE" if n.startswith("M3AE") else n.split("_")[0]
    r134[b] = {"A_ft_menos_baseline_mean_per_family": base, "order_by_baseline": ordens, "ordem_igual_sob_todos_os_baselines_with_3_familias": len({tuple(o) for k, o in ordens.items() if k != "cinza"}) == 1,
               "ordem_VLM_sob_cinza_igual_a_ordem_VLM_sob_A_text": [f for f in ordens["A_text"] if f != "M3AE"] == ordens["cinza"],
               "pares_discordantes_TFIDF_x_Atext": [(p["a"], p["b"], "mesma familia" if fam_of(p["a"]) == fam_of(p["b"]) else "familias diferentes") for p in pares]}
    # 13.7: conditional CI (mean per replica of the 3 seeds) x joint CI (resamples images and seeds) and SD between seeds
    r137[b] = {}
    for f in FAM:
        for key in (("G_ft", "G") if f != "M3AE" else ("G",)):
            vs, pts = [], []
            for s in range(3):
                qs, T, F, A, C1, BT = P[(f, s)]; d = np.array([A[q] for q in qs], float) - np.array([(F if key == "G_ft" else T)[q] for q in qs], float)
                vs.append(bvec(bs, img, qs, d)); pts.append(d.mean())
            V = np.vstack(vs).T; cond = V.mean(axis=1); joint = np.take_along_axis(V, rng_seeds, axis=1).mean(axis=1)
            r137[b][f"{f}|{key}"] = {"media": float(np.mean(pts)), "dp_entre_sementes": float(np.std(pts, ddof=1)), "ic_condicional_media_sementes": [float(np.percentile(cond, 2.5)), float(np.percentile(cond, 97.5))],
                                     "ic_conjunto_imagens_e_sementes": [float(np.percentile(joint, 2.5)), float(np.percentile(joint, 97.5))]}
    print(b, "ok", flush=True)
out["13_4"] = r134; out["13_7"] = {"valores": r137, "codigo": {"occlusion_stats.py": ["harness/v7_occlusion/occlusion_stats.py", sha(f"{W}/v7_txt/worktree/harness/v7_occlusion/occlusion_stats.py"), "ImageBootstrap linhas 20-33"],
                                     "decomposicao.py": ["harness/v7_txt/decomposicao.py", sha(f"{W}/v7_txt/worktree/harness/v7_txt/decomposicao.py"), "boot_vec row 24; replicas per row row 74; CI of the mean across seeds rows ~93-97"],
                                     "ic_contagem.py": ["harness/v7_txt/ic_contagem.py", sha(f"{W}/v7_txt/worktree/harness/v7_txt/ic_contagem.py"), "exclui_zero row 8 (CI per row)"]},
                         "sorteio_de_sementes": "numpy RandomState(20260913).randint(0, 3, (2000, 3)): per replica, 3 seeds with replacement; the same image replica for all three"}
# ---- 13.5 VQA-RAD re-split, per subset
b = "VQARAD_REDIV"; TS = {x["qid"]: x for x in json.load(open(f"{PJ}/data/vqarad_rediv/testset.json"))}; closed = [q for q in TS if str(TS[q]["answer_type"]).strip().upper() == "CLOSED"]
_OJ = json.load(open(f"{W}/doutorado/data_drive_sync/raw/vqa_rad_osf/VQA_RAD Dataset Public.json")); osf = {(x["image_name"], NT(x["question"])): x for x in _OJ}
for x in _OJ:   # 'frame' questions linked by question_frame (same image)
    if str(x.get("question_frame", "NULL")).strip() not in ("", "NULL"): osf.setdefault((x["image_name"], NT(x["question_frame"])), x)
src = {}
for f in FAM:
    for s in range(3):
        tl = f"{f}_LORA_TXT_{b}_seed{s}" if f != "M3AE" else f"M3AE_TXT_{b}_seed{s}"; src[(f, s)] = preds(f"{PJ}/results/{tl}_TXT.json")
for s in range(3): src[("PubMedBERT", s)] = preds(f"{PJ}/results/BERT_question_only_{b}_seed{s}.json")
def rec(k): return {"question_type": lambda q: TS[q]["question_type"], "answer_type_osf": lambda q: str((osf.get((TS[q]["image_name"], NT(TS[q]["question"]))) or {}).get("answer_type", "?")).strip(), "prefixo_2_palavras": lambda q: " ".join(NT(TS[q]["question"]).split()[:2])}[k]
r135 = {}
for k in ("question_type", "answer_type_osf", "prefixo_2_palavras"):
    g = collections.defaultdict(list)
    for q in closed: g[rec(k)(q)].append(q)
    r135[k] = {}
    for val, qs in sorted(g.items(), key=lambda kv: -len(kv[1])):
        if k == "prefixo_2_palavras" and len(qs) < 10: continue
        row = {"n": len(qs)}
        for m in ("QWEN2VL", "SMOLVLM", "M3AE", "PubMedBERT"):
            a = [np.mean([src[(m, s)][q] for q in qs]) for s in range(3)]; row[m] = {"media": float(np.mean(a)), "dp": float(np.std(a, ddof=1))}
        row["SmolVLM2_menos_PubMedBERT"] = row["SMOLVLM"]["media"] - row["PubMedBERT"]["media"]; r135[k][val] = row
out["13_5"] = {"base": "VQA-RAD re-split, 418 closed-form", "recortes": r135, "prefixo": "first two words of the normalized question; groups with n >= 10"}
# ---- 13.6 cost
C = []
def add(base, mod, s, gpu, dur, pk, fonte):
    C.append({"base": base, "modelo": mod, "seed": s, "placa": gpu, "horas": None if dur is None else dur / 3600, "pico_GB": None if not pk else pk / 1024, "fonte": fonte})
for st, mp in ((f"{W}/v7_txt/logs/train_state.json", None), (f"{W}/v7_mimic/logs/train_state.json", None), (f"{PJ}/logs/train_state_vqarad_rediv.json", None)):
    for j in json.load(open(st))["jobs"]:
        if j.get("kind", "train") == "train" and "_txt_" in j["name"] and j["status"] == "done" and "vqarad_radorig" not in j["name"]:
            bb = j["name"].split("_txt_")[1].rsplit("_seed", 1)[0].upper(); bb = {"VQARAD": "VQARAD_HF", "VQARAD_REDIV": "VQARAD_REDIV"}.get(bb, bb)
            add(bb, "Qwen2-VL" if j["name"].startswith("qwen") else "SmolVLM2", int(j["name"][-1]), j.get("gpu"), j.get("duration_s"), j.get("peak_mib"), st.replace(W + "/", ""))
for st in (f"{W}/v7_txt/logs/infer_state_m3ae_txt.json", f"{W}/v7_mimic/logs/infer_state_m3ae_mimic.json", f"{PJ}/logs/infer_state_m3ae_rediv.json"):
    for j in json.load(open(st))["jobs"]:
        if j["name"].startswith("train_M3AE_TXT") and j["status"] == "done":
            bb = j["name"][len("train_M3AE_TXT_"):].rsplit("_seed", 1)[0].upper().replace("SLACK", "SLAKE").replace("VQA_RAD", "VQARAD_HF"); pk = j.get("peak_mib")
            cf = glob.glob(f"{os.path.dirname(st)}/{j['name'][6:]}_cost.json")
            if not pk and cf: pk = json.load(open(cf[0]))["train_peak_memory_gb"] * 1024
            add(bb, "M3AE so-texto", int(j["name"][-1]), j.get("gpu"), j.get("duration_s"), pk, st.replace(W + "/", "") + (" (pico do _cost.json, placa inteira)" if not j.get("peak_mib") else ""))
for st in (f"{PJ}/logs/infer_state_bert.json", f"{PJ}/logs/infer_state_bert_rediv.json"):
    for j in json.load(open(st))["jobs"]:
        if j["status"] == "done": add(j["name"][len("BERT_question_only_"):].rsplit("_seed", 1)[0], "PubMedBERT", int(j["name"][-1]), j.get("gpu"), j.get("duration_s"), j.get("peak_mib"), st.replace(W + "/", ""))
out["13_6"] = C
json.dump(out, open(f"{PJ}/analysis/text_analyses.json", "w"), indent=1, default=str); print("FIM")
