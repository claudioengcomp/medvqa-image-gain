"""23/10/2026 (m3ae_no_image): M3AE so-texto SEM imagem (S) x so-texto cinza publicado (Tc) x so-texto preto do m3ae_black_image (Tp, so sementes completas), nas fechadas.
G_enc(S) = F - S e G(S) = A - S (F = encoder congelado do m3ae_frozen_vit; A = completo publicado), ao lado dos publicados (F - Tc, A - Tc); G_ft nao muda.
IC conjunto (imagens e sementes; 2 000 replicas; seed 20260913; sementes RandomState(20260913)), como joint_ci/17/21. Base com menos de 3 sementes de S:
so por semente. Preto: so por semente (m3ae_black_image interrompido). Categorias (so VQA-RAD): contagens e medias. Saida: analysis/m3ae_no_image.json"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, os
os.environ["ARTICLE2_RUNS"] = "1"
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, RESULTS_DIR, V7, TXT_BENCH_M3AE, cells, classes, preds
from harness.stats.bootstrap import ImageBootstrap
from harness.interventions.arrow_io import read_arrow
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; RS = PJ + "/m3ae_semimg/results"; RP = PJ + "/m3ae_preto/results"; FRZ = PJ + "/m3ae_frozen/results"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); SEEDS = np.random.RandomState(20260913).randint(0, 3, size=(2000, 3))
def bvec(bs, img, qs, d):
    idx = np.array([bs.index[img[q]] for q in qs]); n = len(bs.ids); return (bs.W @ np.bincount(idx, weights=d, minlength=n)) / np.maximum(bs.W @ np.bincount(idx, minlength=n).astype(float), 1e-9)
def stat(pts, vs):
    vs = np.vstack(vs).T; cond = vs.mean(axis=1); joint = np.take_along_axis(vs, SEEDS, axis=1).mean(axis=1)
    return {"media": float(np.mean(pts)), "dp": float(np.std(pts, ddof=1)), "por_semente": [float(x) for x in pts], "ic_condicional": [float(np.percentile(cond, 2.5)), float(np.percentile(cond, 97.5))], "ic_conjunto": [float(np.percentile(joint, 2.5)), float(np.percentile(joint, 97.5))]}
def files(b, s):
    return {"S": f"{RS}/M3AE_SEMIMG_{b}_seed{s}_TXT.json", "Tc": f"{RESULTS_DIR.get(b, V7 + '/results')}/M3AE_TXT_{TXT_BENCH_M3AE.get(b, b)}_seed{s}_TXT.json",
            "F": f"{FRZ}/M3AE_FROZEN_{b}_seed{s}_C0.json", "A": cells(b, "M3AE", s)["frozen"], "Tp": f"{RP}/M3AE_PRETO_{b}_seed{s}_PRETA.json"}
PARES = {"A_text_sem_imagem": ("S", None), "A_text_cinza": ("Tc", None), "sem_imagem_menos_cinza": ("S", "Tc"), "G_enc_sem_imagem": ("F", "S"), "G_enc_publicado": ("F", "Tc"),
         "G_sem_imagem": ("A", "S"), "G_publicado": ("A", "Tc")}
out = {"bases": {}, "fontes": {}}
for b in ("VQARAD_REDIV", "SLAKE", "HEAL", "MIMIC", "KVASIR", "VQARAD_QLEVEL"):
    ok = [s for s in range(3) if all(os.path.exists(files(b, s)[k]) for k in ("S", "Tc", "F", "A"))]
    if not ok: out["bases"][b] = {"estado": "sem sementes completas"}; continue
    df = read_arrow(ARROWS[b]); cls, img = classes(b, df); allq = sorted(q for q in img if cls[q].get("answer_type") == "closed"); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
    acc = collections.defaultdict(lambda: ([], [])); preto = {}; nps = {}; difr = {}
    for s in ok:
        P = {k: preds(p) for k, p in files(b, s).items() if os.path.exists(p)}
        for k, p in files(b, s).items():
            if os.path.exists(p): out["fontes"][p] = sha(p)
        qs = [q for q in allq if all(q in P[k] for k in ("S", "Tc", "F", "A"))]; v = lambda k: np.array([P[k][q] for q in qs], float); nps[s] = len(qs)
        RA = {k: {int(x["qid"]): x["predicted"] for x in json.load(open(files(b, s)[k]))["predictions"]} for k in ("S", "Tc")}
        difr[s] = {"respostas_diferentes_S_x_Tc": sum(RA["S"][q] != RA["Tc"][q] for q in qs), "acerto_diferente_S_x_Tc": int((v("S") != v("Tc")).sum()), "n": len(qs)}
        for nome, (x, y) in PARES.items():
            d = v(x) - (v(y) if y else 0); acc[nome][0].append(float(d.mean())); acc[nome][1].append(bvec(bs, img, qs, d))
        if "Tp" in P:
            qp = [q for q in qs if q in P["Tp"]]; vp = lambda k: np.array([P[k][q] for q in qp], float)
            preto[s] = {"A_text_preto": float(vp("Tp").mean()), "sem_imagem_menos_preto": float((vp("S") - vp("Tp")).mean()), "n": len(qp)}
    B = {"estado": "completa" if len(ok) == 3 else "parcial", "sementes": ok, "n_fechadas": len(allq), "n_por_semente": nps,
         "por_semente": {nome: dict(zip(ok, v[0])) for nome, v in acc.items()}, "preto_por_semente": preto, "discordancia_por_semente": difr}
    if len(ok) == 3: B["estat"] = {nome: stat(*v) for nome, v in acc.items()}
    if b.startswith("VQARAD"):
        tp = {int(x["qid"]): str(x.get("question_type")) for x in json.load(open(f"{PJ}/data/{'vqarad_rediv' if b == 'VQARAD_REDIV' else 'vqarad_qlevel'}/testset.json"))}; B["categorias"] = {}
        for cat in ("PRES", "ABN", "ATTRIB"):
            qc = [q for q in allq if cat in tp.get(q, "").upper().replace(" ", "").split(",")]; r = {"n": len(qc)}
            for k in ("S", "Tc", "F", "A"): r[k] = float(np.mean([np.mean([preds(files(b, s)[k])[q] for q in qc]) for s in ok])) if qc else None
            sp = [s for s in ok if os.path.exists(files(b, s)["Tp"])]
            r["Tp"] = float(np.mean([np.mean([preds(files(b, s)["Tp"])[q] for q in qc]) for s in sp])) if sp and qc else None; r["Tp_sementes"] = sp
            B["categorias"][cat] = r
    if ok == [0, 1, 2]:
        RS3 = [{int(x["qid"]): x["predicted"] for x in json.load(open(files(b, s)["S"]))["predictions"]} for s in ok]
        B["respostas_diferentes_entre_sementes_S"] = {f"s{i}_s{j}": sum(RS3[i][q] != RS3[j][q] for q in allq if q in RS3[i] and q in RS3[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
    out["bases"][b] = B; print(b, B["estado"], ok, flush=True)
out["script"] = os.path.abspath(__file__); out["script_sha256"] = sha(__file__)
json.dump(out, open(PJ + "/analysis/m3ae_no_image.json", "w"), indent=1); print("FIM")
