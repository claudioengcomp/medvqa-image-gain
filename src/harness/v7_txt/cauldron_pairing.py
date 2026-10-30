"""22/10/2026 (vqarad_cauldron, stage 1; CPU only): pairing of the vqarad subset of The Cauldron (HuggingFaceM4/the_cauldron, revision 847a98a7) with the pool
of the paper's VQA-RAD (the 314 images of the redivision = pool of the split by question). Images: exact hash of the decoded pixels (RGB) and, if it does not match,
pHash (DCT 32x32 -> 8x8, threshold at the median; scipy) with Hamming distance <= 4. Questions (normalize_text of the paper): (a) same image and same
question; (b) (a) and same answer; (c) same image and the template version (question_frame from the OSF) of the question present. Origin in the official split:
qid in M3AE's testset.json (official test, 451) or not (official training). Output (stays on gpu1): cauldron_vqarad/pareamento.json"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, io, json, random, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
from scipy.fft import dct
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from med_vlm.metrics.process_text import normalize_text as N
W = f"{_MEDVQA_ROOT}"; P = W + "/v7_txt/article_runs"; O = P + "/cauldron_vqarad"
PQ = O + "/raw/vqarad/train-00000-of-00001-ab894dd16896d4e3.parquet"; REV = "847a98a779b1652d65111daf20c972dfcd333605"
OSF = W + "/doutorado/data_drive_sync/raw/vqa_rad_osf/VQA_RAD Dataset Public.json"; OFTEST = W + "/doutorado/M3AE/data/finetune_data/vqa_rad/testset.json"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def pix(b): im = Image.open(io.BytesIO(b)).convert("RGB"); return im, hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()
def phash(im):
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=float); d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return (d > np.median(d)).flatten()
# Cauldron
t = pq.read_table(PQ); rows = t.to_pylist(); C = []
for i, r in enumerate(rows):
    im, ph = pix(r["images"][0]["bytes"]); C.append({"i": i, "pix": ph, "phash": phash(im), "size": im.size, "path": r["images"][0].get("path"), "img": im,
                                                     "qa": [(N(x["user"].split("\n")[0]), N(x["assistant"].strip().rstrip("."))) for x in r["texts"]]})
# pool do artigo (redivisao: train/val/test cobrem as 314 imagens)
POOL = {}
for sp in ("train", "val", "test"):
    tb = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{P}/data/vqarad_rediv/vqa_vqa_rad_{sp}.arrow", "r")).read_all()
    for r in tb.select(["image", "image_id"]).to_pylist():
        im, ph = pix(r["image"]); POOL[str(r["image_id"]).split("/")[-1]] = {"pix": ph, "phash": phash(im), "size": im.size, "img": im}
# pareamento de imagens
bypix = {}; [bypix.setdefault(c["pix"], []).append(c["i"]) for c in C]
match = {}; metodo = {}
for name, v in POOL.items():
    if v["pix"] in bypix: match[name] = bypix[v["pix"]]; metodo[name] = "exato"; continue
    ds = [(int((v["phash"] != c["phash"]).sum()), c["i"]) for c in C]; dmin = min(ds)[0]
    if dmin <= 4: match[name] = [i for d, i in ds if d == dmin]; metodo[name] = f"phash d={dmin}"
usadas = {i for v in match.values() for i in v}
# numeric check of 10 random pairs (without copying images outside gpu1)
rnd = random.Random(20260913); amostra = rnd.sample(sorted(match), min(10, len(match))); conf = []
for name in amostra:
    a = np.asarray(POOL[name]["img"].convert("L").resize((256, 256)), dtype=float); b = np.asarray(C[match[name][0]]["img"].convert("L").resize((256, 256)), dtype=float)
    conf.append({"imagem": name, "cauldron_linha": match[name][0], "metodo": metodo[name], "tamanhos": [POOL[name]["size"], C[match[name][0]]["size"]], "dif_media_abs_0a255": round(float(np.abs(a - b).mean()), 3), "correlacao_pixels": round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 5)})
# questions per image in the Cauldron
CQ = {}
for name, idx in match.items():
    for i in idx:
        for q, a in C[i]["qa"]: CQ.setdefault(name, {}).setdefault(q, set()).add(a)
Ojs = json.load(open(OSF)); nul = lambda s: s is None or str(s).strip() in ("", "NULL")
fr = {(x["image_name"], N(x["question"])): x["question_frame"] for x in Ojs if not nul(x["question_frame"])}
oftest = {int(x["qid"]) for x in json.load(open(OFTEST))}
out = {"cauldron": {"repo": "HuggingFaceM4/the_cauldron", "config": "vqarad", "revisao": REV, "arquivo": PQ, "sha256": sha(PQ), "linhas": len(rows), "turnos": sum(len(c["qa"]) for c in C),
                    "imagens_distintas_pixels": len(bypix), "path_preenchido": sum(1 for c in C if c["path"])},
       "pool": {"imagens": len(POOL), "pareadas": len(match), "por_metodo": {m: sum(1 for x in metodo.values() if x.startswith(m)) for m in ("exato", "phash")}, "linhas_cauldron_usadas": len(usadas),
                "imagens_com_mais_de_uma_linha": sum(1 for v in match.values() if len(v) > 1)},
       "conferencia_10_pares": conf, "osf_sha256": sha(OSF), "testset_oficial_sha256": sha(OFTEST), "divisoes": {}}
for div, d in (("VQARAD_REDIV", f"{P}/data/vqarad_rediv"), ("VQARAD_QLEVEL", f"{P}/data/vqarad_qlevel")):
    S = {k: json.load(open(f"{d}/{k}set.json")) for k in ("train", "val", "test")}; te = S["test"]
    timg = {x["image_name"] for x in te}; trimg = {x["image_name"] for x in S["train"] + S["val"]}
    Q = {}
    for x in te:
        if str(x["answer_type"]).strip().upper() != "CLOSED": continue
        im = x["image_name"]; q = N(x["question"]); a = N(str(x["answer"])); cq = CQ.get(im, {}); f = fr.get((im, q))
        Q[int(x["qid"])] = {"imagem": im, "na_cauldron_imagem": im in match, "a": q in cq, "b": q in cq and a in cq[q], "c": f is not None and N(f) in cq,
                            "origem_oficial": "teste" if int(x["qid"]) in oftest else "treino", "question_type": str(x.get("question_type"))}
    out["divisoes"][div] = {"imagens_teste": len(timg), "imagens_teste_na_cauldron": len(timg & set(match)), "imagens_treino_val": len(trimg), "imagens_treino_val_na_cauldron": len(trimg & set(match)),
                            "fechadas_teste": len(Q), **{f"nivel_{k}": sum(1 for v in Q.values() if v[k]) for k in ("a", "b", "c")},
                            "nivel_b_por_origem": {o: sum(1 for v in Q.values() if v["b"] and v["origem_oficial"] == o) for o in ("treino", "teste")},
                            "nivel_a_por_origem": {o: sum(1 for v in Q.values() if v["a"] and v["origem_oficial"] == o) for o in ("treino", "teste")},
                            "nivel_c_por_origem": {o: sum(1 for v in Q.values() if v["c"] and v["origem_oficial"] == o) for o in ("treino", "teste")},
                            "fechadas_por_origem": {o: sum(1 for v in Q.values() if v["origem_oficial"] == o) for o in ("treino", "teste")}, "perguntas": Q}
json.dump(out, open(O + "/pareamento.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("divisoes", "conferencia_10_pares")}, default=str))
for c in conf: print(c)
for div, v in out["divisoes"].items(): print(div, {k: x for k, x in v.items() if k != "perguntas"})
