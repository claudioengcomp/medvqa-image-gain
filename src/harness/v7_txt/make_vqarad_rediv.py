"""17/10/2026 (article_runs, vqarad_image_split): VQA-RAD re-split by image. Pool = train + validation + test of the original package
(doutorado/M3AE/data/finetune_data/vqa_rad/{train,val,test}set.json, read-only: 2 758 + 306 + 451 = 3 515 questions, 314 images).
Split by image: unique images (image_name) in alphabetical order, permuted by numpy.random.default_rng(20260913).permutation;
the first round(0,70 N) go to training, the next round(0,10 N) to validation, the rest to test; each question follows
its image (no image in more than one partition). Arrows with the SAME function as the M3AE and the A1 (prepro/make_arrow.make_arrow_vqa),
lists in the package's format (for the TF-IDF, the majority one and the BERT), test C1 (grey image, make_c1_blank_images.blank_out_images)
for the 6 frozen rows, and the grey data_root (train/validation/test) that the M3AE text-only would use (prepared, not used without ok).
Output: v7_txt/article_runs/data/vqarad_rediv/ + manifest_vqarad_rediv.json (ids and counts per partition, checks). Does not overwrite."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, os, socket, sys, time
import numpy as np, pyarrow as pa
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c1_blank_images import blank_out_images, DEFAULT_SIZE, DEFAULT_FILL_COLOR
M3 = f"{_MEDVQA_ROOT}/doutorado/M3AE"; OUT = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv"; SEED = 20260913; FR = (0.70, 0.10, 0.20)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(OUT, exist_ok=True); assert not os.listdir(OUT), "output folder not empty; nothing done"
t0 = time.time(); SRC = {s: f"{M3}/data/finetune_data/vqa_rad/{s}set.json" for s in ("train", "val", "test")}
pool = [dict(x, origem=s) for s in ("train", "val", "test") for x in json.load(open(SRC[s]))]
assert len(pool) == 3515 and len({x["qid"] for x in pool}) == 3515
imgs = sorted({x["image_name"] for x in pool}); N = len(imgs); perm = np.random.default_rng(SEED).permutation(N)
ntr, nva = int(round(FR[0] * N)), int(round(FR[1] * N)); part = {}
for k, i in enumerate(perm): part[imgs[i]] = "train" if k < ntr else ("val" if k < ntr + nva else "test")
S = {s: [x for x in pool if part[x["image_name"]] == s] for s in ("train", "val", "test")}
clo = lambda x: str(x["answer_type"]).strip().upper() == "CLOSED"; yn = lambda x: clo(x) and str(x["answer"]).strip().lower() in ("yes", "no")
for s, v in S.items(): json.dump([{k: x[k] for k in x if k != "origem"} for x in v], open(f"{OUT}/{s}set.json", "w"), indent=1)
os.chdir(M3); sys.path.insert(0, f"{M3}/prepro")
from make_arrow import make_arrow_vqa
data = {s: [{"img_path": os.path.join("data/finetune_data/vqa_rad//images", x["image_name"]), "qid": x["qid"], "question": x["question"], "answer": x["answer"], "answer_type": x["answer_type"]} for x in v] for s, v in S.items()}
make_arrow_vqa(data, "vqa_vqa_rad", OUT + "/")
def resumo(p):
    t = pa.ipc.open_file(pa.memory_map(p, "rb")).read_all(); q = t.column("question_id").to_pylist(); lab = {l for row in t.column("answer_labels").to_pylist() for ls in row for l in ls}
    return {"path": p, "sha256": sha(p), "linhas_imagens": t.num_rows, "perguntas": sum(len(x) for x in q), "image_ids": sorted(t.column("image_id").to_pylist())}, [i for x in q for i in x], lab
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__), "seed": SEED, "fracoes": FR,
       "regra": "imagens unicas em ordem alfabetica; numpy.random.default_rng(20260913).permutation; primeiras round(0,7N) treino, round(0,1N) seguintes validacao, resto teste; pergunta segue a imagem",
       "fonte": {s: {"path": p, "sha256": sha(p)} for s, p in SRC.items()}, "funcao": f"{M3}/prepro/make_arrow.py::make_arrow_vqa (sha256 {sha(M3 + '/prepro/make_arrow.py')})", "n_imagens_pool": N, "particoes": {}, "conferencias": {}}
labs = {}
for s, v in S.items():
    r, qn, labs[s] = resumo(f"{OUT}/vqa_vqa_rad_{s}.arrow")
    man["particoes"][s] = {"imagens": sorted({x["image_name"] for x in v}), "n_imagens": len({x["image_name"] for x in v}), "qids": sorted(x["qid"] for x in v), "n_perguntas": len(v), "n_fechadas": sum(clo(x) for x in v), "n_sim_nao": sum(yn(x) for x in v),
                           "n_abertas": sum(not clo(x) for x in v), "origem_no_pacote": dict(collections.Counter(x["origem"] for x in v)), "arrow": r}
    man["conferencias"][s] = {"qids_arrow_iguais_lista": sorted(qn) == sorted(x["qid"] for x in v)}
I = {s: set(man["particoes"][s]["imagens"]) for s in S}
man["conferencias"]["imagens_disjuntas"] = not (I["train"] & I["val"] or I["train"] & I["test"] or I["val"] & I["test"])
man["conferencias"]["uniao_imagens_igual_pool"] = (I["train"] | I["val"] | I["test"]) == set(imgs)
man["conferencias"]["uniao_qids_3515"] = sum(man["particoes"][s]["n_perguntas"] for s in S) == 3515
dup = collections.Counter((x["image_name"], x["question"].strip().lower()) for x in pool); man["conferencias"]["pares_imagem_pergunta_repetidos_no_pool"] = sum(1 for v in dup.values() if v > 1)
man["conferencias"]["pares_repetidos_entre_particoes"] = 0   # by construction: the question follows the image
man["conferencias"]["perguntas_teste_identicas_a_alguma_do_treino_texto"] = len({x["question"].strip().lower() for x in S["test"]} & {x["question"].strip().lower() for x in S["train"]})
assert all(v for k, v in man["conferencias"].items() if k in ("imagens_disjuntas", "uniao_imagens_igual_pool", "uniao_qids_3515")) and all(man["conferencias"][s]["qids_arrow_iguais_lista"] for s in S)
t = f"{OUT}/vqa_vqa_rad_test_C1.arrow"; blank_out_images(f"{OUT}/vqa_vqa_rad_test.arrow", t); man["c1_teste"] = {"path": t, "sha256": sha(t), "gray": {"size": DEFAULT_SIZE, "rgb": DEFAULT_FILL_COLOR}}
G = f"{OUT}/data_root_cinza"; os.makedirs(G)
for s in S: blank_out_images(f"{OUT}/vqa_vqa_rad_{s}.arrow", f"{G}/vqa_vqa_rad_{s}.arrow")
man["data_root_cinza_m3ae_txt"] = {"path": G, "sha256": {s: sha(f"{G}/vqa_vqa_rad_{s}.arrow") for s in S}, "nota": "preparado para o M3AE so-texto; so roda com ok do candidato"}
man["segundos"] = round(time.time() - t0, 1); json.dump(man, open(f"{OUT}/manifest_vqarad_rediv.json", "w"), indent=1)
print(json.dumps({s: {k: man["particoes"][s][k] for k in ("n_imagens", "n_perguntas", "n_fechadas", "n_sim_nao", "n_abertas", "origem_no_pacote")} for s in S}, indent=1)); print({k: v for k, v in man["conferencias"].items() if not isinstance(v, dict)})
