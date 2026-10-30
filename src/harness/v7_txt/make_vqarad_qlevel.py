"""19/10/2026 (the candidate, vqarad_question_split): VQA-RAD split BY QUESTION (overlap control). Same pool of 3 515 questions from the original package (train + validation
+ test), qids in increasing order permuted by numpy.random.default_rng(20260913).permutation; first round(0,70 N) training, next round(0,10 N) validation,
rest test (images free to fall into any partition, as in the official split). Arrows via the same make_arrow_vqa; test C1 (blank_out_images); test C2
(make_c2_shuffled_pairs, derangement of the test images, seed 42) with M3AE data_root; grey data_root (M3AE text-only). Output: v7_txt/article_runs/data/vqarad_qlevel/."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, os, socket, sys, time
import numpy as np, pyarrow as pa
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c1_blank_images import blank_out_images
from harness.interventions.make_c2_shuffled_pairs import shuffle_image_pairs
M3 = f"{_MEDVQA_ROOT}/doutorado/M3AE"; OUT = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_qlevel"; SEED = 20260913; FR = (0.70, 0.10, 0.20)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(OUT, exist_ok=True); assert not os.listdir(OUT), "folder not empty"
SRC = {s: f"{M3}/data/finetune_data/vqa_rad/{s}set.json" for s in ("train", "val", "test")}
pool = sorted([dict(x, origem=s) for s in ("train", "val", "test") for x in json.load(open(SRC[s]))], key=lambda x: x["qid"]); N = len(pool); assert N == 3515
perm = np.random.default_rng(SEED).permutation(N); ntr, nva = int(round(FR[0] * N)), int(round(FR[1] * N)); part = {}
for k, i in enumerate(perm): part[pool[i]["qid"]] = "train" if k < ntr else ("val" if k < ntr + nva else "test")
S = {s: [x for x in pool if part[x["qid"]] == s] for s in ("train", "val", "test")}
for s, v in S.items(): json.dump([{k: x[k] for k in x if k != "origem"} for x in v], open(f"{OUT}/{s}set.json", "w"), indent=1)
os.chdir(M3); sys.path.insert(0, f"{M3}/prepro")
from make_arrow import make_arrow_vqa
make_arrow_vqa({s: [{"img_path": os.path.join("data/finetune_data/vqa_rad//images", x["image_name"]), "qid": x["qid"], "question": x["question"], "answer": x["answer"], "answer_type": x["answer_type"]} for x in v] for s, v in S.items()}, "vqa_vqa_rad", OUT + "/")
labs = set()
for s in S:
    t = pa.ipc.open_file(pa.memory_map(f"{OUT}/vqa_vqa_rad_{s}.arrow", "rb")).read_all(); labs |= {l for row in t.column("answer_labels").to_pylist() for ls in row for l in ls}
    assert sorted(i for x in t.column("question_id").to_pylist() for i in x) == sorted(x["qid"] for x in S[s])
assert len(labs) == 498, len(labs)
blank_out_images(f"{OUT}/vqa_vqa_rad_test.arrow", f"{OUT}/vqa_vqa_rad_test_C1.arrow")
os.makedirs(f"{OUT}/data_root_cinza"); [blank_out_images(f"{OUT}/vqa_vqa_rad_{s}.arrow", f"{OUT}/data_root_cinza/vqa_vqa_rad_{s}.arrow") for s in S]
os.makedirs(f"{OUT}/c2/data_root"); c2 = f"{OUT}/c2/vqa_vqa_rad_test_c2.arrow"; shuffle_image_pairs(f"{OUT}/vqa_vqa_rad_test.arrow", c2, seed=42)
os.symlink(c2, f"{OUT}/c2/data_root/vqa_vqa_rad_test.arrow"); [os.symlink(f"{OUT}/vqa_vqa_rad_{s}.arrow", f"{OUT}/c2/data_root/vqa_vqa_rad_{s}.arrow") for s in ("train", "val")]
clo = lambda x: str(x["answer_type"]).strip().upper() == "CLOSED"; yn = lambda x: clo(x) and str(x["answer"]).strip().lower() in ("yes", "no")
I = {s: {x["image_name"] for x in v} for s, v in S.items()}
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__), "seed": SEED, "fracoes": FR, "regra": "qids em ordem crescente; default_rng(20260913).permutation; 70/10/20 por pergunta",
       "fonte": {s: {"path": p, "sha256": sha(p)} for s, p in SRC.items()}, "particoes": {s: {"qids": sorted(x["qid"] for x in v), "n_perguntas": len(v), "n_imagens": len(I[s]), "n_fechadas": sum(clo(x) for x in v), "n_sim_nao": sum(yn(x) for x in v),
       "arrow": f"{OUT}/vqa_vqa_rad_{s}.arrow", "arrow_sha256": sha(f"{OUT}/vqa_vqa_rad_{s}.arrow")} for s, v in S.items()}, "imagens_teste_em_treino_ou_validacao": len(I["test"] & (I["train"] | I["val"])), "labels": len(labs),
       "c1": {"path": f"{OUT}/vqa_vqa_rad_test_C1.arrow", "sha256": sha(f"{OUT}/vqa_vqa_rad_test_C1.arrow")}, "c2": {"path": c2, "sha256": sha(c2), "seed": 42}}
json.dump(man, open(f"{OUT}/manifest_vqarad_qlevel.json", "w"), indent=1)
print({s: {k: v for k, v in p.items() if k in ("n_perguntas", "n_imagens", "n_fechadas", "n_sim_nao")} for s, p in man["particoes"].items()}, "test images in train+val:", man["imagens_teste_em_treino_ou_validacao"], "de", len(I["test"]))
