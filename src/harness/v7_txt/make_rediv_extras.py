"""17/10/2026 (article_runs, vqarad_image_split + decisions of the candidate): extra arrows of the re-split VQA-RAD.
(1) C2 = image swapped for another from the same test set: harness.interventions.make_c2_shuffled_pairs.shuffle_image_pairs (derangement over the rows
    = images of the test arrow, seed 42), the same function and seed as the C2 of Kvasir/SLAKE/VQA-RAD from p0 (C2_manifest_*.json); manifest in the same format
    (pairs qid -> original/donor image, recebida_sha256). (2) M3AE data_root for the C2 (test = C2; train/validation = links).
(3) arrow with only the 418 closed-form questions of the new test set, for the zero-shot. Output: v7_txt/article_runs/data/vqarad_rediv/{c2,zeroshot_fechadas}/."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c2_shuffled_pairs import shuffle_image_pairs
from harness.interventions.arrow_io import read_arrow, write_arrow, count_questions
from harness.circle_data_bridge import load_condition_dataset
RD = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv"; TE = f"{RD}/vqa_vqa_rad_test.arrow"; SEED = 42
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
C2D = f"{RD}/c2"; os.makedirs(C2D, exist_ok=False); c2 = f"{C2D}/vqa_vqa_rad_test_c2.arrow"; shuffle_image_pairs(TE, c2, seed=SEED)
a, b = read_arrow(TE), read_arrow(c2); assert len(a) == len(b) and all(list(x) == list(y) for x, y in zip(a["question_id"], b["question_id"]))
ha = [hashlib.sha256(x).hexdigest() for x in a["image"]]; hb = [hashlib.sha256(x).hexdigest() for x in b["image"]]
assert sum(x == y for x, y in zip(ha, hb)) == 0 and sorted(ha) == sorted(hb)
pairs = [{"qid": int(q), "qid_image": a["image_id"][i], "donor_image": b["image_id"][i], "received_sha256": hb[i]} for i in range(len(a)) for q in a["question_id"][i]]
os.makedirs(f"{C2D}/data_root"); os.symlink(c2, f"{C2D}/data_root/vqa_vqa_rad_test.arrow")
for s in ("train", "val"): os.symlink(f"{RD}/vqa_vqa_rad_{s}.arrow", f"{C2D}/data_root/vqa_vqa_rad_{s}.arrow")
man = {"header": {"benchmark": "VQARAD_REDIV", "condition": "C2", "source_arrow": TE, "source_arrow_sha256": sha(TE), "c2_arrow": c2, "c2_arrow_sha256": sha(c2), "seed": SEED, "funcao": "harness.interventions.make_c2_shuffled_pairs.shuffle_image_pairs (same as the C2 of p0)",
                  "images": len(a), "entries": len(pairs), "images_that_kept_their_own": 0, "sha_multiset_matches_c0": True, "load_condition_dataset_n": len(load_condition_dataset(c2)), "m3ae_data_root": f"{C2D}/data_root", "t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": __file__, "script_sha256": sha(__file__)}, "pairs": pairs}
json.dump(man, open(f"{C2D}/C2_manifest_VQARAD_REDIV.json", "w"), indent=1)
ZD = f"{RD}/zeroshot_fechadas"; os.makedirs(ZD, exist_ok=False); rows = []
for _, r in a.iterrows():
    k = [i for i, t in enumerate(r["answer_type"]) if int(t) == 0]
    if not k: continue
    r = r.copy()
    for c in ("questions", "answers", "answer_labels", "answer_scores", "question_id", "answer_type"): r[c] = [r[c][i] for i in k]
    rows.append(r)
z = f"{ZD}/vqa_vqa_rad_test_fechadas.arrow"; write_arrow(pd.DataFrame(rows).reset_index(drop=True), z); zz = read_arrow(z)
tl = json.load(open(f"{RD}/testset.json")); closed = sorted(int(x["qid"]) for x in tl if str(x["answer_type"]).strip().upper() == "CLOSED")
assert count_questions(zz) == 418 and sorted(int(q) for r in zz["question_id"] for q in r) == closed
json.dump({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "source": TE, "source_sha256": sha(TE), "output": z, "sha256": sha(z), "rows": len(zz), "questions": count_questions(zz), "load_condition_dataset_n": len(load_condition_dataset(z))}, open(f"{ZD}/manifest.json", "w"), indent=1)
print("C2", man["header"]["images"], "imagens", man["header"]["entries"], "perguntas", man["header"]["c2_arrow_sha256"][:16], "| zero-shot closed-form", count_questions(zz), sha(z)[:16])
