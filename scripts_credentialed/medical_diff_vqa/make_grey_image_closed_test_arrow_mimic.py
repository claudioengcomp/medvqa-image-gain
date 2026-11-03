"""17/10/2026 (article_runs, baselines): arrow of the 1 499 CLOSED-FORM test questions of MIMIC (Medical-Diff-VQA) with the constant grey image
from harness.interventions.make_c1_blank_images.blank_out_images (the same function as the C1 of Kvasir, SLAKE, VQA-RAD and HEAL; 384x384 RGB 128).
Step 1: subset of the test arrow (read only) to the questions with answer_type 0 (closed-form), rows without a closed-form question are removed.
Step 2: blank_out_images on the subset. Checks: 1 499 questions; qids = closed-form of items_test.parquet = closed-form of the decomposition;
load_condition_dataset reads 1 499. MIMIC images never leave gpu1 (the subset with pixels stays in v7_txt/article_runs/data, on gpu1).
Outputs: v7_txt/article_runs/data/mimic_c1_fechadas/{vqa_mimic_test_fechadas.arrow, vqa_mimic_test_fechadas_C1.arrow, manifest.json}"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, socket, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
import numpy as np, pandas as pd
from harness.interventions.make_c1_blank_images import blank_out_images, DEFAULT_SIZE, DEFAULT_FILL_COLOR
from harness.interventions.arrow_io import read_arrow, write_arrow, count_questions
from harness.circle_data_bridge import load_condition_dataset
SRC = f"{_MEDVQA_ROOT}/v7_mimic/data/vqa_mimic_test.arrow"; OUT = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/mimic_c1_fechadas"
PERQ = ["questions", "answers", "answer_labels", "answer_scores", "question_id", "answer_type"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(OUT, exist_ok=True); assert not os.listdir(OUT), "output folder not empty"; t0 = time.time()   # 17/10: the first attempt created the folder and died before writing
df = read_arrow(SRC); rows = []
for _, r in df.iterrows():
    keep = [i for i, t in enumerate(r["answer_type"]) if int(t) == 0]
    if not keep: continue
    r = r.copy()
    for c in PERQ: r[c] = [r[c][i] for i in keep]   # Python lists (np.asarray flattened answers of length 1 into 2D)
    rows.append(r)
sub = pd.DataFrame(rows).reset_index(drop=True); s1 = f"{OUT}/vqa_mimic_test_fechadas.arrow"; write_arrow(sub, s1)
o = f"{OUT}/vqa_mimic_test_fechadas_C1.arrow"; blank_out_images(s1, o)
a, b = read_arrow(s1), read_arrow(o); qs = lambda d: sorted(int(q) for row in d["question_id"] for q in row)
it = pd.read_parquet(f"{_MEDVQA_ROOT}/v7_mimic/data/items_test.parquet"); closed = sorted(int(q) for q in it[it.answer_type == "closed"].qid)
assert count_questions(a) == count_questions(b) == 1499 and qs(a) == qs(b) == closed, (count_questions(a), count_questions(b))
assert all(len(set(x)) == 1 and int(list(x)[0]) == 0 for x in b["answer_type"])
ds = load_condition_dataset(o); assert len(ds) == 1499, len(ds)
src_df = read_arrow(SRC); m = {int(q): (str(qq), list(map(str, aa))) for _, r in src_df.iterrows() for q, qq, aa in zip(r["question_id"], r["questions"], r["answers"])}
assert all(m[int(q)] == (str(qq), list(map(str, aa))) for _, r in b.iterrows() for q, qq, aa in zip(r["question_id"], r["questions"], r["answers"]))
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__), "gray": {"size": DEFAULT_SIZE, "rgb": DEFAULT_FILL_COLOR, "encoder": "harness.interventions.make_c1_blank_images.blank_out_images"},
       "source": SRC, "source_sha256": sha(SRC), "recorte": {"path": s1, "sha256": sha(s1), "rows": len(a), "questions": count_questions(a)}, "output": o, "sha256": sha(o), "rows": len(b), "questions": count_questions(b),
       "qids_iguais_fechadas_items_test": True, "perguntas_e_respostas_iguais_ao_teste": True, "load_condition_dataset_n": len(ds), "seconds": round(time.time() - t0, 1)}
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1); print(json.dumps({k: man[k] for k in ("rows", "questions", "load_condition_dataset_n", "sha256", "seconds")}))
