"""v7_txt (14/10/2026, the candidate): HEAL C1 for the 27 fine-tuned rows. Image swapped for the constant grey 384x384 RGB(128,128,128)
from harness.interventions.make_c1_blank_images (the same one as the C1 of Kvasir and of MIMIC and of the grey arrows of the text-only M3AE).
Source (read only): v6_medida/results/arrows/stable_vqa_heal_test.arrow (the arrow of the C0 of the 12 tower rows; 1 446 evaluable in 1 360 images).
Outputs: v7_txt/data/heal_c1/vqa_heal_test_C1.arrow + manifest_heal_c1.json (sha256 of source, output and script);
M3AE data_root in v7_txt/m3ae_roots/heal_C1/ (test -> grey arrow; train/val -> original M3AE arrows, read only, for the vocabulary).
Validation: same rows and questions as the source, same question_id, and load_condition_dataset loads 1 446 items.
HEAL images never leave gpu1."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, socket, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c1_blank_images import blank_out_images, DEFAULT_SIZE, DEFAULT_FILL_COLOR
from harness.interventions.arrow_io import read_arrow, count_questions
from harness.circle_data_bridge import load_condition_dataset
SRC = f"{_MEDVQA_ROOT}/v6_medida/results/arrows/stable_vqa_heal_test.arrow"; OUT = f"{_MEDVQA_ROOT}/v7_txt/data/heal_c1"
M3 = f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows"; ROOT = f"{_MEDVQA_ROOT}/v7_txt/m3ae_roots/heal_C1"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(OUT, exist_ok=False); os.makedirs(ROOT, exist_ok=False)
o = f"{OUT}/vqa_heal_test_C1.arrow"; t0 = time.time(); blank_out_images(SRC, o)
src_df, df = read_arrow(SRC), read_arrow(o)
assert len(df) == len(src_df) and count_questions(df) == count_questions(src_df) == 1446, (len(df), count_questions(df))
qs = lambda d: sorted(q for row in d["question_id"] for q in row)
assert qs(df) == qs(src_df)
ds = load_condition_dataset(o); assert len(ds) == 1446, len(ds)
os.symlink(o, f"{ROOT}/vqa_heal_test.arrow")
for sp in ("train", "val"): os.symlink(f"{M3}/vqa_heal_{sp}.arrow", f"{ROOT}/vqa_heal_{sp}.arrow")
old = f"{M3}/vqa_heal_test_c1.arrow"
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__),
       "gray": {"size": DEFAULT_SIZE, "rgb": DEFAULT_FILL_COLOR, "encoder": "harness.interventions.make_c1_blank_images.blank_out_images"},
       "source": SRC, "source_sha256": sha(SRC), "output": o, "sha256": sha(o), "bytes": os.path.getsize(o), "rows": len(df), "questions": count_questions(df),
       "load_condition_dataset_n": len(ds), "seconds": round(time.time() - t0, 1),
       "m3ae_data_root": ROOT, "m3ae_train_val": {sp: {"path": f"{M3}/vqa_heal_{sp}.arrow", "sha256": sha(f"{M3}/vqa_heal_{sp}.arrow")} for sp in ("train", "val")},
       "conferencia_c1_antigo_m3ae": {"path": old, "sha256": sha(old) if os.path.exists(old) else None}}
man["conferencia_c1_antigo_m3ae"]["igual_ao_novo"] = man["conferencia_c1_antigo_m3ae"]["sha256"] == man["sha256"]
json.dump(man, open(f"{OUT}/manifest_heal_c1.json", "w"), indent=1); print(json.dumps({k: man[k] for k in ("rows", "questions", "load_condition_dataset_n", "sha256", "seconds", "conferencia_c1_antigo_m3ae")}))
