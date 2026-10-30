"""v7_mimic (2026-10-13, the candidate): M3AE arrows with the image swapped for the constant grey 384x384 RGB(128,128,128) from
harness/interventions/make_c1_blank_images.py (the same one from the C1 and from the text-only M3AE of Kvasir, SLAKE and HEAL; tmp/v7/make_gray_arrows.py),
for MIMIC training, validation and test. Sources (read only): v7_mimic/data/vqa_mimic_{train,val,test}.arrow.
Output: v7_mimic/m3ae_arrows_gray/vqa_mimic_{split}_gray.arrow + data_root/vqa_mimic_{split}.arrow (links) + manifest_gray_arrows.json.
No MIMIC pixel enters the grey arrows; the questions and answers do (credentialed: they stay on gpu1)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, socket, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c1_blank_images import blank_out_images, DEFAULT_SIZE, DEFAULT_FILL_COLOR
from harness.interventions.arrow_io import read_arrow, count_questions
SRC = f"{_MEDVQA_ROOT}/v7_mimic/data"; OUT = f"{_MEDVQA_ROOT}/v7_mimic/m3ae_arrows_gray"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(OUT + "/data_root", exist_ok=False)
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__), "gray": {"size": DEFAULT_SIZE, "rgb": DEFAULT_FILL_COLOR, "encoder": "harness.interventions.make_c1_blank_images.blank_out_images (JPEG)"}, "files": {}}
for split in ("train", "val", "test"):
    s = f"{SRC}/vqa_mimic_{split}.arrow"; o = f"{OUT}/vqa_mimic_{split}_gray.arrow"
    t0 = time.time(); blank_out_images(s, o)
    src_df = read_arrow(s); df = read_arrow(o)
    assert len(df) == len(src_df) and count_questions(df) == count_questions(src_df), (split, len(df), len(src_df))
    os.symlink(f"../vqa_mimic_{split}_gray.arrow", f"{OUT}/data_root/vqa_mimic_{split}.arrow")
    man["files"][os.path.basename(o)] = {"source": s, "source_sha256": sha(s), "sha256": sha(o), "bytes": os.path.getsize(o), "rows": len(df), "questions": count_questions(df), "seconds": round(time.time() - t0, 1)}
    print(split, man["files"][os.path.basename(o)], flush=True)
json.dump(man, open(f"{OUT}/manifest_gray_arrows.json", "w"), indent=1)
print("FIM")
