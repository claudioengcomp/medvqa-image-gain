"""v7_txt (2026-10-13, candidato: Baseline no VQA-RAD): arrows do M3AE com a imagem cinza constante 384x384 RGB(128,128,128) de
harness/interventions/make_c1_blank_images.py (a mesma de tmp/v7/make_gray_arrows.py para Kvasir, SLAKE e HEAL), para treino, validacao e
teste do VQA-RAD. Fontes (so leitura): M3AE/data/finetune_arrows/vqa_vqa_rad_{train,val,test}.arrow. Saida: v7_txt/m3ae_arrows/vqa_vqa_rad_*_gray.arrow,
links em m3ae_arrows/data_root/ e manifesto proprio manifest_gray_arrows_vqarad.json (o manifesto de 09/10 nao e tocado)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, socket, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c1_blank_images import blank_out_images, DEFAULT_SIZE, DEFAULT_FILL_COLOR
from harness.interventions.arrow_io import read_arrow, count_questions
SRC = f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows"; OUT = f"{_MEDVQA_ROOT}/v7_txt/m3ae_arrows"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__), "gray": {"size": DEFAULT_SIZE, "rgb": DEFAULT_FILL_COLOR, "encoder": "harness.interventions.make_c1_blank_images.blank_out_images (JPEG)"}, "files": {}}
for split in ("train", "val", "test"):
    s = f"{SRC}/vqa_vqa_rad_{split}.arrow"; o = f"{OUT}/vqa_vqa_rad_{split}_gray.arrow"; l = f"{OUT}/data_root/vqa_vqa_rad_{split}.arrow"
    assert not os.path.exists(o) and not os.path.lexists(l), (o, l)
    t0 = time.time(); blank_out_images(s, o); src = read_arrow(s); df = read_arrow(o)
    assert len(df) == len(src) and count_questions(df) == count_questions(src)
    os.symlink(f"../vqa_vqa_rad_{split}_gray.arrow", l)
    man["files"][os.path.basename(o)] = {"source": s, "source_sha256": sha(s), "sha256": sha(o), "bytes": os.path.getsize(o), "rows": len(df), "questions": count_questions(df), "seconds": round(time.time() - t0, 1)}
    print(split, man["files"][os.path.basename(o)], flush=True)
json.dump(man, open(f"{OUT}/manifest_gray_arrows_vqarad.json", "w"), indent=1); print("FIM")
