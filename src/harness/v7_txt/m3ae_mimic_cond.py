"""v7_mimic B: predicao do M3AE_MIMIC_seed<s> numa condicao do Occlusion study/Occlusion (lapa). data_root proprio por condicao em
v7_mimic/m3ae_arrows/cond/<COND>/ (links: vqa_mimic_test.arrow -> arrow da condicao; train/val -> data/ para o vocabulario), checkpoint best
do Lightning, m3ae_infer_v7 com --arrow-sha256 esperado. Uso: m3ae_mimic_cond.py <seed> <COND> <arrow> <sha256> <output.json>"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import glob, os, subprocess, sys
seed, cond, arrow, sha, out = sys.argv[1:6]; R = f"{_MEDVQA_ROOT}/v7_mimic"
cands = glob.glob("%s/m3ae/task_finetune_vqa_mimic-seed%s-from_*/*/checkpoints/epoch=*.ckpt" % (R, seed))
if len(cands) != 1: print("esperado exatamente 1 checkpoint best, achei", cands); sys.exit(3)
dr = "%s/m3ae_arrows/cond/%s" % (R, cond); os.makedirs(dr, exist_ok=True)
for name, tgt in (("vqa_mimic_test.arrow", arrow), ("vqa_mimic_train.arrow", R + "/data/vqa_mimic_train.arrow"), ("vqa_mimic_val.arrow", R + "/data/vqa_mimic_val.arrow")):
    l = os.path.join(dr, name)
    if not os.path.islink(l): os.symlink(tgt, l)
    assert os.path.realpath(l) == os.path.realpath(tgt), (l, tgt)
name = "M3AE_MIMIC_seed%s" % seed
cmd = [sys.executable, "-u", f"{_MEDVQA_ROOT}/v7_txt/worktree/harness/v7_txt/m3ae_infer_v7.py", "--name", name + "_" + cond, "--checkpoint", cands[0], "--data-root", dr + "/", "--dataset-key", "vqa_mimic", "--vqa-label-size", "82",
       "--dataset", "MIMIC", "--condition", cond, "--output", out, "--model-label", "M3AE", "--arrow-sha256", sha, "--extra", "bateria Occlusion study/Occlusion do MIMIC (HANDOFF_OCCLUSION_STUDY_GPU.md); arrow " + arrow]
print(" ".join(cmd), flush=True); sys.exit(subprocess.call(cmd))
