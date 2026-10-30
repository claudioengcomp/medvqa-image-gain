"""P1.3 prediction (lapa): locates the 'best' checkpoint saved by Lightning in v7_txt/m3ae/<run>/<versao>/checkpoints/epoch=*.ckpt
(save_top_k=1, monitor val/the_metric, as in main.py), and runs m3ae_infer_v7 on the GREY test set (data_root = m3ae_arrows/data_root).
Usage: m3ae_txt_predict.py <slack|kvasir|heal> <seed> <DATASET> <output.json>"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import glob, json, os, subprocess, sys
bench, seed, DS, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]; mode = sys.argv[5] if len(sys.argv) > 5 else "gray"; assert mode in ("gray", "img", "rediv", "redivgray", "redivc2", "qlevel", "qlevelgray", "qlevelc2"), mode   # 17/10: rediv/redivgray (article_runs, vqarad_image_split)   # 13/10
R = f"{_MEDVQA_ROOT}/v7_txt"; task = {"slack": "task_finetune_vqa_slack", "kvasir": "task_finetune_vqa_kvasir", "heal": "task_finetune_vqa_heal", "vqa_rad": "task_finetune_vqa_vqa_rad"}[bench]
LS = {"slack": 222, "kvasir": 500, "heal": 288, "vqa_rad": 498}[bench]
CK, DROOT, COND, NAME, LABEL, EXTRA = (("m3ae", R + "/m3ae_arrows/data_root/", "TXT", "M3AE_TXT_%s_seed%s" % (DS, seed), "M3AE_TXT", "braco so-texto do M3AE: imagem cinza constante no treino/val/teste") if mode == "gray" else
                                       ("m3ae_gpu1", f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows/", "C0", "M3AE_SCRATCH_%s_seed%s_gpu1" % (DS, seed), "M3AE", "retreino gpu1 (13/10) com a receita publicada; nao substitui a predicao do Colab"))
if mode in ("qlevel", "qlevelgray", "qlevelc2"):   # 19/10 (vqarad_question_split): split by question
    PJ = R + "/article_runs"; QD = PJ + "/data/vqarad_qlevel/"; CK, DROOT, COND, NAME, LABEL, EXTRA = {"qlevel": ("article_runs/m3ae_qlevel", QD, "C0", "M3AE_SCRATCH_VQARAD_QLEVEL_seed%s" % seed, "M3AE", "VQA-RAD dividido por pergunta (vqarad_question_split)"), "qlevelgray": ("article_runs/m3ae_txt_qlevel", QD + "data_root_cinza/", "TXT", "M3AE_TXT_VQARAD_QLEVEL_seed%s" % seed, "M3AE_TXT", "VQA-RAD dividido por pergunta, so-texto (vqarad_question_split)"), "qlevelc2": ("article_runs/m3ae_qlevel", QD + "c2/data_root/", "C2", "M3AE_SCRATCH_VQARAD_QLEVEL_seed%s" % seed, "M3AE", "VQA-RAD dividido por pergunta, imagem trocada (vqarad_question_split)")}[mode]
elif mode == "redivc2":   # 17/10 (decision of the candidate, 7-B): same rediv checkpoint, test with swapped image (derangement seed 42)
    PJ = R + "/article_runs"; CK, DROOT, COND, NAME, LABEL, EXTRA = ("article_runs/m3ae", PJ + "/data/vqarad_rediv/c2/data_root/", "C2", "M3AE_SCRATCH_VQARAD_REDIV_seed%s" % seed, "M3AE", "VQA-RAD redividido, imagem trocada (C2_manifest_VQARAD_REDIV.json)")
elif mode in ("rediv", "redivgray"):   # 17/10 (article_runs, vqarad_image_split): checkpoints and arrows from v7_txt/article_runs; C0 on the new test set (rediv) or TXT on the new grey test set (redivgray)
    PJ = R + "/article_runs"; CK, DROOT, COND, NAME, LABEL, EXTRA = (("article_runs/m3ae", PJ + "/data/vqarad_rediv/", "C0", "M3AE_SCRATCH_VQARAD_REDIV_seed%s" % seed, "M3AE", "VQA-RAD redividido por imagem (article_runs, vqarad_image_split), receita publicada") if mode == "rediv" else
                                         ("article_runs/m3ae_txt", PJ + "/data/vqarad_rediv/data_root_cinza/", "TXT", "M3AE_TXT_VQARAD_REDIV_seed%s" % seed, "M3AE_TXT", "VQA-RAD redividido por imagem (article_runs, vqarad_image_split), so-texto (imagem cinza)"))
cands = [p for p in glob.glob("%s/%s/%s-seed%s-from_*/*/checkpoints/epoch=*.ckpt" % (R, CK, task, seed))]
NOTA_CK = ""
if len(cands) > 1 and mode in ("rediv", "redivgray", "redivc2", "qlevel", "qlevelgray", "qlevelc2"):   # 17/10 10:5x: training relaunched after OOM (third parties) leaves one best per version; use the one from the most recent round (the one that finished)
    cands = sorted(cands, key=lambda p: os.path.getmtime(os.path.dirname(os.path.dirname(p)))); NOTA_CK = "; checkpoints best de outras versoes (relancamentos apos OOM) nao usados: " + ", ".join(cands[:-1]); cands = cands[-1:]
if len(cands) != 1: print("esperado exatamente 1 checkpoint best, achei", cands); sys.exit(3)
ck = cands[0]; name = NAME
cmd = [sys.executable, "-u", R + "/worktree/harness/v7_txt/m3ae_infer_v7.py", "--name", name, "--checkpoint", ck, "--data-root", DROOT, "--dataset-key", "vqa_" + bench, "--vqa-label-size", str(LS),
       "--dataset", DS, "--condition", COND, "--output", out, "--model-label", LABEL, "--extra", EXTRA + NOTA_CK + "; cost=" + (R + "/article_runs/logs/" + name if mode in ("rediv", "redivgray", "redivc2", "qlevel", "qlevelgray", "qlevelc2") else R + "/logs/" + (name.replace(DS, bench) if mode == "gray" else name)) + "_cost.json"]
print(" ".join(cmd), flush=True); sys.exit(subprocess.call(cmd))
