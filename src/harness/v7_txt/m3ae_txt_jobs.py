"""P1.3 (2026-10-09): M3AE text-only jobs for the infer_scheduler_v7 (env lapa): 9 trainings (slack, kvasir, heal x seeds 0-2;
exclusive GPU, 27 GB reserve; wrapper m3ae_txt_train.py -> run_m3ae_txt.sh; 'output' = <NAME>_cost.json) and 9 predictions on the
grey test set (m3ae_txt_predict.py locates the best checkpoint saved by Lightning and calls m3ae_infer_v7 with grey data_root)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import json, os
R = f"{_MEDVQA_ROOT}/v7_txt"; H = R + "/worktree/harness/v7_txt"
jobs = []
for bench, DS in (("slack", "SLAKE"), ("heal", "HEAL"), ("kvasir", "KVASIR")):
    for s in range(3):
        name = "M3AE_TXT_%s_seed%d" % (bench, s); line = "M3AE_TXT_%s_seed%d" % (DS, s)
        jobs.append({"name": "train_" + name, "env": "lapa", "reserve_mib": 27 * 1024, "exclusive": True, "output": "%s/logs/%s_cost.json" % (R, name), "log": "%s/logs/%s.sched.log" % (R, name), "cwd": R + "/worktree_m3ae",
                     "cmd": [H + "/m3ae_txt_train.py", bench, str(s)], "skip_order": True})
        jobs.append({"name": "predict_" + name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s_TXT.json" % (R, line), "log": "%s/logs/predict_%s.log" % (R, name), "cwd": R + "/worktree_m3ae",
                     "cmd": [H + "/m3ae_txt_predict.py", bench, str(s), DS, "%s/results/%s_TXT.json" % (R, line)], "after": "train_" + name, "skip_order": True})
# 13/10 (the candidate, Baseline): (a) M3AE text-only on VQA-RAD x3; (b) retraining of the M3AE on SLAKE and VQA-RAD, seeds 1 and 2, published recipe, suffix _gpu1,
# C0 prediction. Reserve 27 862 MiB (measured peak of the M3AE + 1,5 GB); no GPU exclusivity (decision of 12/10).
RES = 27862
for s in range(3):
    name = "M3AE_TXT_vqa_rad_seed%d" % s; line = "M3AE_TXT_VQARAD_seed%d" % s
    jobs.append({"name": "train_" + name, "env": "lapa", "reserve_mib": RES, "output": "%s/logs/%s_cost.json" % (R, name), "log": "%s/logs/%s.sched.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_train.py", "vqa_rad", str(s)], "skip_order": True})
    jobs.append({"name": "predict_" + name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s_TXT.json" % (R, line), "log": "%s/logs/predict_%s.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_predict.py", "vqa_rad", str(s), "VQARAD", "%s/results/%s_TXT.json" % (R, line)], "after": "train_" + name, "skip_order": True})
# 14/10 (the candidate): _gpu1 retrainings CANCELLED -- the Colab weights for seeds 1 and 2 (SLAKE, VQA-RAD) exist on Drive and were copied to
# doutorado/M3AE/checkpoints_colab/; no _gpu1 job is generated. The block stays below, disabled, for the record.
for bench, DS in (() if True else (("slack", "SLAKE"), ("vqa_rad", "VQARAD"))):
    for s in (1, 2):
        name = "M3AE_SCRATCH_%s_seed%d_gpu1" % (DS, s)
        jobs.append({"name": "train_" + name, "env": "lapa", "reserve_mib": RES, "output": "%s/logs/%s_cost.json" % (R, name), "log": "%s/logs/%s.sched.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_train.py", bench, str(s), "img"], "skip_order": True})
        jobs.append({"name": "predict_" + name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s_C0.json" % (R, name), "log": "%s/logs/predict_%s.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_predict.py", bench, str(s), DS, "%s/results/%s_C0.json" % (R, name), "img"], "after": "train_" + name, "skip_order": True})
# 14/10 (the candidate): reproduction of the VQA-RAD C0, seeds 1 and 2, with the Colab weights (commit 6cc9eac; Drive copy checked by md5),
# on the original test set (own data_root with links to the original arrows, read-only). Item-by-item comparison with
# doutorado/medvqa-main/results/M3AE_SCRATCH_VQARAD_seed<s>_C0.json (band between environments 2.7-5.9 %).
VR = R + "/m3ae_roots/vqarad_C0/"; ORIG_SHA = "e1e464f0e99dc21ce00c37117e36288c"   # md5 of vqa_vqa_rad_test.arrow == .orig_backup (14/10)
# 14/10 (the candidate): priority 1 -- both jobs moved to make_jobs_c0repro.py (target c0repro); block disabled here, for the record.
for s in (() if True else (1, 2)):
    name = "M3AE_SCRATCH_VQARAD_seed%d_C0_colabrepro" % s; ck = f"{_MEDVQA_ROOT}/doutorado/M3AE/checkpoints_colab/M3AE_SCRATCH_VQARAD_seed%d.ckpt" % s
    jobs.append({"name": name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s.json" % (R, name), "log": "%s/logs/%s.log" % (R, name), "cwd": R + "/worktree_m3ae", "skip_order": True,
                 "cmd": [H + "/m3ae_infer_v7.py", "--name", name, "--checkpoint", ck, "--data-root", VR, "--dataset-key", "vqa_vqa_rad", "--vqa-label-size", "498", "--dataset", "VQARAD", "--condition", "C0",
                         "--output", "%s/results/%s.json" % (R, name), "--model-label", "M3AE_SCRATCH", "--extra", "C0 reproduction, Colab weights 6cc9eac; md5 test arrow %s" % ORIG_SHA]})
json.dump(jobs, open(R + "/logs/jobs_m3ae_txt.json", "w"), indent=1); print(len(jobs), "jobs")
