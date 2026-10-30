"""20/10/2026 (m3ae_frozen_vit): queue of the M3AE with frozen ViT for the infer_scheduler_v7 (env lapa). Order SLAKE -> VQA-RAD (by image) -> HEAL ->
Medical-Diff-VQA -> Kvasir, seed 0, 1, 2; then VQA-RAD by question (vqarad_question_split). Each training: 3 predictions (C0, grey C1, swap C2), after training.
RESERVE: 27 862 MiB on trainings (measured peak of the full M3AE + 1,5 GB; rule: it only goes down after measuring the frozen peak); 8 GiB on predictions."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_frozen_cfg as C
H = C.W + "/v7_txt/worktree/harness/v7_txt"; L = C.OUT + "/logs"; R = C.OUT + "/results"; os.makedirs(R, exist_ok=True); os.makedirs(L + "/sched", exist_ok=True)
RES_TRAIN = int(os.environ.get("M3AE_FROZEN_RES_TRAIN", 17638))   # 20/10 19:4x: measured peak of the frozen (SLAKE seed 0, epoch 0 with validation, GPU only ours) 16 102 MiB + 1,5 GB; before 27 862
RES_PRED = 8192
jobs = []
for b in C.ORDEM:
    for s in range(3):
        n = C.name(b, s)
        jobs.append({"name": "train_" + n, "env": "lapa", "reserve_mib": RES_TRAIN, "output": "%s/%s_cost.json" % (L, n), "log": "%s/sched/%s.train.log" % (L, n), "cwd": C.W + "/v7_txt/worktree_m3ae",
                     "cmd": [H + "/m3ae_frozen_train.py", b, str(s)], "skip_order": True})
        for cond in C.BENCH[b]["test"]:
            o = "%s/%s_%s.json" % (R, n, cond)
            jobs.append({"name": "predict_%s_%s" % (n, cond), "env": "lapa", "reserve_mib": RES_PRED, "output": o, "log": "%s/sched/%s_%s.predict.log" % (L, n, cond), "cwd": C.W + "/v7_txt/worktree_m3ae",
                         "cmd": [H + "/m3ae_frozen_predict.py", b, str(s), cond, o], "after": "train_" + n, "skip_order": True})
# 20/10 21:xx (the candidate): trainings before all predictions; the scheduler goes through the list in order and only launches a prediction when no training fits
jobs = [j for j in jobs if j["name"].startswith("train_")] + [j for j in jobs if not j["name"].startswith("train_")]
json.dump(jobs, open(L + "/jobs_m3ae_frozen.json", "w"), indent=1); print(len(jobs), "jobs; training reserve", RES_TRAIN)
