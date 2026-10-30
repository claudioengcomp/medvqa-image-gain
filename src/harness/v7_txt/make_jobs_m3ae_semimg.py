"""22/10/2026 (m3ae_no_image): queue of the M3AE without image. Trainings first (VQA-RAD by image -> SLAKE -> HEAL -> Medical-Diff-VQA -> Kvasir -> VQA-RAD by question;
seeds 0, 1, 2), then the predictions. Training reserve 8 GiB (peak measured in the trial run 6 584 MiB per PID + 1,5 GB); prediction 6 GiB."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_semimg_cfg as C
H = C.W + "/v7_txt/worktree/harness/v7_txt"; L = C.O + "/logs"; R = C.O + "/results"; os.makedirs(R, exist_ok=True); os.makedirs(L + "/sched", exist_ok=True)
RES_TRAIN = 8192
RES_PRED = 6144
tr, pr = [], []
for b in C.ORDEM:
    for s in range(3):
        n = C.name(b, s); cwd = C.W + "/v7_txt"
        tr.append({"name": "train_" + n, "env": "lapa", "reserve_mib": RES_TRAIN, "output": "%s/%s_cost.json" % (L, n), "log": "%s/sched/%s.train.log" % (L, n), "cwd": cwd, "cmd": [H + "/m3ae_semimg_train.py", b, str(s)], "skip_order": True})
        o = "%s/%s_TXT.json" % (R, n)
        pr.append({"name": "predict_" + n, "env": "lapa", "reserve_mib": RES_PRED, "output": o, "log": "%s/sched/%s.predict.log" % (L, n), "cwd": cwd, "cmd": [H + "/m3ae_semimg_predict.py", b, str(s), o], "after": "train_" + n, "skip_order": True})
jobs = tr + pr
json.dump(jobs, open(L + "/jobs_m3ae_semimg.json", "w"), indent=1); print(len(tr), "treinos +", len(pr), "predicoes")
