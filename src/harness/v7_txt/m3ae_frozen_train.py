"""20/10/2026 (m3ae_frozen_vit; lapa env): M3AE training with the ViT frozen. Usage: m3ae_frozen_train.py <bench> <seed>. Inherits CUDA_VISIBLE_DEVICES from the scheduler.
Same command as run_m3ae_txt.sh / run_m3ae_mimic.sh (worktree_m3ae's main.py) + freeze_vision_encoder=True freeze_report_path=<logs>/<NAME>_freeze.json.
Cost: nvidia-smi only of the training GPU every 5 s -> <NAME>_cost.json (like the previous M3AEs)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import json, os, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_frozen_cfg as C
b, s = sys.argv[1], int(sys.argv[2]); B = C.BENCH[b]; NAME = C.name(b, s); LOG = C.OUT + "/logs"; CK = C.ck_root(b); os.makedirs(LOG, exist_ok=True); os.makedirs(CK, exist_ok=True)
TK = B["tk"].get(s, "clip"); dev = os.environ.get("CUDA_VISIBLE_DEVICES", ""); WM = C.W + "/v7_txt/worktree_m3ae"
DR = "%s/data_roots/%s_train/" % (C.OUT, b); os.makedirs(DR, exist_ok=True)
for sp in ("train", "val", "test"):   # training data_root: links to the canonical arrows (test = C0; only validation chooses the checkpoint)
    l = DR + "%s_%s.arrow" % (B["key"], sp); t = (B["test"]["C0"] if sp == "test" else "%s/%s_%s.arrow" % (B["train_root"], B["key"], sp))
    if not os.path.islink(l): os.symlink(t, l)
    assert os.path.realpath(l) == os.path.realpath(t), (l, t)
REP = "%s/%s_freeze.json" % (LOG, NAME)
cmd = [sys.executable, "-u", "main.py", "with", "data_root=" + DR, "num_gpus=1", "num_nodes=1", B["task"], "per_gpu_batchsize=16", "clip16", "text_roberta", "image_size=384",
       "tokenizer=downloaded/roberta-base", "load_path=downloaded/pretrained/m3ae.ckpt", "seed=%d" % s, "vqa_label_size=%d" % B["ls"], 'train_transform_keys=["%s"]' % TK,
       "default_root_dir=" + CK, "log_dir=" + CK + "/result", "freeze_vision_encoder=True", "freeze_report_path=" + REP]
NV = f"{_MEDVQA_ROOT}/nvidia-580.173.02/nvidia-smi" if os.path.exists("/proc/driver/nvidia/version") and "580.173.02" in open("/proc/driver/nvidia/version").read() else "nvidia-smi"
mem = []; stop = threading.Event()
def poll():
    while not stop.is_set():
        try: mem.append(int(subprocess.run([NV, "-i", dev, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.split()[0]))
        except Exception: pass
        stop.wait(5)
th = threading.Thread(target=poll, daemon=True); th.start(); t0 = time.time()
print("[m3ae_frozen_train] %s %s bench=%s seed=%d task=%s ls=%d tk=%s data_root=%s placa=%s\n%s" % (time.strftime("%F %T"), NAME, b, s, B["task"], B["ls"], TK, DR, dev, " ".join(cmd)), flush=True)
rc = subprocess.call(cmd, cwd=WM, env=dict(os.environ, WANDB_MODE="disabled")); stop.set(); th.join(10); t = time.time() - t0
json.dump({"run_name": NAME, "rc": rc, "task": B["task"], "seed": s, "vqa_label_size": B["ls"], "train_transform_keys": [TK], "data_root": DR, "placa": dev, "train_wall_time_s": round(t),
           "train_peak_memory_gb": round(max(mem) / 1024, 3) if mem else None, "freeze_report": REP, "start_ckpt": C.START_CKPT, "start_ckpt_sha256": C.START_SHA,
           "note": "pico = nvidia-smi da placa do treino (inclui terceiros nessa placa)", "cmd": cmd}, open("%s/%s_cost.json" % (LOG, NAME) if rc == 0 else "%s/%s_cost_FALHOU.json" % (LOG, NAME), "w"), indent=2)
print("FIM_TREINO_M3AE_FROZEN", NAME, "rc", rc, "s", round(t), flush=True); sys.exit(rc)
