"""22/10/2026 (m3ae_no_image; env lapa): training of the M3AE text-only WITHOUT image. Usage: m3ae_semimg_train.py <base> <seed>. Same command as the published grey text-only
(run_m3ae_txt.sh / run_m3ae_mimic.sh: main.py, per_gpu_batchsize 16, clip16, text_roberta, image_size 384, tokenizer roberta-base, load_path m3ae.ckpt,
published vqa_label_size and transform, grey data_root), in worktree_m3ae_semimg, plus no_image=True."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_semimg_cfg as C
b, s = sys.argv[1], int(sys.argv[2]); task, ls, key, ds, tkd, DR = C.B[b]; NAME = C.name(b, s); LOG = C.O + "/logs"; CK = C.ck_root(b); os.makedirs(LOG, exist_ok=True); os.makedirs(CK, exist_ok=True)
TK = tkd.get(s, "clip")
cmd = [sys.executable, "-u", "main.py", "with", "data_root=" + DR, "num_gpus=1", "num_nodes=1", task, "per_gpu_batchsize=16", "clip16", "text_roberta", "image_size=384",
       "tokenizer=downloaded/roberta-base", "load_path=downloaded/pretrained/m3ae.ckpt", "seed=%d" % s, "vqa_label_size=%d" % ls, 'train_transform_keys=["%s"]' % TK,
       "default_root_dir=" + CK, "log_dir=" + CK + "/result", "no_image=True"]
print("[m3ae_semimg_train] %s %s base=%s seed=%d task=%s ls=%d tk=%s data_root=%s gpu=%s\n%s" % (time.strftime("%F %T"), NAME, b, s, task, ls, TK, DR, os.environ.get("CUDA_VISIBLE_DEVICES"), " ".join(cmd)), flush=True)
t0 = time.time(); rc = subprocess.call(cmd, cwd=C.WM, env=dict(os.environ, WANDB_MODE="disabled")); t = time.time() - t0
json.dump({"run_name": NAME, "rc": rc, "task": task, "seed": s, "vqa_label_size": ls, "train_transform_keys": [TK], "data_root": DR, "placa": os.environ.get("CUDA_VISIBLE_DEVICES"), "train_wall_time_s": round(t),
           "codigo": C.WM, "no_image": True, "start_ckpt": C.START_CKPT, "start_ckpt_sha256": C.START_SHA, "cmd": cmd}, open("%s/%s_cost.json" % (LOG, NAME) if rc == 0 else "%s/%s_cost_FALHOU.json" % (LOG, NAME), "w"), indent=2)
print("FIM_TREINO_M3AE_SEMIMG", NAME, "rc", rc, "s", round(t), flush=True); sys.exit(rc)
