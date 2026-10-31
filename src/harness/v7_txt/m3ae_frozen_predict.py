"""20/10/2026 (m3ae_frozen_vit; lapa env): M3AE prediction with frozen ViT under one condition. Usage: m3ae_frozen_predict.py <bench> <seed> <cond> <output>.
Before predicting: locates the best checkpoint (save_top_k=1, val/the_metric) and checks that the sha256 of the ViT's state_dict in the best and in the last
is identical to the one recorded at the start of training (freeze.json); if it differs, aborts without predicting. Timestamp: starting checkpoint (path, sha),
final checkpoint (m3ae_infer_v7 writes path and sha), ViT shas before/after."""
import glob, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_frozen_cfg as C
import torch
b, s, cond, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]; B = C.BENCH[b]; NAME = C.name(b, s); LOG = C.OUT + "/logs"
cands = sorted(glob.glob("%s/%s-seed%d-from_*/*/checkpoints/epoch=*.ckpt" % (C.ck_root(b), B["task"], s)), key=lambda p: os.path.getmtime(os.path.dirname(os.path.dirname(p))))
nota = "" if len(cands) == 1 else "; best checkpoints from other versions (reruns) not used: " + ", ".join(cands[:-1])
assert cands, "no best checkpoint"; ck = cands[-1]; last = os.path.join(os.path.dirname(ck), "last.ckpt")
def vit_sha(path):
    sd = torch.load(path, map_location="cpu")["state_dict"]; sub = {k[len("vision_encoder."):]: v for k, v in sd.items() if k.startswith("vision_encoder.")}; h = hashlib.sha256()
    for k in sorted(sub): h.update(k.encode()); h.update(sub[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest(), len(sub)
rep = json.load(open("%s/%s_freeze.json" % (LOG, NAME))); antes = rep["vision_encoder_state_dict_sha256_antes"]
depois, nk = vit_sha(ck); depois_last = vit_sha(last)[0] if os.path.exists(last) else None
ver = {"vit_sha_antes": antes, "vit_sha_best": depois, "vit_sha_last": depois_last, "vit_tensores": nk, "identicos": antes == depois and depois_last in (None, depois)}
json.dump(dict(ver, best=ck, last=last), open("%s/%s_vit_check.json" % (LOG, NAME), "w"), indent=1)
if not ver["identicos"]: print("ViT CHANGED during training:", ver); sys.exit(4)
dr = "%s/data_roots/%s_%s/" % (C.OUT, b, cond); os.makedirs(dr, exist_ok=True)
for sp in ("train", "val", "test"):   # test -> arrow of the condition; training/validation -> the same as training (vocabulary)
    l = dr + "%s_%s.arrow" % (B["key"], sp); t = B["test"][cond] if sp == "test" else "%s/%s_%s.arrow" % (B["train_root"], B["key"], sp)
    if not os.path.islink(l): os.symlink(t, l)
    assert os.path.realpath(l) == os.path.realpath(t), (l, t)
sha = hashlib.sha256(open(B["test"][cond], "rb").read()).hexdigest()
extra = ("m3ae_frozen_vit: M3AE with frozen visual encoder; cond %s; ckpt_partida %s sha256 %s; vit_sha256 before=after=%s (best and last); cost=%s/%s_cost.json; freeze=%s/%s_freeze.json%s"
         % (cond, C.START_CKPT, C.START_SHA, depois, LOG, NAME, LOG, NAME, nota))
cmd = [sys.executable, "-u", C.W + "/v7_txt/worktree/harness/v7_txt/m3ae_infer_v7.py", "--name", NAME, "--checkpoint", ck, "--data-root", dr, "--dataset-key", B["key"], "--vqa-label-size", str(B["ls"]),
       "--dataset", B["ds"], "--condition", "C2" if cond.startswith("C2") else cond, "--output", out, "--model-label", "M3AE_FROZEN", "--arrow-sha256", sha, "--extra", extra]
print(" ".join(cmd), flush=True); sys.exit(subprocess.call(cmd))
