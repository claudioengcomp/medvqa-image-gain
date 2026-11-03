"""v7 (2026-10-09): a M3AE battery on a condition arrow, in the lapa env (python 3.8), WITHOUT touching the M3AE repository:
cwd = v7_txt/worktree_m3ae (14da377 + patch), data_root = a dedicated directory where <dataset_key>_test.arrow is the arrow of the
condition (symbolic link) and train/val point to the original arrows (read-only, for the answer vocabulary).
Same inference as harness/extract_predictions_m3ae.py (forward, argmax, score >= 0.5), copied here only to accept
data_root; provenance timestamp in the JSON.
Usage: --name N --checkpoint CKPT --data-root DIR --dataset-key vqa_kvasir --vqa-label-size 500 --dataset KVASIR --condition OCCLUSION_ART_black --output OUT.json [--arrow-sha256 X]"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, copy, hashlib, json, os, socket, sys, time
WM = f"{_MEDVQA_ROOT}/v7_txt/worktree_m3ae"; HARNESS = f"{_MEDVQA_ROOT}/v7_txt/worktree"
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def build_config(checkpoint, dataset_key, data_root, vqa_label_size, image_size=384):
    sys.argv = ["main.py", "with", "data_root=%s" % data_root, "num_gpus=1", "num_nodes=1", "task_finetune_%s" % dataset_key, "per_gpu_batchsize=16", "clip16", "text_roberta",
                "image_size=%d" % image_size, "test_only=True", "tokenizer=downloaded/roberta-base", "load_path=%s" % checkpoint]
    if vqa_label_size is not None: sys.argv.append("vqa_label_size=%d" % vqa_label_size)
    from m3ae.config import ex
    cap = {}
    @ex.main
    def _c(_config): cap["config"] = _config
    ex.run_commandline(); return cap["config"]
def move(batch, device):
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor): out[k] = [t.to(device) for t in v]
        else: out[k] = v
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--checkpoint", required=True); ap.add_argument("--data-root", required=True); ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--vqa-label-size", type=int, default=None); ap.add_argument("--dataset", required=True); ap.add_argument("--condition", required=True); ap.add_argument("--output", required=True); ap.add_argument("--arrow-sha256", default=None)
    ap.add_argument("--model-label", default="M3AE"); ap.add_argument("--extra", default="")
    a = ap.parse_args(); out_path = os.path.abspath(a.output); data_root = os.path.abspath(a.data_root).rstrip("/") + "/"
    test_arrow = os.path.join(data_root, "%s_test.arrow" % a.dataset_key); arrow_sha = sha256_file(test_arrow)
    if a.arrow_sha256 and arrow_sha != a.arrow_sha256: print("unexpected arrow sha", arrow_sha); sys.exit(5)
    sys.path.insert(0, HARNESS); sys.path.insert(0, WM); os.chdir(WM)
    import torch
    from harness.results_schema import validate_results_dict
    cfg = copy.deepcopy(build_config(a.checkpoint, a.dataset_key, data_root, a.vqa_label_size))
    from m3ae.datamodules.multitask_datamodule import MTDataModule
    from m3ae.modules import M3AETransformerSS
    t0 = time.time(); dm = MTDataModule(cfg, dist=False); dm.setup(stage="test"); model = M3AETransformerSS(cfg); model.eval(); dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    id2answer = dm.dms[0].id2answer; preds = []; n_correct = 0
    with torch.no_grad():
        for batch in dm.test_dataloader():
            b = move(batch, dev); infer = model.infer(b, mask_text=False, mask_image=False); logits = model.vqa_head(infer["multi_modal_cls_feats"])
            targets = torch.zeros(len(logits), cfg["vqa_label_size"], device=dev)
            for i, (labels, scores) in enumerate(zip(batch["vqa_labels"], batch["vqa_scores"])):
                for l, s in zip(labels, scores): targets[i, l] = s
            pid = logits.argmax(dim=-1)
            for i in range(len(pid)):
                p = int(pid[i].item()); correct = float(targets[i, p].item()) >= 0.5; n_correct += int(correct); ga = batch["vqa_answer"][i]
                preds.append({"qid": int(batch["qid"][i]), "predicted": str(id2answer[p]), "gold": str(ga[0]) if ga else "", "correct": bool(correct)})
    n = len(preds); rec = {"model": a.model_label, "dataset": a.dataset, "condition": a.condition, "accuracy": (n_correct / n) if n else 0.0, "n": n, "n_correct": n_correct, "predictions": preds}
    validate_results_dict(rec)
    rec["provenance"] = {"name": a.name, "engine": "m3ae", "checkpoint": a.checkpoint, "checkpoint_sha256": sha256_file(a.checkpoint), "dataset_key": a.dataset_key, "vqa_label_size": cfg["vqa_label_size"], "data_root": data_root,
                         "input_arrow": test_arrow, "input_arrow_resolved": os.path.realpath(test_arrow), "input_arrow_sha256": arrow_sha, "train_val_arrows_for_vocab": {s: os.path.realpath(os.path.join(data_root, "%s_%s.arrow" % (a.dataset_key, s))) for s in ("train", "val")},
                         "worktree_m3ae": WM, "commit": "14da377 + patches/m3ae_uncommitted_14da377.patch", "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
                         "libcuda": {"ld_library_path": os.environ.get("LD_LIBRARY_PATH"), "libcuda": os.path.realpath(os.path.join(os.environ.get("LD_LIBRARY_PATH", "").split(":")[0], "libcuda.so.1")) if os.environ.get("LD_LIBRARY_PATH") else os.path.realpath("/usr/lib/x86_64-linux-gnu/libcuda.so.1")}, "dtype": "M3AE fp32 (Lightning, test_only)", "inference": "copy of harness.extract_predictions_m3ae.extract_predictions with its own data_root (no swap in the M3AE repository)", "t_start": now(), "wall_seconds": round(time.time() - t0, 1), "extra": a.extra}
    json.dump(rec, open(out_path, "w"), indent=1); print("FIM_BATERIA", a.name, a.condition, n, "itens acc=%.4f" % rec["accuracy"], round(time.time() - t0), "s", flush=True)
if __name__ == "__main__": main()
