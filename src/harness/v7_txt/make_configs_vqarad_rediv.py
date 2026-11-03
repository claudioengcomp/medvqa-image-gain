"""17/10/2026 (article_runs, vqarad_image_split): 6 _rediv YAMLs copied from the _radorig ones (v7_txt/configs, A1 recipes), changing only the name, data.arrows
(train/validation/test of the image split, v7_txt/article_runs/data/vqarad_rediv) and output paths (v7_txt/article_runs). Checks via
yaml.safe_load that no other key differs from the source. Does not overwrite. Manifest: v7_txt/article_runs/configs/configs_vqarad_rediv.json."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, pathlib, yaml
SRC = pathlib.Path(f"{_MEDVQA_ROOT}/v7_txt/configs"); OUT = pathlib.Path(f"{_MEDVQA_ROOT}/v7_txt/article_runs/configs"); OUT.mkdir(parents=True, exist_ok=True)
RD = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv"; ARR = {"train": f"{RD}/vqa_vqa_rad_train.arrow", "validation": f"{RD}/vqa_vqa_rad_val.arrow", "test": f"{RD}/vqa_vqa_rad_test.arrow"}
NAMES = ["qwen2vl_lora_txt_vqarad", "qwen2vl_lora_vqarad", "qwen2vl_lora_vt_vqarad", "smolvlm2_lora_txt_vqarad", "smolvlm2_lora_vqarad", "smolvlm2_lora_ft_vqarad"]
ALLOWED = {("run_name",), ("logging", "wandb_run_name"), ("outputs_dir",), ("results_json_dir",), ("data", "arrows", "train"), ("data", "arrows", "validation"), ("data", "arrows", "test")}
def flat(d, p=()):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + (k,))
        else: yield p + (k,), v
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rep = {}
for n in NAMES:
    src = SRC / f"{n}_radorig.yaml"; name = f"{n}_rediv"; dst = OUT / f"{name}.yaml"; assert not dst.exists(), dst
    s = yaml.safe_load(src.read_text()); d = json.loads(json.dumps(s))
    d["run_name"] = name; d.setdefault("logging", {})["wandb_run_name"] = name
    d["outputs_dir"] = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; d["results_json_dir"] = f"{_MEDVQA_ROOT}/v7_txt/article_runs/logs"; d["data"]["arrows"] = dict(ARR)
    a, b = dict(flat(s)), dict(flat(d)); diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}; assert diff <= ALLOWED, (name, diff - ALLOWED)
    assert d["eval"]["split"] == "validation", name
    dst.write_text("# article_runs vqarad_image_split (2026-10-17): copy of %s (sha256 %s); only name, data.arrows (split by image, seed 20260913) and output paths change.\n" % (src, sha(src)[:16]) + yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    assert yaml.safe_load(dst.read_text()) == d
    rep[name] = {"source": str(src), "source_sha256": sha(src), "yaml_sha256": sha(dst), "changed_keys": sorted(".".join(k) for k in diff), "target_modules": d["peft_config"]["target_modules"], "no_image": d.get("data", {}).get("no_image", False), "epochs": d["train"]["epochs"]}
    print(name, "<-", src.name, "| changed:", rep[name]["changed_keys"])
json.dump(rep, open(OUT / "configs_vqarad_rediv.json", "w"), indent=1); print("FIM", len(rep))
