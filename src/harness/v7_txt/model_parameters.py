"""20/10/2026 (model_parameters; CPU only): number of parameters per model and approach, counted in the code. VLM: med_vlm.models.loader.load_model_bundle (same class and
training precision: bfloat16) + med_vlm.models.peft_adapters.apply_peft_adapter with the peft_config from each approach's YAML + measure_model_params (the same
function that run_experiment.py calls right after the adapter). Model reloaded from scratch for each approach. PubMedBERT: AutoModelForSequenceClassification
(fixed revision of bert_question_only.py) with num_labels = n_rotulos from each benchmark's training (BERT2 result, seed 0); all parameters trainable.
Peft_config checked identical across the benchmarks (all training YAMLs of the paper's rows). Output: analysis/model_parameters.json."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, glob, hashlib, json, os, re, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, yaml, transformers, peft
import med_vlm.enums
from med_vlm.models.loader import load_model_bundle
from med_vlm.models.peft_adapters import apply_peft_adapter
from med_vlm.metrics.efficiency import measure_model_params
from transformers import AutoModelForSequenceClassification
W = f"{_MEDVQA_ROOT}"; P0C = W + "/doutorado/medvqa-main/med_vlm/configs"; V7C = W + "/v7_txt/configs"; PJ = W + "/v7_txt/article_runs"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ABORD = {"QWEN2VL": [("text-decoder-only", f"{V7C}/qwen2vl_lora_txt_slake.yaml"), ("frozen visual encoder", f"{P0C}/qwen2vl_lora_slake.yaml"), ("fine-tuned visual encoder", f"{P0C}/qwen2vl_lora_vt_slake.yaml")],
         "SMOLVLM": [("text-decoder-only", f"{V7C}/smolvlm2_lora_txt_slake.yaml"), ("frozen visual encoder", f"{P0C}/smolvlm2_lora_ft_slake.yaml"), ("fine-tuned visual encoder", f"{P0C}/smolvlm2_lora_slake.yaml")]}
out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "versoes": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}, "vlm": {}, "pubmedbert": {}, "peft_config_entre_bases": {}}
# check: peft_config of each approach identical across all training YAMLs with the same prefix (all benchmarks)
for fam, L in ABORD.items():
    for ab, ref in L:
        stem = os.path.basename(ref).rsplit("_slake.yaml", 1)[0]; pref = yaml.safe_load(open(ref))["peft_config"]; difs = []; vistos = []
        for p in sorted(set(glob.glob(f"{P0C}/{stem}_*.yaml") + glob.glob(f"{V7C}/{stem}_*.yaml") + glob.glob(f"{W}/v7_mimic/configs/{stem}_*.yaml") + glob.glob(f"{PJ}/configs/{stem}_*.yaml"))):
            rest = os.path.basename(p)[len(stem) + 1:-5]
            if "_" in rest and rest not in ("vqa_rad", "heal_treino") and not rest.startswith(("vqarad", "vqa_rad")): continue   # so <stem>_<base>.yaml (exclui variantes como encattn64)
            pc = (yaml.safe_load(open(p)) or {}).get("peft_config"); vistos.append(os.path.basename(p))
            if pc != pref: difs.append([os.path.basename(p), pc])
        out["peft_config_entre_bases"][f"{fam}|{ab}"] = {"referencia": os.path.basename(ref), "yaml_conferidas": vistos, "diferentes": difs}
for fam, L in ABORD.items():
    for ab, ycfg in L:
        load_model_bundle.cache_clear(); b = load_model_bundle(getattr(med_vlm.enums.ModelName, fam)); base = measure_model_params(b.model)
        cfg = yaml.safe_load(open(ycfg)); m = apply_peft_adapter(b.model, peft_method=med_vlm.enums.PEFTMethod.LORA, peft_config=cfg.get("peft_config", {}) or {}); r = measure_model_params(m)
        alvos = collections.Counter(); 
        for nm, p in m.named_parameters():
            if p.requires_grad and ".lora_A." in nm: alvos[("encoder" if ("visual" in nm or "vision" in nm) else "decodificador") + ":" + re.sub(r"\.\d+\.", ".N.", nm.split(".lora_A.")[0])] += 1
        dt = sorted({str(p.dtype) for p in m.parameters()})
        out["vlm"][f"{fam}|{ab}"] = {"yaml": ycfg, "yaml_sha256": sha(ycfg), "peft_config": cfg.get("peft_config"), "total_base": base["total_params"], "total_com_adaptadores": r["total_params"], "treinaveis": r["trainable_params"],
                                     "modulos_alvo": dict(sorted(alvos.items())), "dtypes": dt, "modelo_hf": b.model.config._name_or_path if hasattr(b.model, "config") else None}
        print(fam, ab, r, flush=True); del m, b
BERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"; REV = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
for base in ("KVASIR", "SLAKE", "HEAL_TREINO", "MIMIC", "VQARAD_REDIV", "VQARAD_QLEVEL"):
    f = f"{PJ}/results/BERT2_question_only_{base}_seed0.json"; n = json.load(open(f))["treino"]["n_rotulos"]
    m = AutoModelForSequenceClassification.from_pretrained(BERT, revision=REV, num_labels=n); r = measure_model_params(m); cab = sum(p.numel() for nm, p in m.named_parameters() if nm.startswith("classifier"))
    out["pubmedbert"][base] = {"n_rotulos": n, "fonte_n_rotulos": f, "total": r["total_params"], "treinaveis": r["trainable_params"], "cabeca": cab}; print("PubMedBERT", base, n, r["total_params"], flush=True)
out["script"] = os.path.abspath(__file__); out["script_sha256"] = sha(__file__)
json.dump(out, open(f"{PJ}/analysis/model_parameters.json", "w"), indent=1, default=str); print("FIM")
