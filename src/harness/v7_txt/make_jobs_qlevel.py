"""19/10/2026 (vqarad_question_split): VQA-RAD queues by question. jobs_m3ae_qlevel: M3AE with image (qlevel) and text-only (qlevelgray), 3 seeds, training + prediction.
jobs_qlevel_ext: PubMedBERT (3 seeds, selection on this split's validation; records the last epoch) and C2 (derangement seed 42) on the 9 A_ft rows."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json
W = f"{_MEDVQA_ROOT}"; PJ = f"{W}/v7_txt/article_runs"; WT = f"{W}/v7_txt/worktree/harness/v7_txt"; CWD = f"{W}/v7_txt/worktree_m3ae"; QD = f"{PJ}/data/vqarad_qlevel"
M = []
for s in range(3):
    for mode, tag, cond in (("qlevel", "M3AE_SCRATCH_VQARAD_QLEVEL", "C0"), ("qlevelgray", "M3AE_TXT_VQARAD_QLEVEL", "TXT")):
        M.append({"name": f"train_{tag}_seed{s}", "env": "lapa", "reserve_mib": 27862, "output": f"{PJ}/logs/{tag}_seed{s}_cost.json", "log": f"{PJ}/logs/m3ae_qlevel/{tag}_seed{s}.sched.log", "cwd": CWD, "cmd": [f"{WT}/m3ae_txt_train.py", "vqa_rad", str(s), mode], "skip_order": True})
        M.append({"name": f"predict_{tag}_seed{s}", "env": "lapa", "reserve_mib": 8192, "output": f"{PJ}/results/{tag}_seed{s}_{cond}.json", "log": f"{PJ}/logs/m3ae_qlevel/predict_{tag}_seed{s}.log", "cwd": CWD,
                  "cmd": [f"{WT}/m3ae_txt_predict.py", "vqa_rad", str(s), "VQARAD", f"{PJ}/results/{tag}_seed{s}_{cond}.json", mode], "after": f"train_{tag}_seed{s}", "skip_order": True})
json.dump(M, open(f"{PJ}/logs/jobs_m3ae_qlevel.json", "w"), indent=1)
E = []; data = f"{PJ}/data/texto/VQARAD_QLEVEL.json"; dsha = hashlib.sha256(open(data, "rb").read()).hexdigest()
for s in range(3):
    n = f"BERT2_question_only_VQARAD_QLEVEL_seed{s}"; out = f"{PJ}/results/{n}.json"
    E.append({"name": n, "env": "circle", "reserve_mib": 4608, "env_extra": {"HF_HUB_OFFLINE": "1"}, "output": out, "skip_order": True, "log": f"{PJ}/logs/qlevel_ext/{n}.log",
              "cmd": ["-m", "harness.v7_txt.bert_question_only", "--base", "VQARAD_QLEVEL", "--seed", str(s), "--data", data, "--output", out, "--save-dir", f"{PJ}/models/bert2_vqarad_qlevel_seed{s}", "--progress-checkpoint", f"{PJ}/results/.progress/{n}.jsonl"],
              "meta": {"line": n, "condition": "C0 (so pergunta)", "base_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext@e1354b7a", "input_arrow": None, "dados": data, "dados_sha256": dsha, "dtype": "fp32"}})
C2 = f"{QD}/c2/vqa_vqa_rad_test_c2.arrow"
for fam, peft, cfg, line in (("QWEN2VL", "LORA_VT", "qwen2vl_lora_vt_vqarad_qlevel", "QWEN2VL_LORA_VT_VQARAD_QLEVEL"), ("SMOLVLM", "LORA", "smolvlm2_lora_vqarad_qlevel", "SMOLVLM_LORA_VQARAD_QLEVEL")):
    for s in range(3):
        n = f"{line}_seed{s}_C2"; out = f"{PJ}/results/{n}.json"; ad = f"{PJ}/models/{cfg}_seed{s}"
        E.append({"name": n, "env": "circle", "reserve_mib": 6144, "env_extra": {"HF_HUB_OFFLINE": "1"}, "output": out, "skip_order": True, "log": f"{PJ}/logs/qlevel_ext/{n}.log", "requires": [f"{PJ}/results/{line}_seed{s}_C0.json"],
                  "cmd": ["-m", "harness.extract_predictions_circle", "--model", fam, "--peft", peft, "--dataset", "VQARAD", "--condition", "C2", "--condition-arrow", C2, "--adapter-dir", ad, "--batch-size", "4", "--progress-checkpoint", f"{PJ}/results/.progress/{n}.jsonl", "--output", out],
                  "meta": {"line": f"{line}_seed{s}", "condition": "C2", "base_model": {"QWEN2VL": "Qwen/Qwen2-VL-2B-Instruct", "SMOLVLM": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}[fam], "adapter_dir": ad, "input_arrow": C2, "dtype": "bf16 (med_vlm/models/loader.py)"}})
for s in range(3):
    n = f"M3AE_SCRATCH_VQARAD_QLEVEL_seed{s}_C2"; out = f"{PJ}/results/{n}.json"
    E.append({"name": n, "env": "lapa", "reserve_mib": 8192, "output": out, "log": f"{PJ}/logs/qlevel_ext/{n}.log", "cwd": CWD, "skip_order": True, "requires": [f"{PJ}/results/M3AE_SCRATCH_VQARAD_QLEVEL_seed{s}_C0.json"],
              "cmd": [f"{WT}/m3ae_txt_predict.py", "vqa_rad", str(s), "VQARAD", out, "qlevelc2"]})
json.dump(E, open(f"{PJ}/logs/jobs_qlevel_ext.json", "w"), indent=1); print(len(M), len(E))
