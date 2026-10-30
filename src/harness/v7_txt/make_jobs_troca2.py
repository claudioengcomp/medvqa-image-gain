"""19/10/2026 (image_swap_v2): inferencia com a troca refeita (C2v2) nas 9 linhas A_ft de HEAL e MIMIC. Mesmo comando dos C2 anteriores; muda so o arrow e a saida."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import json
W = f"{_MEDVQA_ROOT}"; PJ = f"{W}/v7_txt/article_runs"; J = []
for b, B in (("heal", "HEAL"), ("mimic", "MIMIC")):
    man = json.load(open(f"{PJ}/data/troca2/{b}/manifest.json")); arrow, sh = man["output"], man["sha256"]
    ad = {"HEAL": {"QWEN2VL": f"{W}/v6_tower_heal/models/qwen2vl_lora_vt_heal_seed%d", "SMOLVLM": f"{W}/doutorado/medvqa-occlusion-study/outputs/models/smolvlm2_lora_heal_seed%d"},
          "MIMIC": {"QWEN2VL": f"{W}/v7_mimic/models/qwen2vl_lora_vt_mimic_seed%d", "SMOLVLM": f"{W}/v7_mimic/models/smolvlm2_lora_mimic_seed%d"}}[B]
    for fam, peft, line in (("QWEN2VL", "LORA_VT", f"QWEN2VL_LORA_VT_{B}_seed%d"), ("SMOLVLM", "LORA", f"SMOLVLM_LORA_{B}_seed%d")):
        for s in range(3):
            n = line % s + "_C2v2"; out = f"{PJ}/results/{n}.json"
            J.append({"name": n, "env": "circle", "reserve_mib": 6144, "env_extra": {"HF_HUB_OFFLINE": "1"}, "output": out, "skip_order": True, "log": f"{PJ}/logs/troca2/{n}.log",
                      "cmd": ["-m", "harness.extract_predictions_circle", "--model", fam, "--peft", peft, "--dataset", B, "--condition", "C2", "--condition-arrow", arrow, "--adapter-dir", ad[fam] % s, "--batch-size", "4", "--progress-checkpoint", f"{PJ}/results/.progress/{n}.jsonl", "--output", out],
                      "meta": {"line": line % s, "condition": "C2 (troca refeita, image_swap_v2)", "base_model": {"QWEN2VL": "Qwen/Qwen2-VL-2B-Instruct", "SMOLVLM": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}[fam], "adapter_dir": ad[fam] % s, "input_arrow": arrow, "input_arrow_sha256_expected": sh, "dtype": "bf16 (med_vlm/models/loader.py)"}})
    for s in range(3):
        n = f"M3AE_{B}_seed{s}_C2v2"; out = f"{PJ}/results/{n}.json"
        J.append({"name": n, "env": "lapa", "reserve_mib": 8192, "output": out, "log": f"{PJ}/logs/troca2/{n}.log", "cwd": f"{W}/v7_txt/worktree_m3ae", "skip_order": True,
                  "cmd": [f"{W}/v7_txt/worktree/harness/v7_txt/m3ae_cond_article_runs.py", b, str(s), arrow, sh, "C2v2", out]})
json.dump(J, open(f"{PJ}/logs/jobs_troca2.json", "w"), indent=1); print(len(J))
