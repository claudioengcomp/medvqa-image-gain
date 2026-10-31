"""19/10/2026 (heal_classifiers_train_only): PubMedBERT v2 -- same recipe, also records the accuracy of the last epoch; HEAL_TREINO = train-only with selection on validation.
Outputs BERT2_question_only_<BASE>_seed<s>.json (nothing overwritten)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; B = []
for base in ("KVASIR", "SLAKE", "MIMIC", "VQARAD_REDIV", "HEAL_TREINO"):
    data = f"{PJ}/data/texto/{base}.json"; dsha = hashlib.sha256(open(data, "rb").read()).hexdigest()
    for s in range(3):
        n = f"BERT2_question_only_{base}_seed{s}"; out = f"{PJ}/results/{n}.json"
        B.append({"name": n, "env": "circle", "reserve_mib": 4608, "env_extra": {"HF_HUB_OFFLINE": "1"}, "output": out, "skip_order": True, "log": f"{PJ}/logs/bert2/{n}.log",
                  "cmd": ["-m", "harness.v7_txt.bert_question_only", "--base", base, "--seed", str(s), "--data", data, "--output", out, "--save-dir", f"{PJ}/models/bert2_{base.lower()}_seed{s}", "--progress-checkpoint", f"{PJ}/results/.progress/{n}.jsonl"],
                  "meta": {"line": n, "condition": "C0 (question only)", "base_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext@e1354b7a", "input_arrow": None, "dados": data, "dados_sha256": dsha, "dtype": "fp32"}})
json.dump(B, open(f"{PJ}/logs/jobs_bert2.json", "w"), indent=1); print(len(B))
