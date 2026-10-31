"""17/10/2026 (article_runs, vqarad_image_split): M3AE queues (GPU 3, one at a time) and PubMedBERT queue on the re-split VQA-RAD.
jobs_m3ae_rediv.json: 3 M3AE trainings (published recipe, rediv mode) + 3 C0 predictions on the new test set; reserve = measured peak of the VQA-RAD M3AE text-only
(26 394 MiB -> 27 862, as in the m3aetxt queue). jobs_m3ae_txt_rediv.json: the M3AE text-only (redivgray mode) -- PREPARED, NOT LAUNCHED without ok.
jobs_bert_rediv.json: PubMedBERT, 3 seeds, same recipe as pubmedbert_classifier."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; WT = f"{_MEDVQA_ROOT}/v7_txt/worktree/harness/v7_txt"; CWD = f"{_MEDVQA_ROOT}/v7_txt/worktree_m3ae"
def m3(mode, tag, cond):
    J = []
    for s in range(3):
        J.append({"name": f"train_{tag}_seed{s}", "env": "lapa", "reserve_mib": 27862, "output": f"{PJ}/logs/{tag}_seed{s}_cost.json", "log": f"{PJ}/logs/m3ae_rediv/{tag}_seed{s}.sched.log", "cwd": CWD,
                  "cmd": [f"{WT}/m3ae_txt_train.py", "vqa_rad", str(s), mode], "skip_order": True})
        J.append({"name": f"predict_{tag}_seed{s}", "env": "lapa", "reserve_mib": 8192, "output": f"{PJ}/results/{tag}_seed{s}_{cond}.json", "log": f"{PJ}/logs/m3ae_rediv/predict_{tag}_seed{s}.log", "cwd": CWD,
                  "cmd": [f"{WT}/m3ae_txt_predict.py", "vqa_rad", str(s), "VQARAD", f"{PJ}/results/{tag}_seed{s}_{cond}.json", mode], "after": f"train_{tag}_seed{s}", "skip_order": True})
    return J
json.dump(m3("rediv", "M3AE_SCRATCH_VQARAD_REDIV", "C0"), open(f"{PJ}/logs/jobs_m3ae_rediv.json", "w"), indent=1)
json.dump(m3("redivgray", "M3AE_TXT_VQARAD_REDIV", "TXT"), open(f"{PJ}/logs/jobs_m3ae_txt_rediv.json", "w"), indent=1)
data = f"{PJ}/data/texto/VQARAD_REDIV.json"; dsha = hashlib.sha256(open(data, "rb").read()).hexdigest(); B = []
for s in range(3):
    n = f"BERT_question_only_VQARAD_REDIV_seed{s}"; out = f"{PJ}/results/{n}.json"
    B.append({"name": n, "env": "circle", "reserve_mib": 6144, "env_extra": {"HF_HUB_OFFLINE": "1"}, "output": out, "skip_order": True, "log": f"{PJ}/logs/bert/{n}.log",
              "cmd": ["-m", "harness.v7_txt.bert_question_only", "--base", "VQARAD_REDIV", "--seed", str(s), "--data", data, "--output", out, "--save-dir", f"{PJ}/models/bert_vqarad_rediv_seed{s}", "--progress-checkpoint", f"{PJ}/results/.progress/{n}.jsonl"],
              "meta": {"line": n, "condition": "C0 (question only)", "base_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext@e1354b7a", "input_arrow": None, "dados": data, "dados_sha256": dsha, "dtype": "fp32"}})
json.dump(B, open(f"{PJ}/logs/jobs_bert_rediv.json", "w"), indent=1); print("ok")
