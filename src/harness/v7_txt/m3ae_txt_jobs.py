"""P1.3 (2026-10-09): jobs do M3AE so-texto para o infer_scheduler_v7 (env lapa): 9 treinos (slack, kvasir, heal x seeds 0-2;
placa exclusiva, reserva 27 GB; wrapper m3ae_txt_train.py -> run_m3ae_txt.sh; 'output' = <NAME>_cost.json) e 9 predicoes no
teste cinza (m3ae_txt_predict.py localiza o melhor checkpoint salvo pelo Lightning e chama m3ae_infer_v7 com data_root cinza)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import json, os
R = f"{_MEDVQA_ROOT}/v7_txt"; H = R + "/worktree/harness/v7_txt"
jobs = []
for bench, DS in (("slack", "SLAKE"), ("heal", "HEAL"), ("kvasir", "KVASIR")):
    for s in range(3):
        name = "M3AE_TXT_%s_seed%d" % (bench, s); line = "M3AE_TXT_%s_seed%d" % (DS, s)
        jobs.append({"name": "train_" + name, "env": "lapa", "reserve_mib": 27 * 1024, "exclusive": True, "output": "%s/logs/%s_cost.json" % (R, name), "log": "%s/logs/%s.sched.log" % (R, name), "cwd": R + "/worktree_m3ae",
                     "cmd": [H + "/m3ae_txt_train.py", bench, str(s)], "skip_order": True})
        jobs.append({"name": "predict_" + name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s_TXT.json" % (R, line), "log": "%s/logs/predict_%s.log" % (R, name), "cwd": R + "/worktree_m3ae",
                     "cmd": [H + "/m3ae_txt_predict.py", bench, str(s), DS, "%s/results/%s_TXT.json" % (R, line)], "after": "train_" + name, "skip_order": True})
# 13/10 (candidato, Baseline): (a) M3AE so-texto no VQA-RAD x3; (b) retreino do M3AE no SLAKE e no VQA-RAD, seeds 1 e 2, receita publicada, sufixo _gpu1,
# predicao C0. Reserva 27 862 MiB (pico medido do M3AE + 1,5 GB); sem exclusividade de placa (decisao de 12/10).
RES = 27862
for s in range(3):
    name = "M3AE_TXT_vqa_rad_seed%d" % s; line = "M3AE_TXT_VQARAD_seed%d" % s
    jobs.append({"name": "train_" + name, "env": "lapa", "reserve_mib": RES, "output": "%s/logs/%s_cost.json" % (R, name), "log": "%s/logs/%s.sched.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_train.py", "vqa_rad", str(s)], "skip_order": True})
    jobs.append({"name": "predict_" + name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s_TXT.json" % (R, line), "log": "%s/logs/predict_%s.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_predict.py", "vqa_rad", str(s), "VQARAD", "%s/results/%s_TXT.json" % (R, line)], "after": "train_" + name, "skip_order": True})
# 14/10 (candidato): retreinos _gpu1 CANCELADOS -- os pesos do Colab das seeds 1 e 2 (SLAKE, VQA-RAD) existem no Drive e foram copiados para
# doutorado/M3AE/checkpoints_colab/; nenhum job _gpu1 e gerado. O bloco fica abaixo, desligado, para registro.
for bench, DS in (() if True else (("slack", "SLAKE"), ("vqa_rad", "VQARAD"))):
    for s in (1, 2):
        name = "M3AE_SCRATCH_%s_seed%d_gpu1" % (DS, s)
        jobs.append({"name": "train_" + name, "env": "lapa", "reserve_mib": RES, "output": "%s/logs/%s_cost.json" % (R, name), "log": "%s/logs/%s.sched.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_train.py", bench, str(s), "img"], "skip_order": True})
        jobs.append({"name": "predict_" + name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s_C0.json" % (R, name), "log": "%s/logs/predict_%s.log" % (R, name), "cwd": R + "/worktree_m3ae", "cmd": [H + "/m3ae_txt_predict.py", bench, str(s), DS, "%s/results/%s_C0.json" % (R, name), "img"], "after": "train_" + name, "skip_order": True})
# 14/10 (candidato): reproducao do C0 do VQA-RAD, seeds 1 e 2, com os pesos do Colab (commit 6cc9eac; copia do Drive conferida por md5),
# no teste original (data_root proprio com links para os arrows originais, so leitura). Comparacao item a item com
# doutorado/medvqa-main/results/M3AE_SCRATCH_VQARAD_seed<s>_C0.json (banda entre ambientes 2,7-5,9 %).
VR = R + "/m3ae_roots/vqarad_C0/"; ORIG_SHA = "e1e464f0e99dc21ce00c37117e36288c"   # md5 do vqa_vqa_rad_test.arrow == .orig_backup (14/10)
# 14/10 (candidato): prioridade 1 -- os dois jobs passaram para make_jobs_c0repro.py (alvo c0repro); bloco desligado aqui, para registro.
for s in (() if True else (1, 2)):
    name = "M3AE_SCRATCH_VQARAD_seed%d_C0_colabrepro" % s; ck = f"{_MEDVQA_ROOT}/doutorado/M3AE/checkpoints_colab/M3AE_SCRATCH_VQARAD_seed%d.ckpt" % s
    jobs.append({"name": name, "env": "lapa", "reserve_mib": 8 * 1024, "output": "%s/results/%s.json" % (R, name), "log": "%s/logs/%s.log" % (R, name), "cwd": R + "/worktree_m3ae", "skip_order": True,
                 "cmd": [H + "/m3ae_infer_v7.py", "--name", name, "--checkpoint", ck, "--data-root", VR, "--dataset-key", "vqa_vqa_rad", "--vqa-label-size", "498", "--dataset", "VQARAD", "--condition", "C0",
                         "--output", "%s/results/%s.json" % (R, name), "--model-label", "M3AE_SCRATCH", "--extra", "reproducao C0 pesos Colab 6cc9eac; md5 arrow teste %s" % ORIG_SHA]})
json.dump(jobs, open(R + "/logs/jobs_m3ae_txt.json", "w"), indent=1); print(len(jobs), "jobs")
