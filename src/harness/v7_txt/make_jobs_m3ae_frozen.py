"""20/10/2026 (m3ae_frozen_vit): fila do M3AE com ViT congelado para o infer_scheduler_v7 (env lapa). Ordem SLAKE -> VQA-RAD (por imagem) -> HEAL ->
Medical-Diff-VQA -> Kvasir, semente 0, 1, 2; depois VQA-RAD por pergunta (vqarad_question_split). Cada treino: 3 predicoes (C0, C1 cinza, C2 troca), depois do treino.
RESERVA: 27 862 MiB nos treinos (pico medido do M3AE completo + 1,5 GB; regra: so desce depois de medir o pico do congelado); 8 GiB nas predicoes."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_frozen_cfg as C
H = C.W + "/v7_txt/worktree/harness/v7_txt"; L = C.OUT + "/logs"; R = C.OUT + "/results"; os.makedirs(R, exist_ok=True); os.makedirs(L + "/sched", exist_ok=True)
RES_TRAIN = int(os.environ.get("M3AE_FROZEN_RES_TRAIN", 17638))   # 20/10 19:4x: pico medido do congelado (SLAKE seed 0, epoca 0 com validacao, placa so nossa) 16 102 MiB + 1,5 GB; antes 27 862
RES_PRED = 8192
jobs = []
for b in C.ORDEM:
    for s in range(3):
        n = C.name(b, s)
        jobs.append({"name": "train_" + n, "env": "lapa", "reserve_mib": RES_TRAIN, "output": "%s/%s_cost.json" % (L, n), "log": "%s/sched/%s.train.log" % (L, n), "cwd": C.W + "/v7_txt/worktree_m3ae",
                     "cmd": [H + "/m3ae_frozen_train.py", b, str(s)], "skip_order": True})
        for cond in C.BENCH[b]["test"]:
            o = "%s/%s_%s.json" % (R, n, cond)
            jobs.append({"name": "predict_%s_%s" % (n, cond), "env": "lapa", "reserve_mib": RES_PRED, "output": o, "log": "%s/sched/%s_%s.predict.log" % (L, n, cond), "cwd": C.W + "/v7_txt/worktree_m3ae",
                         "cmd": [H + "/m3ae_frozen_predict.py", b, str(s), cond, o], "after": "train_" + n, "skip_order": True})
# 20/10 21:xx (candidato): treinos antes de todas as predicoes; o escalonador percorre a lista em ordem e so lanca predicao quando nenhum treino cabe
jobs = [j for j in jobs if j["name"].startswith("train_")] + [j for j in jobs if not j["name"].startswith("train_")]
json.dump(jobs, open(L + "/jobs_m3ae_frozen.json", "w"), indent=1); print(len(jobs), "jobs; reserva treino", RES_TRAIN)
