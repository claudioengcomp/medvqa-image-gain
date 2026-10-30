"""18/10/2026 (candidato, vqarad_overlap; so leitura): sobreposicao treino(+validacao)/teste do VQA-RAD nos arquivos que usamos -- divisao original
(pacote do M3AE = linhagem MEVF: trainset 2 758 + valset 306 = 3 064; testset 451) e redivisao por imagem (seed 20260913).
Normalizacao de texto = med_vlm.metrics.process_text.normalize_text (strip, minusculas, sem pontuacao, espacos colapsados), para pergunta e resposta.
Sem campo question_frame nos nossos JSON: 'versao em template no treino' = existe pergunta de treino+validacao da MESMA imagem com phrase_type 'frame'
e a MESMA resposta normalizada (regra R1); variante R2 = R1 + mesmo question_type. Saida: v7_txt/article_runs/analysis/vqarad_sobreposicao.json. Sem interpretacao."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, sys, time, collections
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from med_vlm.metrics.process_text import normalize_text as N
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
O = f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_data/vqa_rad"; R = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv"
SETS = {"original": {"train": f"{O}/trainset.json", "val": f"{O}/valset.json", "test": f"{O}/testset.json"}, "redividido": {"train": f"{R}/trainset.json", "val": f"{R}/valset.json", "test": f"{R}/testset.json"}}
closed = lambda x: str(x["answer_type"]).strip().upper() == "CLOSED"; yn = lambda x: closed(x) and N(str(x["answer"])) in ("yes", "no")
out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": os.path.abspath(__file__), "script_sha256": sha(__file__), "normalizacao": "normalize_text: strip, lower, remove pontuacao, colapsa espacos; variante texto cru = strip + lower",
       "regra_template": {"R1": "mesma imagem + phrase_type 'frame' + mesma resposta normalizada", "R2": "R1 + mesmo question_type"}, "divisoes": {}}
for nome, fs in SETS.items():
    D = {k: json.load(open(p)) for k, p in fs.items()}; TR = D["train"] + D["val"]; TE = D["test"]
    timg = {x["image_name"] for x in TE}; trimg = {x["image_name"] for x in TR}
    qall = {N(x["question"]) for x in TR}; qimg = {(x["image_name"], N(x["question"])) for x in TR}
    qraw = {x["question"].strip().lower() for x in TR}; qrawimg = {(x["image_name"], x["question"].strip().lower()) for x in TR}   # variante: texto cru (strip + lower)
    frame = collections.defaultdict(list)
    for x in TR:
        if x.get("phrase_type") == "frame": frame[x["image_name"]].append(x)
    anys = collections.defaultdict(set)
    for x in TR: anys[x["image_name"]].add(N(str(x["answer"])))
    r1 = lambda x: any(N(str(y["answer"])) == N(str(x["answer"])) for y in frame[x["image_name"]])
    r2 = lambda x: any(N(str(y["answer"])) == N(str(x["answer"])) and y.get("question_type") == x.get("question_type") for y in frame[x["image_name"]])
    sub = {"todas": TE, "fechadas": [x for x in TE if closed(x)], "sim_nao": [x for x in TE if yn(x)]}
    r = {"arquivos": {k: {"path": p, "sha256": sha(p), "linhas": len(D[k]), "imagens": len({x['image_name'] for x in D[k]}), "phrase_type": dict(collections.Counter(x.get("phrase_type") for x in D[k])), "fechadas": sum(closed(x) for x in D[k]), "sim_nao": sum(yn(x) for x in D[k])} for k, p in fs.items()},
         "n_treino_mais_validacao": len(TR), "n_teste": len(TE), "imagens_teste": len(timg), "imagens_teste_em_treino_ou_validacao": len(timg & trimg),
         "imagens_teste_em_treino": len(timg & {x["image_name"] for x in D["train"]}), "imagens_teste_em_validacao": len(timg & {x["image_name"] for x in D["val"]}),
         "imagens_validacao_em_treino": len({x["image_name"] for x in D["val"]} & {x["image_name"] for x in D["train"]}), "contagens": {}}
    for s, L in sub.items():
        r["contagens"][s] = {"n": len(L), "pergunta_identica_qualquer_imagem": sum(N(x["question"]) in qall for x in L), "pergunta_identica_mesma_imagem": sum((x["image_name"], N(x["question"])) in qimg for x in L),
                             "pergunta_identica_texto_cru_qualquer_imagem": sum(x["question"].strip().lower() in qraw for x in L), "pergunta_identica_texto_cru_mesma_imagem": sum((x["image_name"], x["question"].strip().lower()) in qrawimg for x in L),
                             "template_mesma_imagem_R1": sum(r1(x) for x in L), "template_mesma_imagem_R2": sum(r2(x) for x in L), "mesma_imagem_mesma_resposta": sum(N(str(x["answer"])) in anys[x["image_name"]] for x in L)}
    out["divisoes"][nome] = r; print(nome, {k: v for k, v in r.items() if k not in ("arquivos", "contagens")}); [print("  ", s, c) for s, c in r["contagens"].items()]
json.dump(out, open(f"{_MEDVQA_ROOT}/v7_txt/article_runs/analysis/vqarad_sobreposicao.json", "w"), indent=1); print("FIM")
