"""19/10/2026 (candidato, vqarad_overlap-B2; so leitura): JSON publico do OSF (VQA_RAD Dataset Public.json) contra os nossos JSON do pacote original.
Chave de ligacao: (image_name, pergunta normalizada, resposta normalizada) com normalize_text do harness; perguntas 'frame' nossas contra o campo
question_frame das linhas do OSF (mesma imagem, mesma resposta). R3 = o question_frame da linha de teste do OSF aparece como pergunta (normalizada)
de treino + validacao nossos na MESMA imagem. Saida: v7_txt/article_runs/analysis/vqarad_osf.json. Sem interpretacao."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import collections, hashlib, json, os, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from med_vlm.metrics.process_text import normalize_text as N
sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
OSF = f"{_MEDVQA_ROOT}/doutorado/data_drive_sync/raw/vqa_rad_osf/VQA_RAD Dataset Public.json"; O = f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_data/vqa_rad"
J = json.load(open(OSF)); US = {k: json.load(open(f"{O}/{k}set.json")) for k in ("train", "val", "test")}
nul = lambda s: s is None or str(s).strip() in ("", "NULL")
osf_tr = [x for x in J if not x["phrase_type"].startswith("test")]; osf_te = [x for x in J if x["phrase_type"].startswith("test")]
out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": os.path.abspath(__file__), "script_sha256": sh(__file__), "osf_json": OSF, "osf_sha256": sh(OSF),
       "estrutura": {"linhas": len(J), "campos": sorted(J[0].keys()), "phrase_type": dict(collections.Counter(x["phrase_type"] for x in J)), "treino": len(osf_tr), "teste": len(osf_te),
                     "particoes": sorted({"teste" if x["phrase_type"].startswith("test") else "treino" for x in J}), "linhas_com_question_frame": sum(not nul(x["question_frame"]) for x in J),
                     "treino_com_question_frame": sum(not nul(x["question_frame"]) for x in osf_tr), "teste_com_question_frame": sum(not nul(x["question_frame"]) for x in osf_te)}}
key = lambda img, q, a: (img, N(str(q)), N(str(a)))
OQ = collections.Counter(key(x["image_name"], x["question"], x["answer"]) for x in J); OF = collections.Counter(key(x["image_name"], x["question_frame"], x["answer"]) for x in J if not nul(x["question_frame"]))
OQtr = {key(x["image_name"], x["question"], x["answer"]) for x in osf_tr}; OQte = {key(x["image_name"], x["question"], x["answer"]) for x in osf_te}; OFtr = {key(x["image_name"], x["question_frame"], x["answer"]) for x in osf_tr if not nul(x["question_frame"])}
lig = {}
for part, L in US.items():
    nf = [x for x in L if x["phrase_type"] != "frame"]; fr = [x for x in L if x["phrase_type"] == "frame"]
    lig[part] = {"n": len(L), "nao_frame": len(nf), "nao_frame_casadas_question_osf": sum(key(x["image_name"], x["question"], x["answer"]) in OQ for x in nf),
                 "nao_frame_casadas_em_linhas_de_treino_osf": sum(key(x["image_name"], x["question"], x["answer"]) in OQtr for x in nf), "nao_frame_casadas_em_linhas_de_teste_osf": sum(key(x["image_name"], x["question"], x["answer"]) in OQte for x in nf),
                 "frame": len(fr), "frame_casadas_question_frame_osf": sum(key(x["image_name"], x["question"], x["answer"]) in OF for x in fr), "frame_casadas_question_frame_de_linha_de_treino_osf": sum(key(x["image_name"], x["question"], x["answer"]) in OFtr for x in fr)}
lig["val_vindas_de_treino_osf"] = sum((key(x["image_name"], x["question"], x["answer"]) in (OQtr if x["phrase_type"] != "frame" else OFtr)) for x in US["val"])
out["ligacao"] = lig
timg = {x["image_name"] for x in osf_te}; out["imagens"] = {"imagens_teste_osf": len(timg), "em_1797_linhas_treino_osf": len(timg & {x["image_name"] for x in osf_tr}), "em_treino_mais_val_nossos_3064": len(timg & {x["image_name"] for x in US["train"] + US["val"]})}
TRV = US["train"] + US["val"]; qimg = {(x["image_name"], N(x["question"])) for x in TRV}
ute = {key(x["image_name"], x["question"], x["answer"]): x for x in US["test"]}
cl = lambda x: str(x["answer_type"]).strip().upper() == "CLOSED"; yn = lambda x: cl(x) and N(str(x["answer"])) in ("yes", "no")
rows = []
for x in osf_te:
    u = ute.get(key(x["image_name"], x["question"], x["answer"])); rows.append({"tem_frame": not nul(x["question_frame"]), "r3": (not nul(x["question_frame"])) and (x["image_name"], N(x["question_frame"])) in qimg, "casou_nosso_teste": u is not None, "fechada": cl(x), "sim_nao": yn(x)})
out["R3"] = {s: {"n": len(R), "com_question_frame": sum(r["tem_frame"] for r in R), "R3": sum(r["r3"] for r in R)} for s, R in (("todas", rows), ("fechadas", [r for r in rows if r["fechada"]]), ("sim_nao", [r for r in rows if r["sim_nao"]]))}
out["R3_nota"] = {"linhas_de_teste_osf_casadas_ao_nosso_testset": sum(r["casou_nosso_teste"] for r in rows), "fechadas_pelo_answer_type_do_osf_com_strip": sum(r["fechada"] for r in rows)}
json.dump(out, open(f"{_MEDVQA_ROOT}/v7_txt/article_runs/analysis/vqarad_osf.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("estrutura", "ligacao", "imagens", "R3", "R3_nota")}, indent=1, default=list))
