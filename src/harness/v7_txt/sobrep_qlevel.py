"""19/10/2026 (vqarad_question_split.1): eight overlap numbers, split by question x by image. R3 = question_frame (OSF JSON) of the test question present as
training + validation (normalized) question on the same image; identical question = normalized text (normalize_text) on the same image. Output: analysis/vqarad_qlevel_sobreposicao.json."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import json, sys, hashlib
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from med_vlm.metrics.process_text import normalize_text as N
P = f"{_MEDVQA_ROOT}/v7_txt/article_runs"; OSF = f"{_MEDVQA_ROOT}/doutorado/data_drive_sync/raw/vqa_rad_osf/VQA_RAD Dataset Public.json"
O = json.load(open(OSF)); nul = lambda s: s is None or str(s).strip() in ("", "NULL")
fr = {(x["image_name"], N(x["question"])): x["question_frame"] for x in O if not nul(x["question_frame"])}
cl = lambda x: str(x["answer_type"]).strip().upper() == "CLOSED"; yn = lambda x: cl(x) and N(str(x["answer"])) in ("yes", "no"); sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"osf_sha256": sh(OSF), "divisoes": {}}
for nome, d in (("por_pergunta", f"{P}/data/vqarad_qlevel"), ("por_imagem", f"{P}/data/vqarad_rediv")):
    S = {k: json.load(open(f"{d}/{k}set.json")) for k in ("train", "val", "test")}; TV = S["train"] + S["val"]; te = S["test"]
    q = {(x["image_name"], N(x["question"])) for x in TV}; timg = {x["image_name"] for x in te}
    r3 = lambda x: (x["image_name"], N(x["question"])) in fr and (x["image_name"], N(fr[(x["image_name"], N(x["question"]))])) in q
    out["divisoes"][nome] = {"perguntas": {k: len(v) for k, v in S.items()}, "imagens": {k: len({x["image_name"] for x in v}) for k, v in S.items()}, "teste_fechadas": sum(cl(x) for x in te), "teste_sim_nao": sum(yn(x) for x in te),
        "imagens_teste_em_treino_mais_val": len(timg & {x["image_name"] for x in TV}), "R3_template_mesma_imagem": {"todas": sum(r3(x) for x in te), "fechadas": sum(r3(x) for x in te if cl(x)), "sim_nao": sum(r3(x) for x in te if yn(x))},
        "pergunta_identica_mesma_imagem": {"todas": sum((x["image_name"], N(x["question"])) in q for x in te), "fechadas": sum((x["image_name"], N(x["question"])) in q for x in te if cl(x))},
        "arquivos": {k: sh(f"{d}/{k}set.json") for k in ("train", "val", "test")}}
json.dump(out, open(f"{P}/analysis/vqarad_qlevel_sobreposicao.json", "w"), indent=1); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "arquivos"} for k, v in out["divisoes"].items()}, indent=0))
