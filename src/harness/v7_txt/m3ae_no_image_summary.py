"""23/10/2026 (m3ae_no_image_atext): syntheses used in the paper text (abstract, contributions, results, conclusion), computed BEFORE (current analyses, A_text of the M3AE
with grey image) and AFTER (mirror of m3ae_no_image_atext, A_text of the M3AE without image, m3ae_no_image) by the same rules. Read-only of the analysis JSONs; nothing typed in.
Output: article_runs/m3ae_no_image_atext/sinteses.json"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import glob, hashlib, json, os
import numpy as np
W = f"{_MEDVQA_ROOT}"; PJ = W + "/v7_txt/article_runs"; A = PJ + "/analysis"; ESP = PJ + "/m3ae_no_image_atext/espelho_semimg"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ART = ["KVASIR", "SLAKE", "HEAL", "MIMIC", "VQARAD_REDIV"]; G4 = ["KVASIR", "SLAKE", "HEAL", "MIMIC"]; FAM = ["QWEN2VL", "SMOLVLM", "M3AE"]
NOMES = ("decomposicao_fechadas", "joint_ci", "m3ae_frozen_vit", "analise_troca_v2", "ic_contagem", "text_analyses", "heal_classifiers_train_only")
def carrega(raiz):
    P = {n: f"{raiz}/{n}.json" for n in NOMES}; return {n: json.load(open(p)) for n, p in P.items()}, {p: sha(p) for p in P.values()}
def sint(J):
    D = J["decomposicao_fechadas"]["benchmarks"]; I15 = J["joint_ci"]["bases"]; I17 = J["m3ae_frozen_vit"]["bases"]; S2 = J["analise_troca_v2"]["bases"]; IC = J["ic_contagem"]; I13 = J["text_analyses"]; I10 = J["heal_classifiers_train_only"]["heal_so_treino"]["por_familia"]
    bf = lambda b, f: D[b]["by_family"][f]
    o = {}
    # 2.1 Table 2, M3AE
    o["tab2_m3ae"] = {b: {"A_text": [bf(b, "M3AE")["own_text_only_baseline"]["mean"], bf(b, "M3AE")["own_text_only_baseline"]["sd"]], "G_enc": [I17[b]["diferencas"]["G_enc"]["media"]] + I17[b]["diferencas"]["G_enc"]["ic_conjunto"],
                         "G_ft": [I17[b]["diferencas"]["G_ft"]["media"]] + I17[b]["diferencas"]["G_ft"]["ic_conjunto"], "G": [bf(b, "M3AE")["soma"]["mean"]] + I15[b]["M3AE"]["G"]["ic_conjunto"]} for b in ART}
    # 2.2 counts (45 runs = 30 VLM + 15 frozen M3AE; G: 45 = 5 x 9 fine-tuned rows, M3AE = full fine-tune)
    E = {b: I17[b]["ic_por_execucao"] for b in ART}; inc = lambda x: x[1] <= 0 <= x[2]
    C = IC["contagem"]; ex_G = [[x["base"], x["linha"]] for x in IC["linhas"] if not x["G_exclui_zero"]]
    gft_vlm = [x for x in I13["13_1"]["execucoes"]]; gft_m3 = [(b, s, x) for b in ART for s, x in E[b]["G_ft"].items()]
    exc_vlm = [(x["base"], x["G_ft"]) for x in gft_vlm if x["exclui_zero"]]; exc_m3 = [(b, x[0]) for b, s, x in gft_m3 if not inc(x)]
    o["contagens_45"] = {"G_enc_IC_inclui_zero": C["Genc_inclui_zero"] + sum(inc(x) for b in ART for x in E[b]["G_enc"].values()),
                         "G_enc_IC_inclui_zero_onde": [[b, "M3AE_FROZEN seed" + s] for b in ART for s, x in E[b]["G_enc"].items() if inc(x)] + [[x["base"], x["linha"]] for x in IC["linhas"] if x["familia"] != "M3AE" and x["Genc_inclui_zero"]],
                         "G_IC_exclui_zero": C["G_exclui_zero"], "G_n": C["n_afinadas"], "G_excecoes": ex_G,
                         "G_ft_IC_inclui_zero": C["Gft_inclui_zero"] + sum(inc(x) for b, s, x in gft_m3), "G_ft_exclui_max_4grandes": max(g for b, g in exc_vlm + exc_m3 if b in G4), "G_ft_exclui_max_VQARAD": max(g for b, g in exc_vlm + exc_m3 if b == "VQARAD_REDIV")}
    # 2.3 decomposition
    cel = {}
    for b in ART:
        for f in FAM:
            ge, gf = (bf(b, f)["ganho_torre"]["mean"], bf(b, f)["ganho_adaptacao"]["mean"]) if f != "M3AE" else (I17[b]["diferencas"]["G_enc"]["media"], I17[b]["diferencas"]["G_ft"]["media"])
            cel[f"{b}|{f}"] = [ge, gf]
    o["celulas_Genc_maior_Gft"] = {"n": sum(ge > gf for ge, gf in cel.values()), "de": len(cel), "excecoes": {k: v for k, v in cel.items() if not v[0] > v[1]}}
    o["faixa_G_por_base"] = {b: [min(bf(b, f)["soma"]["mean"] for f in FAM), max(bf(b, f)["soma"]["mean"] for f in FAM)] for b in ART}
    o["M3AE_Genc_4grandes"] = [min(I17[b]["diferencas"]["G_enc"]["media"] for b in G4), max(I17[b]["diferencas"]["G_enc"]["media"] for b in G4)]
    o["M3AE_Gft_4grandes"] = [min(I17[b]["diferencas"]["G_ft"]["media"] for b in G4), max(I17[b]["diferencas"]["G_ft"]["media"] for b in G4)]
    o["VLM_Gft"] = [min(bf(b, f)["ganho_adaptacao"]["mean"] for b in ART for f in FAM[:2]), max(bf(b, f)["ganho_adaptacao"]["mean"] for b in ART for f in FAM[:2])]
    # 2.4 baselines against A_text (HEAL: train-only protocol of heal_classifiers_train_only, as in Table 2)
    tmb = lambda b, f: I10[f]["txt_menos_bert"]["mean"] if b == "HEAL" else bf(b, f)["txt_menos_bert"]["mean"]
    tmt = lambda b, f: I10[f]["txt_menos_tfidf"]["mean"] if b == "HEAL" else bf(b, f)["txt_menos_tfidf"]["mean"]
    o["PubMedBERT_menos_A_text"] = {b: {f: -tmb(b, f) for f in FAM} for b in ART}
    o["max_abs_PubMedBERT_menos_A_text_por_base"] = {b: max(abs(tmb(b, f)) for f in FAM) for b in ART}
    o["bases_dentro_de_1pp"] = [b for b in ART if max(abs(tmb(b, f)) for f in FAM) < 0.01]
    o["A_text_menos_TFIDF_max"] = max((tmt(b, f), b, f) for b in ART for f in FAM)
    o["A_text_menos_majoritaria_max"] = max((bf(b, f)["txt_menos_maj"]["mean"], b, f) for b in ART for f in FAM)
    o["maj_por_pergunta_menos_A_text"] = {b: {f: I13["13_3"][b]["acc"] - bf(b, f)["own_text_only_baseline"]["mean"] for f in FAM} for b in ART}
    o["maj_por_pergunta"] = {b: I13["13_3"][b]["acc"] for b in ART}
    o["A_text_por_familia"] = {b: {f: [bf(b, f)["own_text_only_baseline"]["mean"], bf(b, f)["own_text_only_baseline"]["sd"]] for f in FAM} for b in ART}
    o["A_text_amplitude_entre_familias"] = {b: max(bf(b, f)["own_text_only_baseline"]["mean"] for f in FAM) - min(bf(b, f)["own_text_only_baseline"]["mean"] for f in FAM) for b in ART}
    o["A_text_media_das_familias"] = {b: float(np.mean([bf(b, f)["own_text_only_baseline"]["mean"] for f in FAM])) for b in ART}
    o["A_text_dp_max_por_base"] = {b: max(bf(b, f)["own_text_only_baseline"]["sd"] for f in FAM) for b in ART}
    # 2.5 HEAL: ganho creditado
    o["HEAL_G"] = {f: bf("HEAL", f)["soma"]["mean"] for f in FAM}; o["HEAL_Aft_menos_TFIDF"] = {f: I10[f]["ft_menos_tfidf"]["mean"] for f in FAM}
    # 2.6 swap
    sw = {(b, f): S2[b]["por_familia"][f]["A_swap_menos_A_text"]["mean"] for b in G4 for f in FAM}
    o["A_swap_menos_A_text_4grandes"] = {"min": min(sw.values()), "max": max(sw.values()), "todas_negativas": all(v < 0 for v in sw.values()), "por_celula": {f"{b}|{f}": v for (b, f), v in sw.items()}}
    dr = {(b, f): S2[b]["por_familia"][f]["A_menos_A_swap"]["mean"] for b in ART for f in FAM}
    o["A_menos_A_swap"] = {"VQARAD_REDIV": [min(v for (b, f), v in dr.items() if b == "VQARAD_REDIV"), max(v for (b, f), v in dr.items() if b == "VQARAD_REDIV")], "4grandes": [min(v for (b, f), v in dr.items() if b in G4), max(v for (b, f), v in dr.items() if b in G4)]}
    # 2.7 VQA-RAD, two splits
    v = {}
    for b in ("VQARAD_QLEVEL", "VQARAD_REDIV"):
        v[b] = {"A_text": {f: bf(b, f)["own_text_only_baseline"]["mean"] for f in FAM}, "PubMedBERT": bf(b, "QWEN2VL")["bert"]["mean"], "G": {f: [bf(b, f)["soma"]["mean"]] + I15[b][f]["G"]["ic_conjunto"] for f in FAM}}
    v["A_text_sobe_nas_3"] = {f: v["VQARAD_QLEVEL"]["A_text"][f] > v["VQARAD_REDIV"]["A_text"][f] for f in FAM}
    v["G_sobe"] = {f: v["VQARAD_QLEVEL"]["G"][f][0] > v["VQARAD_REDIV"]["G"][f][0] for f in FAM}
    o["vqarad_duas_divisoes"] = v
    # 2.8 grey probe of the frozen (HEAL)
    o["cinza_congelado_HEAL"] = [I17["HEAL"]["por_semente"][s]["A_cinza"] for s in ("0", "1", "2")]
    return o
out = {}
for tag, raiz in (("antes", A), ("depois", ESP + "/v7_txt/article_runs/analysis")):
    J, F = carrega(raiz); out[tag] = sint(J); out[tag + "_fontes"] = F
# 2.9 cost and parameters
C13 = json.load(open(A + "/text_analyses.json"))["13_6"]; ST = {j["name"]: j for j in json.load(open(PJ + "/m3ae_semimg/logs/infer_state_semimg21b.json"))["jobs"]}
def media(xs): return float(np.mean(xs)) if xs else None
BMAP = {"VQARAD_REDIV": "VQARAD_REDIV"}
cus = {}
for b in ART:
    sem = [ST[f"train_M3AE_SEMIMG_{b}_seed{s}"] for s in range(3)]
    linha = {mo: media([c["horas"] for c in C13 if c["base"] == b and c["modelo"] == mo]) for mo in ("PubMedBERT", "Qwen2-VL", "SmolVLM2", "M3AE so-texto")}
    linha["M3AE sem imagem"] = media([j["duration_s"] / 3600 for j in sem]); linha["M3AE sem imagem: pico por processo (GB) min-max"] = [min(j["peak_mib"] for j in sem) / 1024, max(j["peak_mib"] for j in sem) / 1024]
    cus[b] = linha
def faixa(mods): xs = [cus[b][m] for b in ART for m in mods if cus[b][m] is not None]; return [min(xs), max(xs)]
out["custo"] = {"por_base_h": cus, "faixa_so_texto_antes_h (VLM + M3AE cinza)": faixa(("Qwen2-VL", "SmolVLM2", "M3AE so-texto")), "faixa_so_texto_depois_h (VLM + M3AE sem imagem)": faixa(("Qwen2-VL", "SmolVLM2", "M3AE sem imagem")),
                "faixa_so_texto_VLM_h": faixa(("Qwen2-VL", "SmolVLM2")), "faixa_PubMedBERT_h": faixa(("PubMedBERT",)),
                "PubMedBERT_mais_barato_em_cada_base_antes": {b: cus[b]["PubMedBERT"] < min(cus[b][m] for m in ("Qwen2-VL", "SmolVLM2", "M3AE so-texto")) for b in ART},
                "PubMedBERT_mais_barato_em_cada_base_depois": {b: cus[b]["PubMedBERT"] < min(cus[b][m] for m in ("Qwen2-VL", "SmolVLM2", "M3AE sem imagem")) for b in ART},
                "M3AE_sem_imagem_pico_GB_min_max_5bases": [min(cus[b]["M3AE sem imagem: pico por processo (GB) min-max"][0] for b in ART), max(cus[b]["M3AE sem imagem: pico por processo (GB) min-max"][1] for b in ART)],
                "medida": "training job duration and process-tree peak, both from the scheduler (same measurement as the 'Cost:' rows in section 9)", "fontes": {A + "/text_analyses.json": sha(A + "/text_analyses.json"), PJ + "/m3ae_semimg/logs/infer_state_semimg21b.json": sha(PJ + "/m3ae_semimg/logs/infer_state_semimg21b.json")}}
P = json.load(open(PJ + "/m3ae_no_image_atext/parametros_semimg.json")) if os.path.exists(PJ + "/m3ae_no_image_atext/parametros_semimg.json") else None
out["parametros_sem_imagem"] = P
json.dump(out, open(PJ + "/m3ae_no_image_atext/sinteses.json", "w"), indent=1, default=float); print("FIM")
