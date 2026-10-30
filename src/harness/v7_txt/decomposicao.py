"""P5 (2026-10-10): accuracy decomposition table in three arms, per benchmark x family (and per question class):
own baseline = Acc(TXT); frozen-tower gain = Acc(C0 frozen) - Acc(TXT); adaptation gain = Acc(C0 adapted) -
Acc(C0 frozen); sum = Acc(C0 adapted) - Acc(TXT); alongside TFIDF and C1 and the differences TXT - TFIDF and TXT - C1. CI by bootstrap
by image (2000, seed 20260913, same replicates for all arms), mean and SD across seeds, McNemar mid-p between adjacent
arms (TXT vs C0 frozen; C0 frozen vs C0 adapted), Holm per benchmark. M3AE enters with TWO arms (TXT = constant
grey image; full fine-tune published), without a frozen-tower cell: single gain = full FT - TXT (different
design, marked). No interpretation."""
from __future__ import annotations
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, collections, json, os, time
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, V7, cells, classes, tfidf_preds, tfidf_fechadas, preds, sha, TXT_BENCH_M3AE, ARTICLE_RUNS, extra_baselines
BLABEL = {"VQARAD": "VQARAD -- versao Hugging Face (VLM: treino 1 793, sem validacao); linhas antigas, so para a nota de setup", "VQARAD_RADORIG": "VQARAD_RADORIG -- pacote original, divisao do M3AE (2 758 treino / 306 validacao / 451 teste); revision A1 do artigo 2 (15/10)", "VQARAD_QLEVEL": "VQARAD_QLEVEL -- dividido por pergunta (seed 20260913; 2 460 / 352 / 703 perguntas); controle da sobreposicao, vqarad_question_split (19/10)", "VQARAD_REDIV": "VQARAD_REDIV -- redividido por imagem (seed 20260913; 2 467 / 343 / 705 perguntas, 220 / 31 / 63 imagens); article_runs, vqarad_image_split (17/10)"}
from harness.stats.bootstrap import ImageBootstrap, holm, mcnemar_midp
from harness.interventions.arrow_io import read_arrow
VQK = ("TFIDF_publicado_272", "TFIDF_novo_272", "TFIDF_novo_451")   # 15/10 (the candidate): TFIDF of VQA-RAD in two subsets, in both versions
def vqarad_floors():
    """TFIDF published from p0 (272 closed-form), new TFIDF (training of 2 758) on the 272 closed-form and on the 451; qid -> correct dicts."""
    test = {int(x["qid"]): x["answer_type"].upper() for x in json.load(open(f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_data/vqa_rad/testset.json"))}
    pub = preds(f"{_MEDVQA_ROOT}/doutorado/medvqa-main/results/TFIDF_question_only_VQARAD_C0.json"); nov = preds(f"{V7}/results/TFIDF_question_only_VQARAD_orig.json")
    return {"TFIDF_publicado_272": pub, "TFIDF_novo_272": {q: v for q, v in nov.items() if test[q] == "CLOSED"}, "TFIDF_novo_451": nov}
def diff(a, b, qids, img, bs):
    va = np.array([a[q] for q in qids], float); vb = np.array([b[q] for q in qids], float); d = va - vb
    return {"diff": float(d.mean()), "ci": bs.ci_mean_diff([img[q] for q in qids], d), "midp": mcnemar_midp(va.astype(bool), vb.astype(bool))["p_midp"], "n": len(qids)}
def boot_vec(a, b, qids, img, bs):
    """17/10 (article_runs, Fig. 1 v2): vector of the 2 000 replicates of the mean per item of a - b (b = None: of a), resampling images with the SAME replicates as the base."""
    d = np.array([a[q] for q in qids], float) - (np.array([b[q] for q in qids], float) if b is not None else 0.0); idx = np.array([bs.index[img[q]] for q in qids]); n = len(bs.ids)
    num = np.bincount(idx, weights=d, minlength=n); den = np.bincount(idx, minlength=n).astype(float); return (bs.W @ num) / np.maximum(bs.W @ den, 1e-9)
def acc(a, qids, img, bs):
    v = np.array([a[q] for q in qids], float); return {"acc": float(v.mean()), "ci": bs.ci_mean_diff([img[q] for q in qids], v), "n": len(qids)}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kvasir-simnao", action="store_true", help="15/10 (the candidate): robustness -- only Kvasir, only questions with yes/no answers (2 294); implies --fechadas"); ap.add_argument("--fechadas", action="store_true", help="15/10 (the candidate): only closed-form questions (answer_type of the arrow); TFIDF = classifier on the closed-form questions"); ap.add_argument("--out-json", default=None); ap.add_argument("--out-md", default=None); a = ap.parse_args()
    KSN = a.kvasir_simnao; FECH = a.fechadas or KSN; SUF = "_kvasir_simnao" if KSN else ("_fechadas" if FECH else ""); a.out_json = a.out_json or f"{V7}/analysis/decomposicao{SUF}.json"; a.out_md = a.out_md or f"{V7}/docs/decomposicao{SUF}.md"
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": 2000, "seed": 20260913, "definitions": {"own_text_only_baseline": "Acc(TXT)", "ganho_torre": "Acc(C0 congelada) - Acc(TXT)", "ganho_adaptacao": "Acc(C0 adaptada) - Acc(C0 congelada)", "soma": "Acc(C0 adaptada) - Acc(TXT)", "m3ae": "dois bracos: TXT (imagem cinza) e fine-tune completo publicado; ganho unico = FT - TXT (sem celula de torre congelada)", "braco_so_texto": "LoRA em todas as celulas; linhas DoRA (congelada e adaptada) usam o TXT LoRA da mesma familia e semente; M3AE: ganho da imagem nao decomposto (14/10/2026)", "c1": "imagem cinza na linha congelada: HEAL em v7_txt/results, MIMIC em v7_mimic/results (14/10/2026); Kvasir/SLAKE/VQA-RAD como antes"}, "benchmarks": {}}
    from harness.v7_txt.txt_floors import RESULTS_DIR, mimic_floors
    for bench in (("KVASIR",) if KSN else ("SLAKE", "HEAL", "KVASIR", "MIMIC", "VQARAD", "VQARAD_RADORIG") + (("VQARAD_REDIV", "VQARAD_QLEVEL") if ARTICLE_RUNS else ())):   # 17/10: VQARAD_REDIV no modo article_runs   # 14/10: VQA-RAD (quinta base do Baseline); 15/10: VQA-RAD fonte unica (A1)
        df = read_arrow(ARROWS[bench]); cls, img = classes(bench, df); tfidf, m0src = tfidf_fechadas(bench) if FECH else tfidf_preds(bench); _g = {int(q): str(list(x)[0]).strip().lower() for _, r in df.iterrows() for q, x in zip(r["question_id"], r["answers"])} if KSN else {}
        allq = sorted(q for q in img if (not FECH or cls[q].get("answer_type") == "closed") and (not KSN or _g.get(q) in ("yes", "no"))); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
        B = {"arrow": ARROWS[bench], "tfidf_source": m0src, "lines": [], "pvals": {}}; extra = mimic_floors() if bench == "MIMIC" else (vqarad_floors() if bench in ("VQARAD", "VQARAD_RADORIG") else {})
        XP = extra_baselines(bench) if ARTICLE_RUNS else None   # 17/10 (article_runs, question_only_sets): majority and PubMedBERT (seed of the row)
        if XP: B["external_baseline_sources"] = {"maj": XP["maj_src"], "bert": XP["bert_src"]}
        if FECH: extra = {"TFIDF_full_fechadas": extra["TFIDF_full_fechadas"]} if bench == "MIMIC" else ({"TFIDF_novo_272": extra["TFIDF_novo_272"]} if bench in ("VQARAD", "VQARAD_RADORIG") else {})   # 15/10: only closed-form columns
        B["recorte"] = "Kvasir so sim/nao (gabarito yes/no; robustez)" if KSN else ("so fechadas (answer_type do arrow)" if FECH else "todas as perguntas"); B["n_qids_recorte"] = len(allq)
        for fam in (("QWEN2VL", "SMOLVLM", "M3AE") if bench == "MIMIC" else ("QWEN2VL", "SMOLVLM", "M3AE")):   # 13/10: SmolVLM2 no MIMIC; M3AE pendente
            for s in range(3):
                line = f"{fam}_LORA_TXT_{bench}_seed{s}" if fam != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(bench, bench)}_seed{s}"; tp = f"{RESULTS_DIR.get(bench, V7 + '/results')}/{line}_TXT.json"; c = cells(bench, fam, s)
                if not os.path.exists(tp): B["lines"].append({"line": line, "family": fam, "seed": s, "status": "TXT ausente"}); continue
                if not os.path.exists(c["frozen"]) or (c["adapted"] and not os.path.exists(c["adapted"])): B["lines"].append({"line": line, "family": fam, "seed": s, "status": "C0 congelada/adaptada ausente"}); continue
                T = preds(tp); F = preds(c["frozen"]); A = preds(c["adapted"]) if c["adapted"] else None; C1 = preds(c["c1"]) if c["c1"] and os.path.exists(c["c1"]) else None
                MJ = XP["maj"] if XP else None; BT = XP["bert"][s] if XP else None
                qids = [q for q in allq if q in T and q in F and (A is None or q in A)]
                rec = {"line": line, "family": fam, "seed": s, "design": "tres bracos" if A is not None else "M3AE: dois bracos (TXT cinza, FT completo)", "files": {"txt": tp, "frozen": c["frozen"], "adapted": c["adapted"], "c1": c["c1"]}, "sha256": {k: sha(v) for k, v in (("txt", tp), ("frozen", c["frozen"]), ("adapted", c["adapted"]), ("c1", c["c1"] if C1 else None))}, "n": len(qids), "status": "ok", "by_class": {}}
                def block(qs):
                    b = {"own_text_only_baseline": acc(T, qs, img, bs), "c0_congelada": acc(F, qs, img, bs), "ganho_torre": diff(F, T, qs, img, bs)}
                    if A is not None: b["c0_adaptada"] = acc(A, qs, img, bs); b["ganho_adaptacao"] = diff(A, F, qs, img, bs); b["soma"] = diff(A, T, qs, img, bs)
                    else: b["soma"] = b["ganho_torre"]
                    if tfidf and [q for q in qs if q in tfidf]: qm = [q for q in qs if q in tfidf]; b["tfidf"] = acc(tfidf, qm, img, bs); b["txt_menos_tfidf"] = diff(T, tfidf, qm, img, bs); b["adaptada_menos_tfidf"] = diff(A if A is not None else F, tfidf, qm, img, bs)   # 15/10 (the candidate): adapted tower - TFIDF (M3AE: full fine-tune - TFIDF)   # 14/10: no qids in common with the TFIDF (VQA-RAD: TFIDF only on the closed-form) the column stays empty
                    if C1:   # 14/10 (exec2): C1 of HEAL/MIMIC covers only the qids of the C1 arrow (evaluable units of the Occlusion study); class with no qid in C1 -> no cell (before: empty bincount used to break)
                        q1 = [q for q in qs if q in C1]
                        if q1: b["c1"] = acc(C1, q1, img, bs); b["txt_menos_c1"] = diff(T, C1, q1, img, bs)
                    for kk, dd in (("maj", MJ), ("bert", BT)):   # 17/10 (article_runs, question_only_sets): A_maj and A_bert; A_text - baseline and A_ft - baseline, CI per row
                        if dd and [q for q in qs if q in dd]: qx = [q for q in qs if q in dd]; b[kk] = acc(dd, qx, img, bs); b["txt_menos_" + kk] = diff(T, dd, qx, img, bs); b["adaptada_menos_" + kk] = diff(A if A is not None else F, dd, qx, img, bs)
                    # 15/10 (the candidate): three baselines on the SAME set -- qids with TXT, TFIDF and C1 (in MIMIC, C1 only exists in the units of the Occlusion study)
                    q3 = [q for q in qs if (tfidf is None or q in tfidf) and (C1 is None or q in C1)]
                    if q3:
                        b["p3_baseline"] = acc(T, q3, img, bs)
                        if tfidf: b["p3_tfidf"] = acc(tfidf, q3, img, bs); b["p3_txt_menos_tfidf"] = diff(T, tfidf, q3, img, bs)
                        if C1: b["p3_c1"] = acc(C1, q3, img, bs); b["p3_txt_menos_c1"] = diff(T, C1, q3, img, bs)
                        for kk, dd in (("maj", MJ), ("bert", BT)):   # 17/10: on the same set (BERT and the majority cover the same closed-form questions as the TF-IDF)
                            if dd and all(q in dd for q in q3): b["p3_" + kk] = acc(dd, q3, img, bs); b["p3_txt_menos_" + kk] = diff(T, dd, q3, img, bs)
                    for k, dd in extra.items():
                        if k != "baseline_subset": qk = [q for q in qs if q in dd]; b[k] = acc(dd, qk, img, bs) if qk else None; b["txt_menos_" + k] = diff(T, dd, qk, img, bs) if qk else None
                    return b
                rec["overall"] = block(qids)
                if ARTICLE_RUNS: rec["_boot"] = {"ganho_torre": boot_vec(F, T, qids, img, bs), "soma": boot_vec(A if A is not None else F, T, qids, img, bs), **({"ganho_adaptacao": boot_vec(A, F, qids, img, bs)} if A is not None else {}), "own_text_only_baseline": boot_vec(T, None, qids, img, bs), **({"txt_menos_bert": boot_vec(T, BT, [q for q in qids if q in BT], img, bs)} if BT else {})}   # 17/10: replicates for the CI of the mean across seeds; 18/10: A_text - PubMedBERT
                for key in sorted({k for q in qids for k in cls[q] if k != "evaluable"}):
                    for val in sorted({str(cls[q].get(key)) for q in qids}):
                        sub = [q for q in qids if str(cls[q].get(key)) == val]
                        if len(sub) >= 20 or key == "template": rec["by_class"][f"{key}={val}"] = block(sub)
                B["pvals"][line + "|torre"] = rec["overall"]["ganho_torre"]["midp"]
                if A is not None: B["pvals"][line + "|adaptacao"] = rec["overall"]["ganho_adaptacao"]["midp"]
                B["lines"].append(rec); print(bench, line, "baseline %.4f torre %+.4f adapt %s" % (rec["overall"]["own_text_only_baseline"]["acc"], rec["overall"]["ganho_torre"]["diff"], ("%+.4f" % rec["overall"]["ganho_adaptacao"]["diff"]) if A is not None else "n/a"), flush=True)
        B["holm"] = holm(B["pvals"]) if B["pvals"] else {}
        B["by_family"] = {}
        for fam in ("QWEN2VL", "SMOLVLM", "M3AE"):
            rs = [r for r in B["lines"] if r["family"] == fam and r.get("status") == "ok"]
            if not rs: continue
            fb = {"n_seeds": len(rs), "design": rs[0]["design"]}
            for k in ("own_text_only_baseline", "c0_congelada", "c0_adaptada", "tfidf", "c1", "TFIDF_subset_fechadas", "TFIDF_full_fechadas", "prior_subset_abertas", "prior_full_abertas", *VQK, "p3_baseline", "p3_tfidf", "p3_c1", "baseline_full", "maj", "bert", "p3_maj", "p3_bert"):
                v = [r["overall"][k]["acc"] for r in rs if r["overall"].get(k)]
                if v: fb[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
            for k in ("ganho_torre", "ganho_adaptacao", "soma", "txt_menos_tfidf", "txt_menos_c1", "adaptada_menos_tfidf", "p3_txt_menos_tfidf", "p3_txt_menos_c1", *["txt_menos_" + k for k in VQK], "txt_menos_maj", "txt_menos_bert", "adaptada_menos_maj", "adaptada_menos_bert", "p3_txt_menos_maj", "p3_txt_menos_bert"):
                v = [r["overall"][k]["diff"] for r in rs if r["overall"].get(k)]
                if v: fb[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
            if ARTICLE_RUNS:   # 17/10 (article_runs, Fig. 1 v2): CI 95 % of the MEAN across seeds, percentiles of the mean per replicate (same replicates per image)
                for k in ("ganho_torre", "ganho_adaptacao", "soma", "own_text_only_baseline", "txt_menos_bert"):
                    vs = [r["_boot"][k] for r in rs if k in r.get("_boot", {})]
                    if len(vs) == len(rs): m = np.mean(vs, axis=0); fb[k + "_ic_media_sementes"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
            fb["by_class"] = {}
            for cname in sorted({k for r in rs for k in r["by_class"]}):
                bb = [r["by_class"][cname] for r in rs if cname in r["by_class"]]
                fb["by_class"][cname] = {"n": bb[0]["own_text_only_baseline"]["n"], **{k: float(np.mean([b[k]["acc"] for b in bb])) for k in ("own_text_only_baseline", "c0_congelada", "c0_adaptada", "tfidf", "c1", "TFIDF_subset_fechadas", "TFIDF_full_fechadas", "prior_subset_abertas", "prior_full_abertas", *VQK, "p3_baseline", "p3_tfidf", "p3_c1") if all(b.get(k) for b in bb)}, **{k: float(np.mean([b[k]["diff"] for b in bb])) for k in ("ganho_torre", "ganho_adaptacao", "soma", "txt_menos_tfidf", "txt_menos_c1", "adaptada_menos_tfidf", "p3_txt_menos_tfidf", "p3_txt_menos_c1", *["txt_menos_" + k for k in VQK]) if all(b.get(k) for b in bb)}}
            B["by_family"][fam] = fb
        for r in B["lines"]: r.pop("_boot", None)
        out["benchmarks"][bench] = B
    # 13/10 (the candidate): tower and published rows of MIMIC against the text-only baseline (TXT) and the composite TFIDF, mean of the three seeds, same
    # bootstrap by image; numbers read from v7_mimic/analysis/tower_mimic.json (generated by harness/v7_txt/tower_report.py), no recalculation.
    TW = f"{_MEDVQA_ROOT}/v7_mimic/analysis/tower_mimic.json"
    if os.path.exists(TW) and not FECH:
        tw = json.load(open(TW)); out["mimic_torre_txt"] = {"fonte": TW, "fonte_sha256": sha(TW), "celulas": [{"celula": r["celula"], "n": r["n"], "baseline_tfidf": r["baseline_acc"], "familias": {f: {k: v for k, v in X.items() if k not in ("publicada", "torre")} | {"publicada_acc": (X["publicada"] or {}).get("acc_media"), "torre_acc": (X["torre"] or {}).get("acc_media"), "publicada_linha": (X["publicada"] or {}).get("linha"), "torre_linha": (X["torre"] or {}).get("linha")} for f, X in r["familias"].items()}} for r in tw["acc_C0"] if r["celula"] in ("todas", "answer_type=closed", "answer_type=open")]}
    json.dump(out, open(a.out_json, "w"), indent=1)
    f3 = lambda b: f"{b['acc']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]"; d3 = lambda b: f"{b['diff']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"; ms = lambda x: (f"{x['mean']:.3f} ({x['sd']:.3f})" if x and x.get("sd") is not None else (f"{x['mean']:.3f}" if x else "-"))
    M = ["# Decomposicao da acuracia em tres bracos (P5)" + (" -- so perguntas fechadas (artigo 2, 15/10)" if FECH else " -- todas as perguntas (suplemento interno do artigo 2)"), "", f"Gerado {out['t']}. baseline proprio = Acc(TXT, braco so-texto treinado); ganho da torre congelada = Acc(C0 congelada) - Acc(TXT); ganho da adaptacao = Acc(C0 adaptada) - Acc(C0 congelada); soma = Acc(C0 adaptada) - Acc(TXT). TFIDF = so-pergunta por item (HEAL alinhado); no MIMIC o 'TFIDF' e o baseline composto: TFIDF subset (LR sim/nao) nas fechadas + prior por template (resposta majoritaria no treino) nas abertas, rotulado assim. C1 = imagem cinza na linha congelada (onde existe; HEAL em v7_txt/results e MIMIC em v7_mimic/results desde 14/10). **Braco so-texto:** e LoRA em todas as celulas (QWEN2VL_LORA_TXT / SMOLVLM_LORA_TXT da mesma familia e semente); para as linhas DoRA (congelada e adaptada), o baseline usado e o so-texto LoRA da mesma familia e semente (nao ha TXT DoRA); o M3AE tem dois bracos (so-texto com imagem constante e ajuste completo), com o ganho da imagem nao decomposto em torre e adaptacao. IC 95 % por bootstrap por imagem (2000, seed 20260913, mesmas replicas); McNemar mid-p entre bracos adjacentes; Holm por benchmark sobre todos os testes. **M3AE: desenho diferente** -- dois bracos (TXT = imagem cinza constante no treino/val/teste; fine-tune completo publicado), sem celula de torre congelada; o 'ganho da torre' do M3AE e FT completo - TXT e nao se soma a uma adaptacao. Sem interpretacao.", ""]
    for bench, B in out["benchmarks"].items():
        M += [f"## {BLABEL.get(bench, bench)}", "", "| linha (braco so-texto = LoRA; DoRA usa o TXT LoRA da familia/semente) | n | baseline proprio (TXT LoRA) | C0 congelada | ganho torre [IC] mid-p (Holm) | C0 adaptada | ganho adaptacao [IC] mid-p (Holm) | soma [IC] | TFIDF | TXT - TFIDF [IC] | C1 (imagem cinza, linha congelada) | TXT - C1 [IC] |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in B["lines"]:
            if r.get("status") != "ok": M.append(f"| {r['line']} | {r['status']} |"); continue
            o = r["overall"]; ad = ("%s | %s p=%.2g (%.2g)" % (f3(o["c0_adaptada"]), d3(o["ganho_adaptacao"]), o["ganho_adaptacao"]["midp"], B["holm"][r["line"] + "|adaptacao"])) if "c0_adaptada" in o else "n/a (M3AE) | n/a"
            M.append(f"| {r['line']} | {r['n']} | {f3(o['own_text_only_baseline'])} | {f3(o['c0_congelada'])} | {d3(o['ganho_torre'])} p={o['ganho_torre']['midp']:.2g} ({B['holm'][r['line'] + '|torre']:.2g}) | {ad} | {d3(o['soma'])} | {f3(o['tfidf']) if 'tfidf' in o else '-'} | {d3(o['txt_menos_tfidf']) if 'txt_menos_tfidf' in o else '-'} | {f3(o['c1']) if 'c1' in o else '-'} | {d3(o['txt_menos_c1']) if 'txt_menos_c1' in o else '-'} |")
        M += ["", "### Media (DP) entre sementes, por familia", "", "| familia | desenho (so-texto = LoRA em todas as celulas; M3AE = 2 bracos, ganho da imagem nao decomposto) | baseline proprio | C0 congelada | ganho torre | C0 adaptada | ganho adaptacao | soma | TFIDF | TXT - TFIDF | C1 | TXT - C1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam, fb in B["by_family"].items(): M.append(f"| {fam} | {fb['design']} | " + " | ".join(ms(fb.get(k)) for k in ("own_text_only_baseline", "c0_congelada", "ganho_torre", "c0_adaptada", "ganho_adaptacao", "soma", "tfidf", "txt_menos_tfidf", "c1", "txt_menos_c1")) + " |")
        M += ["", "### Por classe de pergunta (media entre sementes; baseline / C0f / C0a / TFIDF / C1 ; ganho torre / ganho adaptacao / soma)" + (" -- MIMIC: 'TFIDF' = baseline composto (TFIDF subset nas fechadas + prior por template nas abertas); extras TFIDF full (fechadas) / prior subset (abertas) / prior full (abertas)" if bench == "MIMIC" else ""), ""]
        for fam, fb in B["by_family"].items():
            for cname, b in fb["by_class"].items():
                g = lambda k: f"{b[k]:.3f}" if k in b else "-"; h = lambda k: f"{b[k]:+.3f}" if k in b else "-"
                M.append(f"- {fam} | {cname} | n={b['n']} | {g('own_text_only_baseline')} / {g('c0_congelada')} / {g('c0_adaptada')} / {g('tfidf')} / {g('c1')} ; {h('ganho_torre')} / {h('ganho_adaptacao')} / {h('soma')}" + (f" ; extras {g('TFIDF_full_fechadas')} / {g('prior_subset_abertas')} / {g('prior_full_abertas')}" if bench == "MIMIC" else ""))
        M.append("")
    for VB in [b for b in ("VQARAD", "VQARAD_RADORIG") if b in out["benchmarks"]]:   # 14/10 (the candidate): VQA-RAD separated into closed-form and open, and Acc TXT per seed (read from the JSON itself, no recalculation); 15/10: also VQARAD_RADORIG
        V = out["benchmarks"][VB]; ok = [r for r in V["lines"] if r.get("status") == "ok"]
        g3 = lambda x: "-" if not x else f"{x['acc']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"; d3b = lambda x: "-" if not x else f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
        M += [f"## {VB} por tipo de resposta (fechadas e abertas)", "", "Mesmas colunas da tabela do VQA-RAD acima, por tipo de resposta. " + ("TFIDF publicado do p0 só nas fechadas (n = 272); nas abertas a coluna fica vazia." if VB == "VQARAD" else "TFIDF = TFIDF_question_only_VQARAD_orig (treino de 2 758, todas as 451).") + " IC 95 % por bootstrap por imagem (2 000, seed 20260913). Sem interpretação.", "",
              "| linha | tipo | n | baseline próprio (TXT) [IC] | C0 congelada | ganho torre [IC] | C0 adaptada | ganho adaptação [IC] | soma [IC] | TFIDF | TXT − TFIDF [IC] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in ok:
            for t in ("answer_type=closed", "answer_type=open"):
                c = r["by_class"].get(t)
                if not c: continue
                M.append(f"| {r['line']} | {t.split('=')[1]} | {c['own_text_only_baseline']['n']} | {g3(c.get('own_text_only_baseline'))} | {g3(c.get('c0_congelada'))} | {d3b(c.get('ganho_torre'))} | {g3(c.get('c0_adaptada'))} | {d3b(c.get('ganho_adaptacao'))} | {d3b(c.get('soma'))} | {g3(c.get('tfidf'))} | {d3b(c.get('txt_menos_tfidf'))} |")
        M += ["", "Média (DP) entre sementes por família e tipo:", "", "| família | tipo | baseline próprio | C0 congelada | ganho torre | C0 adaptada | ganho adaptação | soma | TFIDF | TXT − TFIDF |", "|---|---|---|---|---|---|---|---|---|---|"]
        for fam in sorted({r["family"] for r in ok}):
            for t in ("answer_type=closed", "answer_type=open"):
                cs = [r["by_class"][t] for r in ok if r["family"] == fam and t in r["by_class"]]
                def ms2(k, f):
                    v = [c[k][f] for c in cs if c.get(k)]
                    return "-" if not v else (f"{np.mean(v):+.3f} ({np.std(v, ddof=1):.3f})" if f == "diff" else f"{np.mean(v):.3f} ({np.std(v, ddof=1):.3f})") if len(v) > 1 else (f"{v[0]:.3f}" if v else "-")
                M.append(f"| {fam} | {t.split('=')[1]} | {ms2('own_text_only_baseline', 'acc')} | {ms2('c0_congelada', 'acc')} | {ms2('ganho_torre', 'diff')} | {ms2('c0_adaptada', 'acc')} | {ms2('ganho_adaptacao', 'diff')} | {ms2('soma', 'diff')} | {ms2('tfidf', 'acc')} | {ms2('txt_menos_tfidf', 'diff')} |")
        M += ["", "Acc TXT por semente (teste inteiro, n = 451; fechadas; abertas):", "", "| linha TXT | semente | todas [IC] | fechadas [IC] | abertas [IC] |", "|---|---|---|---|---|"]
        for r in ok:
            M.append(f"| {r['line']} | {r['seed']} | {g3(r['overall'].get('own_text_only_baseline'))} | {g3(r['by_class'].get('answer_type=closed', {}).get('own_text_only_baseline'))} | {g3(r['by_class'].get('answer_type=open', {}).get('own_text_only_baseline'))} |")
        M.append("")
    VV = [b for b in ("VQARAD", "VQARAD_RADORIG") if b in out["benchmarks"]]   # 15/10 (the candidate): TFIDF in two subsets and before/after by type
    if VV:
        lab = {"VQARAD": "versao HF (suplemento)", "VQARAD_RADORIG": "pacote original (artigo)"}
        M += ["## VQA-RAD: TFIDF em dois recortes, antes e depois (media (DP) entre sementes)", "", "TFIDF publicado = p0 (treino so nas fechadas), nas 272 fechadas; TFIDF novo = TFIDF_question_only_VQARAD_orig (treino de 2 758, todas), nas 272 fechadas e nas 451. TXT - TFIDF nos qids comuns. Sem interpretacao.", "",
              "| familia | versao | baseline proprio (451) | TFIDF publicado (272) | TXT - TFIDF publicado (272) | TFIDF novo (272) | TXT - TFIDF novo (272) | TFIDF novo (451) | TXT - TFIDF novo (451) |", "|---|---|---|---|---|---|---|---|---|"]
        for fam in ("QWEN2VL", "SMOLVLM", "M3AE"):
            for b in VV:
                fb = out["benchmarks"][b]["by_family"].get(fam)
                if fb: M.append(f"| {fam} | {lab[b]} | {ms(fb.get('own_text_only_baseline'))} | " + " | ".join(f"{ms(fb.get(k))} | {ms(fb.get('txt_menos_' + k))}" for k in VQK) + " |")
        M += ["", "## VQA-RAD antes e depois por tipo de resposta (media entre sementes)", "", "TFIDF novo nas abertas = TFIDF novo (451) restrito as abertas; TFIDF publicado so nas fechadas. Sem interpretacao.", "",
              "| familia | tipo | versao | n | baseline proprio | C0 congelada | ganho torre | C0 adaptada | ganho adaptacao | soma | TFIDF publicado | TFIDF novo | C1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam in ("QWEN2VL", "SMOLVLM", "M3AE"):
            for t in ("answer_type=closed", "answer_type=open"):
                for b in VV:
                    c = (out["benchmarks"][b]["by_family"].get(fam) or {}).get("by_class", {}).get(t)
                    if not c: continue
                    g = lambda k: f"{c[k]:.3f}" if k in c else "-"; h = lambda k: f"{c[k]:+.3f}" if k in c else "-"
                    M.append(f"| {fam} | {t.split('=')[1]} | {lab[b]} | {c['n']} | {g('own_text_only_baseline')} | {g('c0_congelada')} | {h('ganho_torre')} | {g('c0_adaptada')} | {h('ganho_adaptacao')} | {h('soma')} | {g('TFIDF_publicado_272')} | {g('TFIDF_novo_451')} | {g('c1')} |")
        M.append("")
    if "mimic_torre_txt" in out:
        T = out["mimic_torre_txt"]; c3 = lambda v, c=None: "-" if v is None else (f"{v:+.3f}" + (f" [{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""))
        M += ["## MIMIC: linhas de torre e publicadas contra o TXT", "", f"Numeros de `{T['fonte']}` (sha256 {T['fonte_sha256'][:12]}), media das tres sementes, IC 95 % por bootstrap por imagem (2000, seed 20260913), diferencas pareadas por pergunta. TXT = braco so-texto da familia (QWEN2VL_LORA_TXT_MIMIC; nao ha braco DoRA TXT; nao ha braco TXT do SmolVLM2 no MIMIC). TFIDF = baseline composto (TFIDF subset nas fechadas + prior por template nas abertas). Papel das linhas: no Qwen2-VL a linha VT adapta a torre e a publicada a congela; no SmolVLM2 a linha FT congela a torre e a publicada a adapta. Sem interpretacao.", "",
              "| familia | celula | n | TFIDF | TXT | publicada | torre | publicada - TXT [IC] | torre - TXT [IC] | torre - TFIDF [IC] |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in T["celulas"]:
            for fam, X in r["familias"].items():
                g = lambda v: "-" if v is None else f"{v:.3f}"
                M.append(f"| {fam} | {r['celula']} | {r['n']} | {g(r['baseline_tfidf'])} | {g(X.get('txt_acc_media'))} | {g(X.get('publicada_acc'))} | {g(X.get('torre_acc'))} | {c3(X.get('publicada_menos_txt'), X.get('publicada_menos_txt_ci'))} | {c3(X.get('torre_menos_txt'), X.get('torre_menos_txt_ci'))} | {c3(X.get('encoder_minus_baseline'), X.get('encoder_minus_baseline_ci'))} |")
        M.append("")
    open(a.out_md, "w").write("\n".join(M) + "\n"); print("FIM", a.out_json)
if __name__ == "__main__": main()
