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
BLABEL = {"VQARAD": "VQARAD -- Hugging Face version (VLM: train 1 793, no validation); old rows, only for the setup note", "VQARAD_RADORIG": "VQARAD_RADORIG -- original package, M3AE split (2 758 train / 306 validation / 451 test); revision A1 of paper 2 (15/10)", "VQARAD_QLEVEL": "VQARAD_QLEVEL -- split by question (seed 20260913; 2 460 / 352 / 703 questions); overlap control, vqarad_question_split (19/10)", "VQARAD_REDIV": "VQARAD_REDIV -- re-split by image (seed 20260913; 2 467 / 343 / 705 questions, 220 / 31 / 63 images); article_runs, vqarad_image_split (17/10)"}
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
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": 2000, "seed": 20260913, "definitions": {"own_text_only_baseline": "Acc(TXT)", "ganho_torre": "Acc(C0 congelada) - Acc(TXT)", "ganho_adaptacao": "Acc(C0 adaptada) - Acc(C0 congelada)", "soma": "Acc(C0 adaptada) - Acc(TXT)", "m3ae": "two branches: TXT (grey image) and published full fine-tune; single gain = FT - TXT (no frozen-encoder cell)", "braco_so_texto": "LoRA in all cells; DoRA rows (frozen and fine-tuned) use the TXT LoRA of the same family and seed; M3AE: image gain not decomposed (14/10/2026)", "c1": "grey image in the frozen row: HEAL in v7_txt/results, MIMIC in v7_mimic/results (14/10/2026); Kvasir/SLAKE/VQA-RAD as before"}, "benchmarks": {}}
    from harness.v7_txt.txt_floors import RESULTS_DIR, mimic_floors
    for bench in (("KVASIR",) if KSN else ("SLAKE", "HEAL", "KVASIR", "MIMIC", "VQARAD", "VQARAD_RADORIG") + (("VQARAD_REDIV", "VQARAD_QLEVEL") if ARTICLE_RUNS else ())):   # 17/10: VQARAD_REDIV no modo article_runs   # 14/10: VQA-RAD (quinta base do Baseline); 15/10: VQA-RAD fonte unica (A1)
        df = read_arrow(ARROWS[bench]); cls, img = classes(bench, df); tfidf, m0src = tfidf_fechadas(bench) if FECH else tfidf_preds(bench); _g = {int(q): str(list(x)[0]).strip().lower() for _, r in df.iterrows() for q, x in zip(r["question_id"], r["answers"])} if KSN else {}
        allq = sorted(q for q in img if (not FECH or cls[q].get("answer_type") == "closed") and (not KSN or _g.get(q) in ("yes", "no"))); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
        B = {"arrow": ARROWS[bench], "tfidf_source": m0src, "lines": [], "pvals": {}}; extra = mimic_floors() if bench == "MIMIC" else (vqarad_floors() if bench in ("VQARAD", "VQARAD_RADORIG") else {})
        XP = extra_baselines(bench) if ARTICLE_RUNS else None   # 17/10 (article_runs, question_only_sets): majority and PubMedBERT (seed of the row)
        if XP: B["external_baseline_sources"] = {"maj": XP["maj_src"], "bert": XP["bert_src"]}
        if FECH: extra = {"TFIDF_full_fechadas": extra["TFIDF_full_fechadas"]} if bench == "MIMIC" else ({"TFIDF_novo_272": extra["TFIDF_novo_272"]} if bench in ("VQARAD", "VQARAD_RADORIG") else {})   # 15/10: only closed-form columns
        B["recorte"] = "Kvasir yes/no only (gold answer yes/no; robustness)" if KSN else ("closed-form only (answer_type from the arrow)" if FECH else "all questions"); B["n_qids_recorte"] = len(allq)
        for fam in (("QWEN2VL", "SMOLVLM", "M3AE") if bench == "MIMIC" else ("QWEN2VL", "SMOLVLM", "M3AE")):   # 13/10: SmolVLM2 no MIMIC; M3AE pendente
            for s in range(3):
                line = f"{fam}_LORA_TXT_{bench}_seed{s}" if fam != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(bench, bench)}_seed{s}"; tp = f"{RESULTS_DIR.get(bench, V7 + '/results')}/{line}_TXT.json"; c = cells(bench, fam, s)
                if not os.path.exists(tp): B["lines"].append({"line": line, "family": fam, "seed": s, "status": "TXT ausente"}); continue
                if not os.path.exists(c["frozen"]) or (c["adapted"] and not os.path.exists(c["adapted"])): B["lines"].append({"line": line, "family": fam, "seed": s, "status": "C0 congelada/adaptada ausente"}); continue
                T = preds(tp); F = preds(c["frozen"]); A = preds(c["adapted"]) if c["adapted"] else None; C1 = preds(c["c1"]) if c["c1"] and os.path.exists(c["c1"]) else None
                MJ = XP["maj"] if XP else None; BT = XP["bert"][s] if XP else None
                qids = [q for q in allq if q in T and q in F and (A is None or q in A)]
                rec = {"line": line, "family": fam, "seed": s, "design": "tres bracos" if A is not None else "M3AE: two branches (TXT grey, full FT)", "files": {"txt": tp, "frozen": c["frozen"], "adapted": c["adapted"], "c1": c["c1"]}, "sha256": {k: sha(v) for k, v in (("txt", tp), ("frozen", c["frozen"]), ("adapted", c["adapted"]), ("c1", c["c1"] if C1 else None))}, "n": len(qids), "status": "ok", "by_class": {}}
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
                B["lines"].append(rec); print(bench, line, "baseline %.4f encoder %+.4f adapt %s" % (rec["overall"]["own_text_only_baseline"]["acc"], rec["overall"]["ganho_torre"]["diff"], ("%+.4f" % rec["overall"]["ganho_adaptacao"]["diff"]) if A is not None else "n/a"), flush=True)
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
    M = ["# Decomposicao da acuracia em tres bracos (P5)" + (" -- closed-form questions only (paper 2, 15/10)" if FECH else " -- all questions (internal supplement of paper 2)"), "", f"Gerado {out['t']}. own baseline = Acc(TXT, trained text-decoder-only approach); frozen-encoder gain = Acc(C0 frozen) - Acc(TXT); fine-tuning gain = Acc(C0 fine-tuned) - Acc(C0 frozen); total = Acc(C0 fine-tuned) - Acc(TXT). TFIDF = question-only per item (HEAL aligned); on MIMIC the 'TFIDF' is the composite baseline: TFIDF subset (yes/no LR) on the closed-form questions + per-template prior (majority answer in training) on the open ones, labelled as such. C1 = grey image on the frozen row (where it exists; HEAL in v7_txt/results and MIMIC in v7_mimic/results since 14/10). **Text-decoder-only approach:** LoRA in every cell (QWEN2VL_LORA_TXT / SMOLVLM_LORA_TXT of the same family and seed); for the DoRA rows (frozen and fine-tuned) the baseline used is the LoRA text-decoder-only run of the same family and seed (there is no TXT DoRA); M3AE has two approaches (text-only with a constant image and full fine-tuning), with the image gain not decomposed into frozen-encoder and fine-tuning parts. 95 % CI by bootstrap over images (2000, seed 20260913, same replicates); McNemar mid-p between adjacent approaches; Holm per benchmark over all tests. **M3AE: different design** -- two approaches (TXT = constant grey image in train/val/test; published full fine-tuning), without a frozen-encoder cell; the M3AE 'frozen-encoder gain' is full FT - TXT and does not add up with a fine-tuning gain. No interpretation.", ""]
    for bench, B in out["benchmarks"].items():
        M += [f"## {BLABEL.get(bench, bench)}", "", "| row (text-decoder-only branch = LoRA; DoRA uses the TXT LoRA of the family/seed) | n | own baseline (TXT LoRA) | C0 frozen | frozen-encoder gain [CI] mid-p (Holm) | C0 fine-tuned | fine-tuning gain [CI] mid-p (Holm) | total gain [CI] | TFIDF | TXT - TFIDF [CI] | C1 (grey image, frozen row) | TXT - C1 [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in B["lines"]:
            if r.get("status") != "ok": M.append(f"| {r['line']} | {r['status']} |"); continue
            o = r["overall"]; ad = ("%s | %s p=%.2g (%.2g)" % (f3(o["c0_adaptada"]), d3(o["ganho_adaptacao"]), o["ganho_adaptacao"]["midp"], B["holm"][r["line"] + "|adaptacao"])) if "c0_adaptada" in o else "n/a (M3AE) | n/a"
            M.append(f"| {r['line']} | {r['n']} | {f3(o['own_text_only_baseline'])} | {f3(o['c0_congelada'])} | {d3(o['ganho_torre'])} p={o['ganho_torre']['midp']:.2g} ({B['holm'][r['line'] + '|torre']:.2g}) | {ad} | {d3(o['soma'])} | {f3(o['tfidf']) if 'tfidf' in o else '-'} | {d3(o['txt_menos_tfidf']) if 'txt_menos_tfidf' in o else '-'} | {f3(o['c1']) if 'c1' in o else '-'} | {d3(o['txt_menos_c1']) if 'txt_menos_c1' in o else '-'} |")
        M += ["", "### Mean (SD) across seeds, by family", "", "| family | design (text-decoder-only = LoRA in all cells; M3AE = 2 branches, image gain not decomposed) | own baseline | C0 frozen | frozen-encoder gain | C0 fine-tuned | fine-tuning gain | total gain | TFIDF | TXT - TFIDF | C1 | TXT - C1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam, fb in B["by_family"].items(): M.append(f"| {fam} | {fb['design']} | " + " | ".join(ms(fb.get(k)) for k in ("own_text_only_baseline", "c0_congelada", "ganho_torre", "c0_adaptada", "ganho_adaptacao", "soma", "tfidf", "txt_menos_tfidf", "c1", "txt_menos_c1")) + " |")
        M += ["", "### By question class (mean across seeds; baseline / C0f / C0a / TFIDF / C1 ; frozen-encoder gain / fine-tuning gain / total gain)" + (" -- MIMIC: 'TFIDF' = composite baseline (TFIDF subset on closed-form + prior by template on open); extras TFIDF full (closed-form) / prior subset (open) / prior full (open)" if bench == "MIMIC" else ""), ""]
        for fam, fb in B["by_family"].items():
            for cname, b in fb["by_class"].items():
                g = lambda k: f"{b[k]:.3f}" if k in b else "-"; h = lambda k: f"{b[k]:+.3f}" if k in b else "-"
                M.append(f"- {fam} | {cname} | n={b['n']} | {g('own_text_only_baseline')} / {g('c0_congelada')} / {g('c0_adaptada')} / {g('tfidf')} / {g('c1')} ; {h('ganho_torre')} / {h('ganho_adaptacao')} / {h('soma')}" + (f" ; extras {g('TFIDF_full_fechadas')} / {g('prior_subset_abertas')} / {g('prior_full_abertas')}" if bench == "MIMIC" else ""))
        M.append("")
    for VB in [b for b in ("VQARAD", "VQARAD_RADORIG") if b in out["benchmarks"]]:   # 14/10 (the candidate): VQA-RAD separated into closed-form and open, and Acc TXT per seed (read from the JSON itself, no recalculation); 15/10: also VQARAD_RADORIG
        V = out["benchmarks"][VB]; ok = [r for r in V["lines"] if r.get("status") == "ok"]
        g3 = lambda x: "-" if not x else f"{x['acc']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"; d3b = lambda x: "-" if not x else f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
        M += [f"## {VB} by answer type (closed-form and open)", "", "Same columns as the VQA-RAD table above, by answer type. " + ("TFIDF published from p0 only on closed-form (n = 272); on open questions the column stays empty." if VB == "VQARAD" else "TFIDF = TFIDF_question_only_VQARAD_orig (train of 2 758, all 451).") + " CI 95 % by bootstrap per image (2 000, seed 20260913). No interpretation.", "",
              "| row | type | n | own baseline (TXT) [CI] | C0 frozen | frozen-encoder gain [CI] | C0 fine-tuned | fine-tuning gain [CI] | total gain [CI] | TFIDF | TXT − TFIDF [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in ok:
            for t in ("answer_type=closed", "answer_type=open"):
                c = r["by_class"].get(t)
                if not c: continue
                M.append(f"| {r['line']} | {t.split('=')[1]} | {c['own_text_only_baseline']['n']} | {g3(c.get('own_text_only_baseline'))} | {g3(c.get('c0_congelada'))} | {d3b(c.get('ganho_torre'))} | {g3(c.get('c0_adaptada'))} | {d3b(c.get('ganho_adaptacao'))} | {d3b(c.get('soma'))} | {g3(c.get('tfidf'))} | {d3b(c.get('txt_menos_tfidf'))} |")
        M += ["", "Mean (SD) across seeds by family and type:", "", "| family | type | own baseline | C0 frozen | frozen-encoder gain | C0 fine-tuned | fine-tuning gain | total gain | TFIDF | TXT − TFIDF |", "|---|---|---|---|---|---|---|---|---|---|"]
        for fam in sorted({r["family"] for r in ok}):
            for t in ("answer_type=closed", "answer_type=open"):
                cs = [r["by_class"][t] for r in ok if r["family"] == fam and t in r["by_class"]]
                def ms2(k, f):
                    v = [c[k][f] for c in cs if c.get(k)]
                    return "-" if not v else (f"{np.mean(v):+.3f} ({np.std(v, ddof=1):.3f})" if f == "diff" else f"{np.mean(v):.3f} ({np.std(v, ddof=1):.3f})") if len(v) > 1 else (f"{v[0]:.3f}" if v else "-")
                M.append(f"| {fam} | {t.split('=')[1]} | {ms2('own_text_only_baseline', 'acc')} | {ms2('c0_congelada', 'acc')} | {ms2('ganho_torre', 'diff')} | {ms2('c0_adaptada', 'acc')} | {ms2('ganho_adaptacao', 'diff')} | {ms2('soma', 'diff')} | {ms2('tfidf', 'acc')} | {ms2('txt_menos_tfidf', 'diff')} |")
        M += ["", "Acc TXT by seed (full test, n = 451; closed-form; open):", "", "| TXT row | seed | all [CI] | closed-form [CI] | open [CI] |", "|---|---|---|---|---|"]
        for r in ok:
            M.append(f"| {r['line']} | {r['seed']} | {g3(r['overall'].get('own_text_only_baseline'))} | {g3(r['by_class'].get('answer_type=closed', {}).get('own_text_only_baseline'))} | {g3(r['by_class'].get('answer_type=open', {}).get('own_text_only_baseline'))} |")
        M.append("")
    VV = [b for b in ("VQARAD", "VQARAD_RADORIG") if b in out["benchmarks"]]   # 15/10 (the candidate): TFIDF in two subsets and before/after by type
    if VV:
        lab = {"VQARAD": "versao HF (suplemento)", "VQARAD_RADORIG": "pacote original (artigo)"}
        M += ["## VQA-RAD: TFIDF em dois recortes, antes e depois (media (DP) entre sementes)", "", "TFIDF published = p0 (train only on closed-form), on the 272 closed-form; TFIDF new = TFIDF_question_only_VQARAD_orig (train of 2 758, all), on the 272 closed-form and on the 451. TXT - TFIDF on common qids. No interpretation.", "",
              "| family | version | own baseline (451) | TFIDF published (272) | TXT - TFIDF published (272) | TFIDF new (272) | TXT - TFIDF new (272) | TFIDF new (451) | TXT - TFIDF new (451) |", "|---|---|---|---|---|---|---|---|---|"]
        for fam in ("QWEN2VL", "SMOLVLM", "M3AE"):
            for b in VV:
                fb = out["benchmarks"][b]["by_family"].get(fam)
                if fb: M.append(f"| {fam} | {lab[b]} | {ms(fb.get('own_text_only_baseline'))} | " + " | ".join(f"{ms(fb.get(k))} | {ms(fb.get('txt_menos_' + k))}" for k in VQK) + " |")
        M += ["", "## VQA-RAD before and after by answer type (mean across seeds)", "", "TFIDF new on open = TFIDF new (451) restricted to open; TFIDF published only on closed-form. No interpretation.", "",
              "| family | type | version | n | own baseline | C0 frozen | frozen-encoder gain | C0 fine-tuned | fine-tuning gain | total gain | TFIDF published | TFIDF new | C1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
        M += ["## MIMIC: linhas de torre e publicadas contra o TXT", "", f"Numeros de `{T['fonte']}` (sha256 {T['fonte_sha256'][:12]}), mean of the three seeds, 95 % CI by bootstrap over images (2000, seed 20260913), paired differences per question. TXT = text-decoder-only approach of the family (QWEN2VL_LORA_TXT_MIMIC; there is no DoRA TXT approach; there is no SmolVLM2 TXT approach on MIMIC). TFIDF = composite baseline (TFIDF subset on the closed-form questions + per-template prior on the open ones). Role of the rows: for Qwen2-VL the VT row fine-tunes the visual encoder and the published row keeps it frozen; for SmolVLM2 the FT row keeps the visual encoder frozen and the published row fine-tunes it. No interpretation.", "",
              "| family | cell | n | TFIDF | TXT | published | encoder | published - TXT [CI] | encoder - TXT [CI] | encoder - TFIDF [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in T["celulas"]:
            for fam, X in r["familias"].items():
                g = lambda v: "-" if v is None else f"{v:.3f}"
                M.append(f"| {fam} | {r['celula']} | {r['n']} | {g(r['baseline_tfidf'])} | {g(X.get('txt_acc_media'))} | {g(X.get('publicada_acc'))} | {g(X.get('torre_acc'))} | {c3(X.get('publicada_menos_txt'), X.get('publicada_menos_txt_ci'))} | {c3(X.get('torre_menos_txt'), X.get('torre_menos_txt_ci'))} | {c3(X.get('encoder_minus_baseline'), X.get('encoder_minus_baseline_ci'))} |")
        M.append("")
    open(a.out_md, "w").write("\n".join(M) + "\n"); print("FIM", a.out_json)
if __name__ == "__main__": main()
