"""P1.4 (2026-10-09): consolidation of the text-only arm. Per row (benchmark, family, seed): global Acc(TXT) and per
question class, alongside Acc(C0) of the frozen, Acc(C0) of the fine-tuned, Acc(C1) where it exists and TFIDF per class, on the same
subset of qids (HEAL: 1 446 assessable). CIs by bootstrap PER IMAGE (2000, seed 20260913, the same replicas for
all arms); McNemar mid-p TXT vs frozen C0 and TXT vs TFIDF per row; Holm per benchmark (over the rows x 2
tests); mean and SD between seeds; count of win/loss/tie per item TXT vs frozen C0. No interpretation.
Runs with whatever exists in v7_txt/results (re-runnable)."""
from __future__ import annotations
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, collections, hashlib, json, os, time
import numpy as np
from harness.interventions.arrow_io import read_arrow
from harness.stats.bootstrap import ImageBootstrap, holm, mcnemar_midp

P0R = f"{_MEDVQA_ROOT}/doutorado/medvqa-main/results"; ISBI = f"{_MEDVQA_ROOT}/doutorado/medvqa-occlusion-study/results/isbi_occlusion"
V5Q = f"{_MEDVQA_ROOT}/v5_qwen_vt/results"; V5S = f"{_MEDVQA_ROOT}/v5_smolvlm_ft/results"; V6R = f"{_MEDVQA_ROOT}/v6_tower_heal/results"; V7 = f"{_MEDVQA_ROOT}/v7_txt"
MIMIC = f"{_MEDVQA_ROOT}/v7_mimic"; RESULTS_DIR = {"MIMIC": f"{MIMIC}/results"}
ARROWS = {"VQARAD": f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows/vqa_vqa_rad_test.arrow", "MIMIC": f"{MIMIC}/data/vqa_mimic_test.arrow", "KVASIR": f"{_MEDVQA_ROOT}/doutorado/medvqa-main/data/finetune_arrows/vqa_kvasir_test_clean.arrow", "SLAKE": f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows/vqa_slack_test.arrow", "HEAL": f"{_MEDVQA_ROOT}/doutorado/medvqa-occlusion-study/data/finetune_arrows/vqa_heal_test.arrow"}
ARROWS["VQARAD_RADORIG"] = ARROWS["VQARAD"]   # 15/10: VQA-RAD single source (revision A1); same test
TXT_BENCH_M3AE = {"VQARAD_RADORIG": "VQARAD"}   # 15/10: the M3AE text-only of VQA-RAD already used the original package
# 17/10 (article_runs): ARTICLE2_RUNS=1 -> (a) VQARAD_REDIV benchmark (VQA-RAD re-split by image, vqarad_image_split); (b) C1 of MIMIC = the 1 499 closed-form (baselines);
# (c) M3AE out of the grey-image comparison in all benchmarks (c1 = None); (d) extra baselines: majority (2a) and PubMedBERT (2b). Without the variable, nothing changes.
ARTICLE_RUNS = os.environ.get("ARTICLE2_RUNS") == "1"; PJR = f"{_MEDVQA_ROOT}/v7_txt/article_runs/results"
ARROWS["VQARAD_REDIV"] = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv/vqa_vqa_rad_test.arrow"; RESULTS_DIR["VQARAD_REDIV"] = PJR
ARROWS["VQARAD_QLEVEL"] = f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_qlevel/vqa_vqa_rad_test.arrow"; RESULTS_DIR["VQARAD_QLEVEL"] = PJR   # 19/10 (vqarad_question_split)
def extra_baselines(bench):
    """17/10 (article_runs, question_only_sets): majority (one file per benchmark) and PubMedBERT (one per seed); qid -> correct dicts. VQARAD_RADORIG uses those of VQARAD (same training)."""
    b = {"VQARAD_RADORIG": "VQARAD"}.get(bench, bench); mj = f"{PJR}/MAJ_question_only_{b}.json"; bt = {s: (f"{PJR}/BERT_question_only_{b}_seed{s}.json" if os.path.exists(f"{PJR}/BERT_question_only_{b}_seed{s}.json") else f"{PJR}/BERT2_question_only_{b}_seed{s}.json") for s in range(3)}   # 19/10: VQARAD_QLEVEL only has the v2 version (vqarad_question_split)
    return {"maj": preds(mj) if os.path.exists(mj) else None, "maj_src": mj, "bert": {s: (preds(p) if os.path.exists(p) else None) for s, p in bt.items()}, "bert_src": bt}
def pub(bench, fam, pe, s):   # name published in p0 (SLAKE seed0 without suffix)
    return "%s_%s_%s" % (fam, pe, bench) + ("" if (bench in ("SLAKE", "VQARAD") and s == 0) else "_seed%d" % s)   # 14/10: VQA-RAD like SLAKE
def cells(bench, fam, s):
    c = _cells(bench, fam, s)
    if ARTICLE_RUNS and c:
        c = dict(c)
        if fam == "M3AE": c["c1"] = None   # 17/10 (article_runs): M3AE leaves the grey-image comparison (its text-only is already trained with grey image)
        elif bench == "MIMIC": c["c1"] = f"{PJR}/{c['frozen_line']}_C1.json"   # 17/10 (article_runs, baselines): C1 on the 1 499 closed-form
    return c
def _cells(bench, fam, s):
    if bench in ("VQARAD_REDIV", "VQARAD_QLEVEL"):   # 17/10 (article_runs, vqarad_image_split): everything in v7_txt/article_runs/results; 19/10: VQARAD_QLEVEL (vqarad_question_split)
        if fam == "M3AE": return {"frozen": f"{PJR}/M3AE_SCRATCH_{bench}_seed{s}_C0.json", "adapted": None, "c1": None, "frozen_line": f"M3AE_SCRATCH_{bench}_seed{s}", "adapted_line": None, "note": "M3AE: full fine-tuning (the only cell)"}
        if fam == "QWEN2VL": return {"frozen": f"{PJR}/QWEN2VL_LORA_{bench}_seed{s}_C0.json", "adapted": f"{PJR}/QWEN2VL_LORA_VT_{bench}_seed{s}_C0.json", "c1": f"{PJR}/QWEN2VL_LORA_{bench}_seed{s}_C1.json", "frozen_line": f"QWEN2VL_LORA_{bench}_seed{s}", "adapted_line": f"QWEN2VL_LORA_VT_{bench}_seed{s}"}
        return {"frozen": f"{PJR}/SMOLVLM_LORA_FT_{bench}_seed{s}_C0.json", "adapted": f"{PJR}/SMOLVLM_LORA_{bench}_seed{s}_C0.json", "c1": f"{PJR}/SMOLVLM_LORA_FT_{bench}_seed{s}_C1.json", "frozen_line": f"SMOLVLM_LORA_FT_{bench}_seed{s}", "adapted_line": f"SMOLVLM_LORA_{bench}_seed{s}"}
    if bench == "VQARAD_RADORIG":   # 15/10 (the candidate, revision A1 of paper 2): VLM from the original package (_RADORIG, v7_txt/results); M3AE same as VQARAD (already used the original package)
        if fam == "M3AE": return _cells("VQARAD", "M3AE", s)
        R7 = f"{V7}/results"
        if fam == "QWEN2VL": return {"frozen": f"{R7}/QWEN2VL_LORA_VQARAD_RADORIG_seed{s}_C0.json", "adapted": f"{R7}/QWEN2VL_LORA_VT_VQARAD_RADORIG_seed{s}_C0.json", "c1": f"{R7}/QWEN2VL_LORA_VQARAD_RADORIG_seed{s}_C1.json", "frozen_line": f"QWEN2VL_LORA_VQARAD_RADORIG_seed{s}", "adapted_line": f"QWEN2VL_LORA_VT_VQARAD_RADORIG_seed{s}"}
        return {"frozen": f"{R7}/SMOLVLM_LORA_FT_VQARAD_RADORIG_seed{s}_C0.json", "adapted": f"{R7}/SMOLVLM_LORA_VQARAD_RADORIG_seed{s}_C0.json", "c1": f"{R7}/SMOLVLM_LORA_FT_VQARAD_RADORIG_seed{s}_C1.json", "frozen_line": f"SMOLVLM_LORA_FT_VQARAD_RADORIG_seed{s}", "adapted_line": f"SMOLVLM_LORA_VQARAD_RADORIG_seed{s}"}
    if bench == "MIMIC":   # v7_mimic (3e): frozen = QWEN2VL_LORA_FROZEN, fine-tuned = QWEN2VL_LORA_VT; 14/10 (exec2): C1 = grey image of the frozen row in v7_mimic/results (same convention as HEAL/Kvasir/SLAKE)
        if fam == "SMOLVLM":   # 13/10: SmolVLM2 on MIMIC (TXT x3 completed), same mapping as HEAL: frozen = LoRA FT (frozen tower), fine-tuned = published LoRA; 14/10: C1 of the frozen
            return {"frozen": f"{MIMIC}/results/SMOLVLM_LORA_FT_MIMIC_seed{s}_C0.json", "adapted": f"{MIMIC}/results/SMOLVLM_LORA_MIMIC_seed{s}_C0.json", "c1": f"{MIMIC}/results/SMOLVLM_LORA_FT_MIMIC_seed{s}_C1.json", "frozen_line": f"SMOLVLM_LORA_FT_MIMIC_seed{s}", "adapted_line": f"SMOLVLM_LORA_MIMIC_seed{s}"}
        if fam == "M3AE":   # 13/10: M3AE on MIMIC (two arms, as in the other benchmarks): C0 = full fine-tune with image; TXT = M3AE_TXT_MIMIC; 14/10: C1 = M3AE_MIMIC_seed s_C1 (grey image on the published row)
            return {"frozen": f"{MIMIC}/results/M3AE_MIMIC_seed{s}_C0.json", "adapted": None, "c1": f"{MIMIC}/results/M3AE_MIMIC_seed{s}_C1.json", "frozen_line": f"M3AE_MIMIC_seed{s}", "adapted_line": None, "note": "M3AE: full fine-tune (the only cell); TXT = constant grey image in train/val/test"}
        if fam != "QWEN2VL": return None
        return {"frozen": f"{MIMIC}/results/QWEN2VL_LORA_FROZEN_MIMIC_seed{s}_C0.json", "adapted": f"{MIMIC}/results/QWEN2VL_LORA_VT_MIMIC_seed{s}_C0.json", "c1": f"{MIMIC}/results/QWEN2VL_LORA_FROZEN_MIMIC_seed{s}_C1.json", "frozen_line": f"QWEN2VL_LORA_FROZEN_MIMIC_seed{s}", "adapted_line": f"QWEN2VL_LORA_VT_MIMIC_seed{s}"}
    if fam == "M3AE":   # M3AE: only the full fine-tune published (no frozen/fine-tuned encoder cell); "frozen" = published C0, "adapted" = n/a
        c0 = {"HEAL": f"{ISBI}/M3AE_HEAL_seed{s}_C0.json", "KVASIR": f"{P0R}/M3AE_KVASIR_seed{s}_C0.json", "SLAKE": f"{P0R}/M3AE_SCRATCH_SLAKE" + ("" if s == 0 else f"_seed{s}") + "_C0.json", "VQARAD": f"{P0R}/M3AE_SCRATCH_VQARAD" + ("" if s == 0 else f"_seed{s}") + "_C0.json"}[bench]
        c1 = {"HEAL": f"{V7}/results/M3AE_HEAL_seed{s}_C1.json", "KVASIR": f"{P0R}/M3AE_KVASIR_seed{s}_C1.json", "SLAKE": f"{P0R}/M3AE_SCRATCH_SLAKE" + ("" if s == 0 else f"_seed{s}") + "_C1.json", "VQARAD": f"{P0R}/M3AE_SCRATCH_VQARAD" + ("" if s == 0 else f"_seed{s}") + "_C1.json"}[bench]
        return {"frozen": c0, "adapted": None, "c1": c1, "frozen_line": c0.split("/")[-1][:-8], "adapted_line": None, "note": "M3AE: full fine-tune (the only cell); TXT = constant grey image in train/val/test"}
    if bench == "HEAL":   # 14/10 (exec2): C1 = grey image of the frozen row, in v7_txt/results (14/10 closing queue)
        if fam == "QWEN2VL": return {"frozen": f"{ISBI}/QWEN2VL_LORA_HEAL_seed{s}_C0.json", "adapted": f"{V6R}/QWEN2VL_LORA_VT_HEAL_seed{s}_C0.json", "c1": f"{V7}/results/QWEN2VL_LORA_HEAL_seed{s}_C1.json", "frozen_line": f"QWEN2VL_LORA_HEAL_seed{s}", "adapted_line": f"QWEN2VL_LORA_VT_HEAL_seed{s}"}
        return {"frozen": f"{V6R}/SMOLVLM_LORA_FT_HEAL_seed{s}_C0.json", "adapted": f"{ISBI}/SMOLVLM_LORA_HEAL_seed{s}_C0.json", "c1": f"{V7}/results/SMOLVLM_LORA_FT_HEAL_seed{s}_C1.json", "frozen_line": f"SMOLVLM_LORA_FT_HEAL_seed{s}", "adapted_line": f"SMOLVLM_LORA_HEAL_seed{s}"}
    if fam == "QWEN2VL": return {"frozen": f"{P0R}/{pub(bench, 'QWEN2VL', 'LORA', s)}_C0.json", "adapted": f"{V5Q}/QWEN2VL_LORA_VT_{bench}_seed{s}_C0.json", "c1": f"{P0R}/{pub(bench, 'QWEN2VL', 'LORA', s)}_C1.json", "frozen_line": pub(bench, "QWEN2VL", "LORA", s), "adapted_line": f"QWEN2VL_LORA_VT_{bench}_seed{s}"}
    return {"frozen": f"{V5S}/SMOLVLM_LORA_FT_{bench}_seed{s}_C0.json", "adapted": f"{P0R}/{pub(bench, 'SMOLVLM', 'LORA', s)}_C0.json", "c1": f"{V5S}/SMOLVLM_LORA_FT_{bench}_seed{s}_C1.json", "frozen_line": f"SMOLVLM_LORA_FT_{bench}_seed{s}", "adapted_line": pub(bench, "SMOLVLM", "LORA", s)}
def sha(p):
    if p is None: return None
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
def preds(p): return {int(x["qid"]): bool(x["correct"]) for x in json.load(open(p))["predictions"]}
def mimic_items():
    import pandas as pd
    d = pd.read_parquet(f"{MIMIC}/data/items_test.parquet"); return {int(r.qid): r for r in d.itertuples()}
def mimic_floors():
    """TFIDF subset/full and prior-majority subset/full per item (dicts qid -> correct), gold from items_test.parquet."""
    from med_vlm.metrics.process_text import answer_matches, normalize_text
    it = mimic_items(); out = {}
    for tag in ("subset", "full"):
        d = json.load(open(f"{MIMIC}/results/TFIDF_question_only_MIMIC_{tag}.json"))
        tfidf = {int(q): bool(answer_matches(normalize_text(str(p)), [normalize_text(str(it[int(q)].answer))])) for q, p in d["predictions"].items()}
        pr = {int(q): bool(answer_matches(normalize_text(str(p)), [normalize_text(str(it[int(q)].answer))])) for q, p in d["prior_predictions"].items()}
        # decision of the candidate (10/10): TFIDF (LR yes/no) only on the CLOSED-FORM; on the open ones the baseline is the majority answer per template ("prior by template"), never labeled TFIDF
        out["TFIDF_" + tag + "_fechadas"] = {q: v for q, v in tfidf.items() if it[q].answer_type == "closed"}
        out["prior_" + tag + "_abertas"] = {q: v for q, v in pr.items() if it[q].answer_type == "open"}
        out["baseline_" + tag] = {**out["TFIDF_" + tag + "_fechadas"], **out["prior_" + tag + "_abertas"]}
    return out
def tfidf_fechadas(bench):
    """15/10/2026 (the candidate): paper 2 only with closed-form; TFIDF = only the classifier on the closed-form (without the MIMIC composite)."""
    if bench == "MIMIC": return mimic_floors()["TFIDF_subset_fechadas"], f"{MIMIC}/results/TFIDF_question_only_MIMIC_subset.json -- TFIDF subset (LR yes/no) only on closed-form"
    if bench == "SLAKE": p = f"{_MEDVQA_ROOT}/v6_medida/design/tfidf_by_type/SLAKE_closed_recomputed.json"; return preds(p), p + " (closed-form only)"
    if bench in ("VQARAD_REDIV", "VQARAD_QLEVEL"): p = f"{PJR}/TFIDF_question_only_{bench}.json"; return preds(p), p + " (TF-IDF + LR, harness/baselines/question_only.py unchanged, on the 1 469 closed-form questions of the re-split train; 418 closed-form questions of the new test)"   # 17/10 (article_runs, vqarad_image_split)
    if bench in ("VQARAD", "VQARAD_RADORIG"): p = f"{P0R}/TFIDF_question_only_VQARAD_C0.json"; return preds(p), p + " (classifier trained on the closed-form questions of the M3AE trainset.json = 2 758 train; 272 closed-form; reproduced item by item by harness/v7_txt/tfidf_vqarad_orig.py, so_fechadas variant)"
    return tfidf_preds(bench)   # Kvasir and HEAL: TFIDF is already only on the closed-form
def tfidf_preds(bench):
    if bench == "MIMIC": return mimic_floors()["baseline_subset"], f"{MIMIC}/results/TFIDF_question_only_MIMIC_subset.json -- baseline = TFIDF subset on closed-form + prior by template (majority answer in the train subset) on open; TFIDF full and prior full in the extra columns"
    if bench == "KVASIR": return preds(f"{ISBI}/TFIDF_question_only_KVASIR_C0.json"), f"{ISBI}/TFIDF_question_only_KVASIR_C0.json"
    if bench in ("VQARAD_REDIV", "VQARAD_QLEVEL"): return tfidf_fechadas(bench)   # 17/10: closed-form only
    if bench == "VQARAD_RADORIG": p = f"{V7}/results/TFIDF_question_only_VQARAD_orig.json"; return preds(p), p + " (TFIDF retrained on the 2 758 train of the original package, all 451; harness/v7_txt/tfidf_vqarad_orig.py; 15/10)"
    if bench == "VQARAD": return preds(f"{P0R}/TFIDF_question_only_VQARAD_C0.json"), f"{P0R}/TFIDF_question_only_VQARAD_C0.json (TFIDF published from p0)"   # 14/10
    if bench == "SLAKE":
        d = {}; fs = [f"{_MEDVQA_ROOT}/v6_medida/design/tfidf_by_type/SLAKE_closed_recomputed.json", f"{_MEDVQA_ROOT}/v6_medida/design/tfidf_by_type/SLAKE_open.json"]
        for f in fs: d.update(preds(f))
        return d, "+".join(fs)
    p = f"{V7}/results/TFIDF_question_only_HEAL_aligned.json"
    if os.path.exists(p): return preds(p), p + " (TFIDF alinhado aos 1446 qids, harness/v7_txt/tfidf_heal_aligned.py)"
    return None, "Per-item TFIDF for HEAL does not exist on disk (tfidf_by_type_heal.json only has aggregates): McNemar TXT vs TFIDF not computed; TFIDF by class read from the aggregates"
def classes(bench, df):
    """qid -> dict of classes; and qid -> image_id."""
    img = {}; cls = collections.defaultdict(dict)
    for _, r in df.iterrows():
        for qid, t in zip(r["question_id"], r["answer_type"]): img[int(qid)] = r["image_id"]; cls[int(qid)]["answer_type"] = {0: "closed", 1: "open"}.get(int(t), str(t))
    if bench == "KVASIR":
        inv = json.load(open(f"{ISBI}/inventory_kvasir_test_clean.json"))
        for q in inv["questions"]: cls[q["qid"]]["question_class"] = q["question_class"]
    elif bench == "SLAKE":
        inv = json.load(open(f"{ISBI}/inventory_slake_test.json"))
        for q in inv["questions"]: cls[q["qid"]]["content_type"] = q.get("content_type"); cls[q["qid"]]["question_class"] = q.get("question_class")
    elif bench == "MIMIC":
        for q, r in mimic_items().items(): cls[q]["question_type"] = r.question_type; cls[q]["template"] = f"{r.template_id} {r.template}"
    elif bench in ("VQARAD", "VQARAD_RADORIG", "VQARAD_REDIV", "VQARAD_QLEVEL"):   # 14/10: no own inventory; only answer_type (from the arrow); 17/10: VQARAD_REDIV
        pass
    else:
        inv = json.load(open(f"{ISBI}/inventory_heal_test.json"))
        for q in inv["questions"]: cls[q["qid"]]["question_class"] = q["question_class"]; cls[q["qid"]]["anatomy"] = q["anatomy"]; cls[q["qid"]]["evaluable"] = q["evaluable"]
    return cls, img
def acc_block(c: dict, qids, img, bs):
    v = np.array([c[q] for q in qids], float); return {"n": len(qids), "acc": float(v.mean()) if len(qids) else None, "acc_ci": bs.ci_mean_diff([img[q] for q in qids], v) if len(qids) else None}
def diff_block(a: dict, b: dict, qids, img, bs):
    va = np.array([a[q] for q in qids], float); vb = np.array([b[q] for q in qids], float)
    return {"n": len(qids), "diff": float((va - vb).mean()), "diff_ci": bs.ci_mean_diff([img[q] for q in qids], va - vb), "mcnemar_midp": mcnemar_midp(va.astype(bool), vb.astype(bool)),
            "txt_right_other_wrong": int(((va == 1) & (vb == 0)).sum()), "txt_wrong_other_right": int(((va == 0) & (vb == 1)).sum()), "both_right": int(((va == 1) & (vb == 1)).sum()), "both_wrong": int(((va == 0) & (vb == 0)).sum())}
NOTA_REPRO = "Cross-environment reproduction (recorded on 14/10/2026, decision of the candidate): the band of 2,7–5,9 % item-by-item disagreement between environments applies to the generative models in bf16. M3AE, an argmax classifier, reproduces item by item across environments: the VQA-RAD C0, seeds 1 and 2, with the Colab weights recovered from the Drive and run on gpu1, has 0 of 451 predictions different from the published ones (generated on Colab) in both seeds (v7_txt/analysis/c0repro_vqarad.json). Consequently, the 10–11 % divergence of M3AE_SCRATCH SLAKE seeds 1 and 2 (39 and 43 of 388 items) against the Colab C0 is a difference of weights (retrainings of 15/09), not of environment."   # 14/10/2026 (the candidate)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default=f"{V7}/analysis/txt_floors.json"); ap.add_argument("--out-md", default=f"{V7}/docs/txt_floors.md"); a = ap.parse_args()
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": 2000, "seed": 20260913, "benchmarks": {}}
    for bench in ("SLAKE", "HEAL", "KVASIR", "MIMIC", "VQARAD", "VQARAD_RADORIG"):   # 14/10: VQA-RAD; 15/10: VQA-RAD fonte unica (A1)
        df = read_arrow(ARROWS[bench]); cls, img = classes(bench, df); tfidf, m0src = tfidf_preds(bench)
        allq = sorted(img); bs = ImageBootstrap([img[q] for q in allq], 2000, 20260913)
        B = {"arrow": ARROWS[bench], "arrow_sha256": sha(ARROWS[bench]), "n_questions": len(allq), "n_images": len(bs.ids), "tfidf_source": m0src, "lines": [], "holm": {}}
        if bench == "HEAL": B["tfidf_by_class_aggregates"] = json.load(open(f"{ISBI}/tfidf_by_type_heal.json"))["by_class"]["evaluable"]
        extra = mimic_floors() if bench == "MIMIC" else {}
        for fam in (("QWEN2VL",) if bench == "MIMIC" else ("QWEN2VL", "SMOLVLM", "M3AE")):
            for s in range(3):
                line = f"{fam}_LORA_TXT_{bench}_seed{s}" if fam != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(bench, bench)}_seed{s}"; tp = f"{RESULTS_DIR.get(bench, V7 + '/results')}/{line}_TXT.json"; c = cells(bench, fam, s)
                rec = {"line": line, "family": fam, "seed": s, "txt_file": tp, "cells": c}
                if not os.path.exists(tp): rec["status"] = "TXT missing"; B["lines"].append(rec); continue
                if not os.path.exists(c["frozen"]) or (c["adapted"] and not os.path.exists(c["adapted"])): rec["status"] = "C0 frozen/fine-tuned missing"; B["lines"].append(rec); continue
                T = preds(tp); F = preds(c["frozen"]); A = preds(c["adapted"]) if c["adapted"] else None; C1 = preds(c["c1"]) if c["c1"] and os.path.exists(c["c1"]) else None
                qids = [q for q in allq if q in T and q in F and (A is None or q in A)]; rec["n_aligned"] = len(qids); rec["sha256"] = {"txt": sha(tp), "frozen": sha(c["frozen"]), "adapted": sha(c["adapted"]), "c1": sha(c["c1"]) if C1 else None}
                rec["overall"] = {"TXT": acc_block(T, qids, img, bs), "C0_frozen": acc_block(F, qids, img, bs)}
                if A is not None: rec["overall"]["C0_adapted"] = acc_block(A, qids, img, bs)
                if C1: q1 = [q for q in qids if q in C1]; rec["overall"]["C1_frozen"] = acc_block(C1, q1, img, bs)
                if tfidf: qm = [q for q in qids if q in tfidf]; rec["overall"]["TFIDF"] = acc_block(tfidf, qm, img, bs); rec["TXT_vs_M0"] = diff_block(T, tfidf, qm, img, bs)
                for k, dd in extra.items():
                    if k != "baseline_subset": qk = [q for q in qids if q in dd]; rec["overall"][k] = acc_block(dd, qk, img, bs); rec["TXT_vs_" + k] = diff_block(T, dd, qk, img, bs)
                rec["TXT_vs_C0_frozen"] = diff_block(T, F, qids, img, bs)
                if A is not None: rec["TXT_vs_C0_adapted"] = diff_block(T, A, qids, img, bs)
                rec["by_class"] = {}
                for key in sorted({k for q in qids for k in cls[q] if k != "evaluable"}):
                    for val in sorted({str(cls[q].get(key)) for q in qids}):
                        sub = [q for q in qids if str(cls[q].get(key)) == val]
                        if len(sub) < 5: continue
                        blk = {"TXT": acc_block(T, sub, img, bs), "C0_frozen": acc_block(F, sub, img, bs), "C0_adapted": acc_block(A, sub, img, bs) if A is not None else None, "TXT_vs_C0_frozen": diff_block(T, F, sub, img, bs)}
                        if C1: blk["C1_frozen"] = acc_block(C1, [q for q in sub if q in C1], img, bs)
                        if tfidf: sm = [q for q in sub if q in tfidf]; blk["TFIDF"] = acc_block(tfidf, sm, img, bs); blk["TXT_vs_M0"] = diff_block(T, tfidf, sm, img, bs) if sm else None
                        for k, dd in extra.items():
                            if k != "baseline_subset": sk = [q for q in sub if q in dd]; blk[k] = acc_block(dd, sk, img, bs) if sk else None
                        rec["by_class"][f"{key}={val}"] = blk
                rec["status"] = "ok"; rec["sha256"]["adapted"] = sha(c["adapted"]) if c["adapted"] else None; B["lines"].append(rec); print(bench, line, "n", len(qids), "TXT", round(rec["overall"]["TXT"]["acc"], 4), "C0f", round(rec["overall"]["C0_frozen"]["acc"], 4), flush=True)
        pv = {}
        for r in B["lines"]:
            if r.get("status") == "ok":
                pv[r["line"] + "|TXT_vs_C0_frozen"] = r["TXT_vs_C0_frozen"]["mcnemar_midp"]["p_midp"]
                if "TXT_vs_M0" in r: pv[r["line"] + "|TXT_vs_M0"] = r["TXT_vs_M0"]["mcnemar_midp"]["p_midp"]
        B["holm"] = holm(pv) if pv else {}
        B["by_family"] = {}
        for fam in ("QWEN2VL", "SMOLVLM", "M3AE"):
            rs = [r for r in B["lines"] if r["family"] == fam and r.get("status") == "ok"]
            if rs: B["by_family"][fam] = {"n_seeds": len(rs), **{k: {"mean": float(np.mean([r["overall"][k]["acc"] for r in rs])), "sd": float(np.std([r["overall"][k]["acc"] for r in rs], ddof=1)) if len(rs) > 1 else None} for k in rs[0]["overall"] if all(k in r["overall"] for r in rs)}}
        out["benchmarks"][bench] = B
    json.dump(out, open(a.out_json, "w"), indent=1)
    M = ["# Text-only baselines (P1.4)", "", NOTA_REPRO, "", f"Generated {out['t']}. Acc(TXT) = trained text-decoder-only approach (same recipe, no image, LoRA only on the LM); C0 frozen / fine-tuned = rows already trained; C1 = grey image (where it exists); TFIDF = TF-IDF+LR per item (Kvasir, SLAKE) or aggregates (HEAL). 95 % CI by bootstrap over images (2000, seed 20260913, shared replicates); McNemar mid-p; Holm per benchmark over all tests of the table. No interpretation.", ""]
    for bench, B in out["benchmarks"].items():
        M += [f"## {bench} (n questions {B['n_questions']}, images {B['n_images']})", "", "| row | n | Acc TXT [CI] | Acc frozen C0 | Acc fine-tuned C0 | Acc C1 | TFIDF | TXT-C0f [CI] mid-p (Holm) | TXT-TFIDF [CI] mid-p (Holm) | win/loss/tie TXT vs C0f |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in B["lines"]:
            if r.get("status") != "ok": M.append(f"| {r['line']} | {r.get('status')} |"); continue
            o = r["overall"]; f = lambda b: f"{b['acc']:.4f} [{b['acc_ci'][0]:.3f}, {b['acc_ci'][1]:.3f}]" if b and b.get("acc") is not None else "-"
            d1 = r["TXT_vs_C0_frozen"]; dm = r.get("TXT_vs_M0")
            M.append(f"| {r['line']} | {r['n_aligned']} | {f(o['TXT'])} | {f(o['C0_frozen'])} | {f(o.get('C0_adapted'))} | {f(o.get('C1_frozen'))} | {f(o.get('TFIDF'))} | {d1['diff']:+.4f} [{d1['diff_ci'][0]:+.3f}, {d1['diff_ci'][1]:+.3f}] p={d1['mcnemar_midp']['p_midp']:.3g} ({B['holm'].get(r['line'] + '|TXT_vs_C0_frozen', float('nan')):.3g}) | " + (f"{dm['diff']:+.4f} [{dm['diff_ci'][0]:+.3f}, {dm['diff_ci'][1]:+.3f}] p={dm['mcnemar_midp']['p_midp']:.3g} ({B['holm'].get(r['line'] + '|TXT_vs_M0', float('nan')):.3g})" if dm else "-") + f" | {d1['txt_right_other_wrong']}/{d1['txt_wrong_other_right']}/{d1['both_right'] + d1['both_wrong']} |")
        if bench == "MIMIC":
            M += ["", "MIMIC: the 'TFIDF' column above is the composite baseline = TFIDF subset (LR yes/no) on the CLOSED-FORM + prior by template (majority answer in training) on the OPEN. By row, the parts and alternatives (Acc [CI]; TXT - baseline [CI]):", ""]
            for r in B["lines"]:
                if r.get("status") == "ok": M.append(f"- {r['line']}: " + "; ".join(f"{k} {f(r['overall'][k])}, TXT-{k} {r['TXT_vs_' + k]['diff']:+.4f} [{r['TXT_vs_' + k]['diff_ci'][0]:+.3f}, {r['TXT_vs_' + k]['diff_ci'][1]:+.3f}]" for k in ("TFIDF_subset_fechadas", "TFIDF_full_fechadas", "prior_subset_abertas", "prior_full_abertas", "baseline_full") if k in r["overall"]))
        M += ["", "Mean (SD) across seeds:", ""]
        for fam, v in B.get("by_family", {}).items(): M.append(f"- {fam} (n={v['n_seeds']}): " + "; ".join(f"{k} {x['mean']:.4f} ({x['sd']:.4f})" if x["sd"] is not None else f"{k} {x['mean']:.4f}" for k, x in v.items() if k != "n_seeds"))
        M += ["", "### By class (same qids; Acc [95 % CI bootstrap per image])", "", "| row | class | n | TXT | frozen C0 | fine-tuned C0 | C1 | TFIDF | TXT-C0f [CI] | TXT-TFIDF [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
        g = lambda z: f"{z['acc']:.3f} [{z['acc_ci'][0]:.3f}, {z['acc_ci'][1]:.3f}]" if z and z.get("acc") is not None else "-"
        for r in B["lines"]:
            if r.get("status") != "ok": continue
            for k, b in r["by_class"].items():
                dm = b.get("TXT_vs_M0")
                M.append(f"| {r['line']} | {k} | {b['TXT']['n']} | {g(b['TXT'])} | {g(b['C0_frozen'])} | {g(b['C0_adapted'])} | {g(b.get('C1_frozen'))} | {g(b.get('TFIDF'))} | {b['TXT_vs_C0_frozen']['diff']:+.3f} [{b['TXT_vs_C0_frozen']['diff_ci'][0]:+.3f}, {b['TXT_vs_C0_frozen']['diff_ci'][1]:+.3f}] | " + (f"{dm['diff']:+.3f} [{dm['diff_ci'][0]:+.3f}, {dm['diff_ci'][1]:+.3f}]" if dm else "-") + " |")
        M += ["", "### Right/wrong/tie per item, TXT vs TFIDF (TXT right & TFIDF wrong / TXT wrong & TFIDF right / both right / both wrong)", ""]
        for r in B["lines"]:
            if r.get("status") == "ok" and r.get("TXT_vs_M0"): d = r["TXT_vs_M0"]; M.append(f"- {r['line']}: {d['txt_right_other_wrong']} / {d['txt_wrong_other_right']} / {d['both_right']} / {d['both_wrong']} (n={d['n']})")
            elif r.get("status") == "ok": M.append(f"- {r['line']}: per-item TFIDF not available for this benchmark")
        M += ["", "### Right/wrong/tie TXT vs TFIDF per class (TXT right & TFIDF wrong / TXT wrong & TFIDF right / both right / both wrong; n)", "", "| row | class | TXT>TFIDF | TFIDF>TXT | both correct | both wrong | n |", "|---|---|---|---|---|---|---|"]
        for r in B["lines"]:
            if r.get("status") != "ok": continue
            for k, b in r["by_class"].items():
                dm = b.get("TXT_vs_M0")
                if dm: M.append(f"| {r['line']} | {k} | {dm['txt_right_other_wrong']} | {dm['txt_wrong_other_right']} | {dm['both_right']} | {dm['both_wrong']} | {dm['n']} |")
        if bench == "HEAL": M += ["", f"TFIDF HEAL (aggregates tfidf_by_type_heal.json, n_test 1512, alignment to the 1446 evaluable not possible per item): {json.dumps(B['tfidf_by_class_aggregates'])}"]
        M.append("")
    open(a.out_md, "w").write("\n".join(M) + "\n"); print("FIM", a.out_json)
if __name__ == "__main__": main()
