"""17/10/2026 (the candidate's decision, 7-B): swapped-image control across the 9 rows of the decomposition (M3AE full fine-tune, Qwen2-VL LoRA VT, SmolVLM2 LoRA
published; cell A_ft of the decomposition), in the closed-form questions, across the five benchmarks of the paper (Kvasir, SLAKE, HEAL, MIMIC, VQA-RAD redivided). Swap = C2 (derangement
of the test images: seed 42 in p0 and in the redivided VQA-RAD; HEAL vqa_heal_test_c2; MIMIC seed 20260913, the Measure's arrow). Per row: A_swap, A - A_swap
[CI 95 % bootstrap by image, 2 000, seed 20260913], A_swap - A_text [CI]; mean (SD) per family. Qids = closed-form with C0, C2 and TXT. No interpretation.
Usage: ARTICLE2_RUNS=1 python -m harness.v7_txt.analise_troca --out <folder>"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, hashlib, json, os, time
import numpy as np
from harness.v7_txt.txt_floors import ARROWS, RESULTS_DIR, V7, TXT_BENCH_M3AE, cells, classes, preds
from harness.stats.bootstrap import ImageBootstrap
from harness.interventions.arrow_io import read_arrow
W = f"{_MEDVQA_ROOT}"; P_MAIN = f"{W}/doutorado/medvqa-main/results"; PJR = f"{W}/v7_txt/article_runs/results"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def swap_file(b, fam, s, line):
    if b == "KVASIR": return f"{P_MAIN}/{line}_C2.json" if fam != "QWEN2VL" else f"{W}/v5_qwen_vt/results/{line}_C2.json"
    if b == "SLAKE": return f"{P_MAIN}/{line}_C2.json" if fam != "QWEN2VL" else f"{W}/v5_qwen_vt/results/{line}_C2.json"
    if b in ("HEAL", "MIMIC") and os.environ.get("TROCA_V2") == "1": return f"{PJR}/{line}_C2v2.json"   # 19/10 (image_swap_v2): swap redone
    if b == "HEAL": return f"{W}/v6_tower_heal/results/{line}_C2.json"
    if b == "MIMIC": return f"{W}/v7_mimic/results/{line}_C2.json"
    if b in ("VQARAD_REDIV", "VQARAD_QLEVEL"): return f"{PJR}/{line}_C2.json"
    if b == "VQARAD_RADORIG": return f"{P_MAIN}/{line}_C2.json" if fam == "M3AE" else f"{PJR}/NAO_RODADO_{line}_C2.json"   # 19/10: official split -- only M3AE has swap (p0); VLM _RADORIG without swap   # 19/10: VQARAD_QLEVEL (vqarad_question_split)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--bases", default="KVASIR,SLAKE,HEAL,MIMIC,VQARAD_REDIV"); a = ap.parse_args()
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": os.path.abspath(__file__), "script_sha256": sha(__file__), "n_boot": 2000, "seed": 20260913, "bases": {}}
    for b in a.bases.split(","):
        df = read_arrow(ARROWS[b]); cls, img = classes(b, df); closed = sorted(q for q in img if cls[q].get("answer_type") == "closed"); bs = ImageBootstrap([img[q] for q in closed], 2000, 20260913)
        B = {"n_fechadas": len(closed), "linhas": [], "ausentes": []}
        for fam in ("QWEN2VL", "SMOLVLM", "M3AE"):
            for s in range(3):
                c = cells(b, fam, s); a_path = c["adapted"] or c["frozen"]; line = c["adapted_line"] or c["frozen_line"]
                tl = f"{fam}_LORA_TXT_{b}_seed{s}" if fam != "M3AE" else f"M3AE_TXT_{TXT_BENCH_M3AE.get(b, b)}_seed{s}"; tp = f"{RESULTS_DIR.get(b, V7 + '/results')}/{tl}_TXT.json"; sp = swap_file(b, fam, s, line)
                if not all(os.path.exists(p) for p in (a_path, tp, sp)): B["ausentes"].append({"linha": line, "c0": os.path.exists(a_path), "txt": os.path.exists(tp), "c2": sp if not os.path.exists(sp) else True}); continue
                A, T, S = preds(a_path), preds(tp), preds(sp); qs = [q for q in closed if q in A and q in T and q in S]
                v = lambda d: np.array([d[q] for q in qs], float); ia = [img[q] for q in qs]
                r = {"familia": fam, "seed": s, "linha": line, "n": len(qs), "A": float(v(A).mean()), "A_text": float(v(T).mean()), "A_swap": float(v(S).mean()), "A_swap_ic": bs.ci_mean_diff(ia, v(S)),
                     "A_menos_A_swap": float((v(A) - v(S)).mean()), "A_menos_A_swap_ic": bs.ci_mean_diff(ia, v(A) - v(S)), "A_swap_menos_A_text": float((v(S) - v(T)).mean()), "A_swap_menos_A_text_ic": bs.ci_mean_diff(ia, v(S) - v(T)),
                     "arquivos": {"c0": a_path, "txt": tp, "c2": sp}, "sha256": {"c0": sha(a_path), "txt": sha(tp), "c2": sha(sp)}}
                idx = np.array([bs.index[img[q]] for q in qs]); den = np.bincount(idx, minlength=len(bs.ids)).astype(float)   # 18/10: replicates for the CI of the mean across seeds
                r["_boot"] = {k: (bs.W @ np.bincount(idx, weights=dv, minlength=len(bs.ids))) / np.maximum(bs.W @ den, 1e-9) for k, dv in (("A_menos_A_swap", v(A) - v(S)), ("A_swap_menos_A_text", v(S) - v(T)))}
                B["linhas"].append(r)
        B["por_familia"] = {f: {k: {"mean": float(np.mean([r[k] for r in B["linhas"] if r["familia"] == f])), "sd": float(np.std([r[k] for r in B["linhas"] if r["familia"] == f], ddof=1)) if sum(r["familia"] == f for r in B["linhas"]) > 1 else None}
                                for k in ("A", "A_text", "A_swap", "A_menos_A_swap", "A_swap_menos_A_text")} for f in ("QWEN2VL", "SMOLVLM", "M3AE") if any(r["familia"] == f for r in B["linhas"])}
        for f in B["por_familia"]:
            for k in ("A_menos_A_swap", "A_swap_menos_A_text"):
                vs = [r["_boot"][k] for r in B["linhas"] if r["familia"] == f]; m = np.mean(vs, axis=0); B["por_familia"][f][k]["ic_media_sementes"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
        for r in B["linhas"]: r.pop("_boot", None)
        out["bases"][b] = B; print(b, len(B["linhas"]), "linhas;", len(B["ausentes"]), "ausentes", flush=True)
    SUFX = os.environ.get("TROCA_SUFIXO", ""); os.makedirs(a.out + "/analysis", exist_ok=True); os.makedirs(a.out + "/docs", exist_ok=True); json.dump(out, open(a.out + f"/analysis/analise_troca{SUFX}.json", "w"), indent=1)
    p = lambda x: f"{100 * x:.1f}"; d = lambda x, c: f"{100 * x:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"; ms = lambda v, sg=False: "-" if not v else ((f"{100 * v['mean']:+.1f}" if sg else f"{100 * v['mean']:.1f}") + (f" ({100 * v['sd']:.1f})" if v["sd"] is not None else ""))
    M = ["# Swapped-image control (7-B), closed-form questions, 9 rows of the decomposition", "", f"Gerado {out['t']} (sha256 {out['script_sha256'][:12]}). A = accuracy with the image (cell A_ft; M3AE: full fine-tune); A_swap = with the swapped image (C2); A_text = text-decoder-only of the same family and seed. CI 95 % by bootstrap per image (2 000, seed 20260913). No interpretation.", ""]
    for b, B in out["bases"].items():
        M += [f"## {b} (fechadas {B['n_fechadas']})", "", "| row | n | A | A_text | A_swap [CI] | A - A_swap [CI] | A_swap - A_text [CI] |", "|---|---|---|---|---|---|---|"]
        M += [f"| {r['linha']} | {r['n']} | {p(r['A'])} | {p(r['A_text'])} | {p(r['A_swap'])} [{p(r['A_swap_ic'][0])}, {p(r['A_swap_ic'][1])}] | {d(r['A_menos_A_swap'], r['A_menos_A_swap_ic'])} | {d(r['A_swap_menos_A_text'], r['A_swap_menos_A_text_ic'])} |" for r in B["linhas"]]
        M += ["", "| familia | A | A_text | A_swap | A - A_swap | A_swap - A_text |", "|---|---|---|---|---|---|"] + [f"| {f} | {ms(v['A'])} | {ms(v['A_text'])} | {ms(v['A_swap'])} | {ms(v['A_menos_A_swap'], True)} | {ms(v['A_swap_menos_A_text'], True)} |" for f, v in B["por_familia"].items()]
        if B["ausentes"]: M += ["", "Ausentes: " + json.dumps(B["ausentes"])]
        M.append("")
    open(a.out + f"/docs/analise_troca{SUFX}.md", "w").write("\n".join(M) + "\n"); print("FIM")
if __name__ == "__main__": main()
