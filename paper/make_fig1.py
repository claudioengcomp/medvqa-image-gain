"""Figure 1 of the paper (file fig1_v11): closed-form test accuracy of the fine-tuned models, stacked as
text-decoder-only accuracy + gain from the frozen visual encoder + gain from fine-tuning it, per benchmark and family,
with the TF-IDF, PubMedBERT and majority-answer baselines as horizontal lines.

Faithful port of the `fig1_v11` branch of the original (internal, not distributed) figure generator, reading only
the aggregates shipped in results/aggregates. Same matplotlib calls, in the same order, so the PDF/PNG are byte-identical
to the submitted figure when run with the pinned environment (environment/requirements-vlm.txt; matplotlib 3.11.1).

Usage:  python paper/make_fig1.py [--aggregates results/aggregates] [--out paper/figs] [--name fig1_v11]
Inputs: decomposicao_fechadas.json (per-family means/SD), heal_classifiers_train_only.json (HEAL-MedVQA classifiers, train-only protocol),
        m3ae_no_image/m3ae_frozen_vit.json (M3AE: text-only accuracy of the no-image model, frozen-ViT and fine-tuned gains); file names
        as renamed in this repository (see README, glossary)."""
import argparse, hashlib, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ap = argparse.ArgumentParser(); ap.add_argument("--aggregates", default=os.path.join(os.path.dirname(__file__), "..", "results", "aggregates")); ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figs")); ap.add_argument("--name", default="fig1_v11"); a = ap.parse_args()
AN = a.aggregates; OUT = a.out; os.makedirs(OUT, exist_ok=True)
DEC = json.load(open(f"{AN}/decomposicao_fechadas.json")); HEALCLS = json.load(open(f"{AN}/heal_classifiers_train_only.json")); FROZEN = json.load(open(f"{AN}/m3ae_no_image/m3ae_frozen_vit.json"))
VQ = "VQARAD_REDIV"; BENCH = ["KVASIR", "SLAKE", "HEAL", "MIMIC", VQ]; BL = {"KVASIR": "Kvasir-VQA-x1", "SLAKE": "SLAKE", "HEAL": "HEAL-MedVQA", "MIMIC": "Medical-Diff-VQA", VQ: "VQA-RAD"}
FAM = ["QWEN2VL", "SMOLVLM", "M3AE"]; FL = {"QWEN2VL": "Qwen2-VL-2B", "SMOLVLM": "SmolVLM2-2.2B", "M3AE": "M3AE"}; FLC = {"QWEN2VL": "Qwen", "SMOLVLM": "Smol", "M3AE": "M3AE"}
C = {"txt": "#9e9e9e", "enc": "#4c72b0", "ft": "#dd8452", "zs": "#2f2f2f", "zsg": "#a6c8e6"}; LS = {"tfidf": (0, (4, 2)), "bert": (0, (5, 1.5, 1, 1.5)), "maj": (0, (1, 1.5))}
ZM = [("MEDGEMMA", "MedGemma"), ("QWEN25VL", "Qwen2.5-VL"), ("MEDVLTHINKER_instr", "MedVLThinker")]
RC_V5 = {"font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral", "Times New Roman"], "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "mathtext.fontset": "stix"}
def _dat(b, heal_novo=False):
    B = DEC["benchmarks"][b]; L = [r for r in B["lines"] if r.get("status") == "ok" and r["family"] == "QWEN2VL"]
    fl = {"tfidf": L[0]["overall"]["tfidf"]["acc"], "bert": float(np.mean([r["overall"]["bert"]["acc"] for r in L])), "maj": L[0]["overall"]["maj"]["acc"]}
    assert all(abs(r["overall"]["tfidf"]["acc"] - fl["tfidf"]) < 1e-12 and abs(r["overall"]["maj"]["acc"] - fl["maj"]) < 1e-12 for r in L)
    if heal_novo and b == "HEAL":
        hl = [r for r in HEALCLS["heal_so_treino"]["linhas"] if r["familia"] == "QWEN2VL"]; fl["tfidf"] = hl[0]["acc_tfidf"]; fl["bert"] = float(np.mean([r["acc_bert"] for r in hl]))
    return fl, {}
def _draw(full, H, ncol=4, com_zs=True, heal_novo=False, curto=False, contorno=False, lw_c=0.4, m3ae_frozen=False, i17=None):
    EC = "#222222" if contorno else "white"; LWB = lw_c if contorno else 0.4; HK = {"hatchcolor": "white"} if contorno else {}
    fig = plt.figure(figsize=(7.2, H), layout="constrained"); fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.04, hspace=0.04); nr = 2 if full else 1
    gs = fig.add_gridspec(nr, len(BENCH), height_ratios=[1.25, 1] if full else [1])
    XF = np.array([0.0, 1.0, 2.0]); XZ = np.array([3.15, 3.9, 4.65]) if com_zs else np.array([]); xt = list(XF) + list(XZ); xl = [(FLC if curto else FL)[f] for f in FAM] + ([n for _, n in ZM] if com_zs else []); XL = (-0.55, 5.1) if com_zs else (-0.6, 2.6)
    axes = []
    for j, b in enumerate(BENCH):
        fl, zs = _dat(b, heal_novo); top = fig.add_subplot(gs[0, j]); axes.append(top)
        for i, f in enumerate(FAM):
            v = DEC["benchmarks"][b]["by_family"][f]; base = v["own_text_only_baseline"]["mean"]; tg = v["ganho_torre"]["mean"]; ad = (v.get("ganho_adaptacao") or {}).get("mean", 0.0) or 0.0; sd = v["soma"].get("sd") or 0.0
            if m3ae_frozen and f == "M3AE":
                _i17 = i17["bases"][b]; _top = base + tg; base = _i17["acuracias_media_dp"]["A_text"][0]; tg = _i17["diferencas"]["G_enc"]["media"]; ad = _i17["diferencas"]["G_ft"]["media"]
                assert abs(base + tg + ad - _top) < 0.002, (b, base + tg + ad, _top)
            top.bar(XF[i], base, 0.7, color=C["txt"], edgecolor=EC, lw=LWB)
            if tg >= 0: top.bar(XF[i], tg, 0.7, bottom=base, color=C["enc"], edgecolor=EC, lw=LWB)
            if contorno and ad == 0: pass
            elif ad >= 0: top.bar(XF[i], ad, 0.7, bottom=base + tg, color=C["ft"], edgecolor=EC, lw=LWB)
            else: top.bar(XF[i], -ad, 0.7, bottom=base + tg + ad, facecolor=C["ft"], edgecolor=EC, hatch="////", lw=LWB, **HK)
            if tg < 0: top.bar(XF[i], -tg, 0.7, bottom=base + tg, facecolor="none", edgecolor=C["enc"], hatch="////", lw=LWB, zorder=3)
            top.errorbar(XF[i], base + tg + ad, yerr=sd, fmt="none", ecolor="black", lw=0.6, capsize=1.5)
        for k in ("tfidf", "bert", "maj"): top.axhline(fl[k], color="black", lw=0.7, ls=LS[k], zorder=0.5)
        top.set_ylim(0, 1); top.set_xlim(*XL); top.set_title(BL[b], fontsize=9, pad=3); top.set_xticks(xt)
        if curto: top.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if j == 0 else [])
        if j == 0: top.set_ylabel("test accuracy")
        else: top.set_yticklabels([])
        last = top
        last.set_xticks(xt); last.set_xticklabels(xl, rotation=0 if curto else 90, fontsize=9)
    hs = [Patch(facecolor=C["txt"], edgecolor=EC, lw=LWB), Patch(facecolor=C["enc"], edgecolor=EC, lw=LWB), Patch(facecolor=C["ft"], edgecolor=EC, lw=LWB)] if contorno else [Patch(color=C["txt"]), Patch(color=C["enc"]), Patch(color=C["ft"])]
    hs += [(Line2D([], [], marker="o", mfc="white", mec=C["zs"], ls="", ms=3.5), Line2D([], [], marker="o", color=C["zs"], ls="", ms=3.5)),
           Patch(facecolor=C["zsg"], edgecolor="#4a6f8f", hatch="////"), Line2D([], [], color="black", lw=0.7, ls=LS["tfidf"]), Line2D([], [], color="black", lw=0.7, ls=LS["bert"]), Line2D([], [], color="black", lw=0.7, ls=LS["maj"])]
    ls_ = ["text-decoder-only", "gain from the frozen visual encoder", "gain from fine-tuning the visual encoder", "zero-shot: no image → with image", "zero-shot image gain", "TF-IDF classifier", "PubMedBERT classifier", "majority answer"]
    if not full: hs, ls_ = hs[:4] + hs[5:], ls_[:4] + ls_[5:]
    if not com_zs: hs, ls_ = [h for h, l in zip(hs, ls_) if not l.startswith("zero-shot")], [l for l in ls_ if not l.startswith("zero-shot")]
    lg = fig.legend(hs, ls_, loc="outside lower center", ncol=ncol, frameon=False, fontsize=9, handlelength=1.6, handletextpad=0.4, columnspacing=0.9, borderaxespad=0.15, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
    return fig, axes, lg
def _overlaps(fig, lg):
    fig.canvas.draw(); R = fig.canvas.get_renderer(); W, Hh = fig.bbox.width, fig.bbox.height; bad = []
    T = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    bb = [(t, t.get_window_extent(R)) for t in T]
    for t, b_ in bb:
        if b_.x0 < -0.5 or b_.y0 < -0.5 or b_.x1 > W + 0.5 or b_.y1 > Hh + 0.5: bad.append(("fora", t.get_text()))
    for i in range(len(bb)):
        for k in range(i + 1, len(bb)):
            if bb[i][1].overlaps(bb[k][1]) and bb[i][1].intersection(bb[i][1], bb[k][1]).width > 0.5 and bb[i][1].intersection(bb[i][1], bb[k][1]).height > 0.5: bad.append((bb[i][0].get_text(), bb[k][0].get_text()))
    lb = lg.get_window_extent(R)
    for ax in fig.axes:
        tb = ax.get_tightbbox(R)
        if lb.overlaps(tb) and lb.intersection(lb, tb).height > 0.5: bad.append(("legenda x eixo", ax.get_title()))
    return bad
nome = a.name; escolhida = None; _rc = plt.rc_context(RC_V5); _rc.__enter__()
for H in [2.5]:
    for NC in (3,):
        fig, axes, lg = _draw(False, H, NC, False, True, curto=True, contorno=True, m3ae_frozen=True, i17=FROZEN, lw_c=0.25); fig.canvas.draw(); R = fig.canvas.get_renderer()
        if lg.get_window_extent(R).width <= fig.bbox.width - 2: break
        raise AssertionError(("legend with 3 columns does not fit in 7.2 in", H))
    fig.canvas.draw(); bad = _overlaps(fig, lg)
    if bad: plt.close(fig); break
    if escolhida is not None: plt.close(escolhida[1])
    escolhida = (H, fig); NC_ok = NC
assert escolhida is not None, (nome, bad)
H, fig = escolhida
for e in ("pdf", "png"): fig.savefig(f"{OUT}/{nome}.{e}", dpi=300, metadata={"CreationDate": None} if e == "pdf" else None)
assert not _overlaps(fig, fig.legends[0]), _overlaps(fig, fig.legends[0])
plt.close(fig); _rc.__exit__(None, None, None)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(json.dumps({f"{nome}.{e}": sha(f"{OUT}/{nome}.{e}") for e in ("pdf", "png")}, indent=1))
