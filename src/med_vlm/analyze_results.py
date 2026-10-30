"""Reads all experiment result JSON files from an outputs directory and produces:

  - <out_dir>/results.csv          — flat table with all metrics
  - <out_dir>/results.tex          — LaTeX booktabs table (ready to paste)
  - <out_dir>/metrics_bar.png      — 2×3 grid of per-metric bar charts
  - <out_dir>/efficiency_bar.png   — bar charts of computational efficiency metrics

Usage:
  poetry run python -m med_vlm.analyze_results \
      --outputs-dir ./outputs \
      --out-dir ./analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── Friendly display names ────────────────────────────────────────────────────
_MODEL_LABELS: dict[str, str] = {
    "Salesforce/blip2-opt-2.7b": "BLIP-2",
    "ibm-granite/granite-vision-3.2-2b": "Granite Vision 3.2-2B",
    "Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
    "HuggingFaceM4/SmolVLM-2.2B-Instruct": "SmolVLM-2.2B",
    "microsoft/llava-med-v1.5-mistral-7b": "LLaVA-Med-7B",
    "JZPeterPan/MedVLM-R1": "MedVLM-R1",
}

_PEFT_LABELS: dict[str, str] = {
    "lora": "LoRA",
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "none": "—",
}

_METRICS = ["accuracy", "bleu", "rouge1", "rouge2", "rougeL", "meteor", "bert_score_f1"]
_METRIC_LABELS = [
    "Accuracy",
    "BLEU",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "METEOR",
    "BERTScore-F1",
]

_EFFICIENCY_COLS = [
    "trainable_params",
    "train_time_s",
    "infer_throughput_sps",
    "infer_peak_mem_gb",
]
_EFFICIENCY_LABELS = [
    "Trainable Params",
    "Train Time (s)",
    "Infer Throughput (sps)",
    "Infer Peak Mem (GB)",
]


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _load_results(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(outputs_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue  # skip list-type files (e.g. qualitative_examples.json)

        model_id = data.get("model", "")
        peft = data.get("peft", "none")
        metrics = data.get("metrics", {})
        efficiency = data.get("efficiency", {})

        row: dict = {
            "run": path.stem,
            "model": _MODEL_LABELS.get(model_id, model_id),
            "peft": _PEFT_LABELS.get(peft, peft),
            "clip_encoder": data.get("clip_encoder") or "",
        }
        for m in _METRICS:
            row[m] = metrics.get(m)

        # efficiency sub-fields
        params = efficiency.get("params") or {}
        training_eff = efficiency.get("training") or {}
        inference_eff = efficiency.get("inference") or {}
        row["trainable_params"] = params.get("trainable_params")
        row["train_time_s"] = training_eff.get("train_wall_time_s")
        row["infer_throughput_sps"] = inference_eff.get("infer_throughput_sps")
        row["infer_peak_mem_gb"] = inference_eff.get("infer_peak_memory_gb")

        rows.append(row)

    if not rows:
        raise SystemExit(f"No JSON files found in {outputs_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values("run").reset_index(drop=True)
    return df


def _save_csv(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "results.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def _save_latex(df: pd.DataFrame, out_dir: Path) -> Path:
    """Emit a booktabs LaTeX table with quality metrics and key efficiency columns."""
    col_map = {m: lbl for m, lbl in zip(_METRICS, _METRIC_LABELS)}
    eff_col_map = {c: lbl for c, lbl in zip(_EFFICIENCY_COLS, _EFFICIENCY_LABELS)}
    display = df.rename(
        columns={"model": "Model", "peft": "PEFT", **col_map, **eff_col_map}
    )
    drop_cols = [c for c in ["run", "clip_encoder"] if c in display.columns]
    display = display.drop(columns=drop_cols)

    # Format floats as percentages × 100 with 1 decimal place.
    for lbl in _METRIC_LABELS:
        if lbl in display.columns:
            display[lbl] = display[lbl].apply(
                lambda v: f"{v * 100:.1f}" if pd.notna(v) else "—"
            )

    all_metric_labels = _METRIC_LABELS + _EFFICIENCY_LABELS
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{VQA results on the combined test split "
        r"(SLAKE + VQA-MED + Path-VQA + VQA-RAD). "
        r"Quality metrics are reported as percentages; "
        r"efficiency metrics are reported in their raw units.}",
        r"\label{tab:vqa-results}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{ll" + "r" * len(all_metric_labels) + r"}",
        r"\toprule",
    ]

    header_cols = ["Model", "PEFT"] + all_metric_labels
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")

    for _, row in display.iterrows():
        lines.append(
            " & ".join(
                str(row[c]) if c in display.columns else "—" for c in header_cols
            )
            + r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = out_dir / "results.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── Short display labels for plot axes ───────────────────────────────────────


def _short_label(run: str) -> str:
    """Convert a run name like 'granite_vision_adalora_pubmedclip_combined'
    into a compact two-line label suitable for a plot axis."""
    mapping = {
        "blip_lora_combined": "BLIP-2\n+LoRA",
        "blip_lora_slake_quick": "BLIP-2\n+LoRA (SLAKE)",
        "granite_vision_lora_combined": "Granite\n+LoRA",
        "granite_vision_adalora_combined": "Granite\n+AdaLoRA",
        "granite_vision_dora_combined": "Granite\n+DoRA",
        "granite_vision_lora_pubmedclip_combined": "Granite\n+LoRA\n+PM-CLIP",
        "granite_vision_lora_medclip_combined": "Granite\n+LoRA\n+MedCLIP",
        "granite_vision_dora_pubmedclip_combined": "Granite\n+DoRA\n+PM-CLIP",
        "granite_vision_dora_medclip_combined": "Granite\n+DoRA\n+MedCLIP",
        "granite_vision_adalora_pubmedclip_combined": "Granite\n+AdaLoRA\n+PM-CLIP",
        "medvlm_zeroshot_combined": "MedVLM-R1\n(0-shot)",
        "qwen2vl_lora_combined": "Qwen2-VL\n+LoRA",
        "qwen2vl_dora_combined": "Qwen2-VL\n+DoRA",
        "qwen2vl_adalora_combined": "Qwen2-VL\n+AdaLoRA",
        "smolvlm2_lora_combined": "SmolVLM2\n+LoRA",
        "smolvlm2_dora_combined": "SmolVLM2\n+DoRA",
        "smolvlm2_adalora_combined": "SmolVLM2\n+AdaLoRA",
    }
    return mapping.get(run, run.replace("_combined", "").replace("_", "\n"))


def _family_key(run: str) -> str:
    """Return a short string identifying the model family of a run."""
    for prefix in ("blip", "granite", "qwen", "smolvlm", "medvlm"):
        if run.startswith(prefix):
            return prefix
    return run


def _family_color(run: str) -> str:
    """Return a consistent hex colour for a model family."""
    if run.startswith("blip"):
        return "#4C72B0"
    if run.startswith("granite"):
        return "#55A868"
    if run.startswith("qwen"):
        return "#C44E52"
    if run.startswith("smolvlm"):
        return "#8172B2"
    if run.startswith("medvlm"):
        return "#937860"
    return "#64B5CD"


# ── Publication-quality chart helpers ────────────────────────────────────────


def _apply_ieee_style(ax) -> None:
    """Remove chart junk, apply light grid — consistent IEEE-adjacent style."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5, color="#aaaaaa")
    ax.set_axisbelow(True)


def _save_chart(df: pd.DataFrame, out_dir: Path) -> Path:
    """Horizontal grouped bar chart for the four most informative quality metrics.

    Produces both a PNG (150 dpi preview) and a PDF (vector, paper-ready).
    """
    # Only keep runs that have at least accuracy data; skip SLAKE-only quick runs
    plot_df = df[df["accuracy"].notna()].copy()
    # Exclude the quick SLAKE run and BLIP-2
    plot_df = plot_df[~plot_df["run"].str.contains("slake_quick", na=False)]
    plot_df = plot_df[~plot_df["run"].str.startswith("blip")]

    labels = [_short_label(r) for r in plot_df["run"]]
    colors = [_family_color(r) for r in plot_df["run"]]
    n = len(labels)

    metrics_to_plot = ["accuracy", "bert_score_f1", "rougeL", "meteor"]
    metric_titles = ["Accuracy", "BERTScore-F1", "ROUGE-L", "METEOR"]

    fig, axes = plt.subplots(
        1, 4, figsize=(14, max(3.5, 0.45 * n)), constrained_layout=True
    )
    y = list(range(n))

    for ax, metric, title in zip(axes, metrics_to_plot, metric_titles):
        values = [float(v) if pd.notna(v) else 0.0 for v in plot_df[metric].tolist()]
        bars = ax.barh(
            y, values, color=colors, edgecolor="white", linewidth=0.4, height=0.7
        )
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.tick_params(axis="x", labelsize=8)
        _apply_ieee_style(ax)

        for bar, val in zip(bars, values):
            if val > 0.02:
                ax.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}",
                    va="center",
                    ha="left",
                    fontsize=7,
                )

        # Invert y so highest bar is on top
        ax.invert_yaxis()

    # Family legend
    from matplotlib.patches import Patch

    legend_entries = [
        Patch(facecolor="#4C72B0", label="BLIP-2"),
        Patch(facecolor="#55A868", label="Granite Vision"),
        Patch(facecolor="#C44E52", label="Qwen2-VL"),
        Patch(facecolor="#8172B2", label="SmolVLM2"),
        Patch(facecolor="#937860", label="MedVLM-R1"),
    ]
    fig.legend(
        handles=legend_entries,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    for fmt, suffix in [("png", ".png"), ("pdf", ".pdf")]:
        path = out_dir / f"metrics_bar{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return out_dir / "metrics_bar.pdf"


def _save_efficiency_chart(df: pd.DataFrame, out_dir: Path) -> Path:
    """Accuracy vs. training time scatter (bubble size = peak GPU memory).

    Produces both a PNG preview and a PDF for the paper.
    """
    plot_df = df[df["accuracy"].notna() & df["train_time_s"].notna()].copy()
    plot_df = plot_df[~plot_df["run"].str.contains("slake_quick", na=False)]
    plot_df = plot_df[~plot_df["run"].str.startswith("blip")]

    # Keep only the best-accuracy run per model family
    plot_df["_fam"] = plot_df["run"].apply(_family_key)
    best_idx = plot_df.groupby("_fam")["accuracy"].idxmax()
    plot_df = plot_df.loc[best_idx].drop(columns=["_fam"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    max_mem = plot_df["infer_peak_mem_gb"].max()

    # Per-point label nudges to avoid overlap
    _LABEL_OFFSETS: dict[str, tuple[float, float]] = {
        "smolvlm2_dora_combined": (-5, 8),
        "qwen2vl_lora_combined": (5, -12),
        "granite_vision_adalora_combined": (5, 5),
    }

    for _, row in plot_df.iterrows():
        acc = float(row["accuracy"])
        t_h = float(row["train_time_s"]) / 3600.0
        mem = (
            float(row["infer_peak_mem_gb"])
            if pd.notna(row["infer_peak_mem_gb"])
            else 5.0
        )
        size = 30 + 120 * (mem / max_mem)
        col = _family_color(str(row["run"]))
        ax.scatter(
            t_h, acc, s=size, color=col, edgecolors="white", linewidths=0.6, zorder=3
        )
        lbl = _short_label(str(row["run"])).replace("\n", " ")
        xytext = _LABEL_OFFSETS.get(str(row["run"]), (5, 5))
        ax.annotate(
            lbl,
            (t_h, acc),
            textcoords="offset points",
            xytext=xytext,
            fontsize=7,
            color="#333333",
            arrowprops=None,
        )

    ax.set_xlabel("Training time (h)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.70)
    ax.tick_params(labelsize=8)
    _apply_ieee_style(ax)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.5, color="#aaaaaa")

    # Size legend: small proportional bubbles (not full-scale) to avoid clutter
    for mem_val, legend_s in [(4, 25), (8, 50), (15, 90)]:
        ax.scatter(
            [],
            [],
            s=legend_s,
            color="#aaaaaa",
            edgecolors="white",
            linewidths=0.6,
            label=f"{mem_val} GB",
        )
    ax.legend(
        title="Peak mem.",
        title_fontsize=7,
        fontsize=7,
        frameon=True,
        framealpha=0.85,
        edgecolor="#cccccc",
        loc="lower right",
        labelspacing=0.5,
        handletextpad=0.6,
        borderpad=0.6,
    )

    for fmt, suffix in [("png", ".png"), ("pdf", ".pdf")]:
        path = out_dir / f"efficiency_bar{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return out_dir / "efficiency_bar.pdf"


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment result JSONs.")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="./outputs",
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./analysis",
        help="Directory to write CSV, LaTeX, and chart outputs.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_results(outputs_dir)

    csv_path = _save_csv(df, out_dir)
    tex_path = _save_latex(df, out_dir)
    _save_chart(df, out_dir)  # returns pdf, png path printed from out_dir
    _save_efficiency_chart(df, out_dir)  # same

    print(f"Loaded {len(df)} result(s) from {outputs_dir}/")
    print(f"  CSV:                 {csv_path}")
    print(f"  LaTeX:               {tex_path}")
    print(f"  Quality chart (PDF): {out_dir / 'metrics_bar.pdf'}")
    print(f"  Quality chart (PNG): {out_dir / 'metrics_bar.png'}")
    print(f"  Efficiency scatter (PDF): {out_dir / 'efficiency_bar.pdf'}")
    print(f"  Efficiency scatter (PNG): {out_dir / 'efficiency_bar.png'}")
    print()
    display_cols = ["run", "model", "peft"] + _METRICS + _EFFICIENCY_COLS
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
