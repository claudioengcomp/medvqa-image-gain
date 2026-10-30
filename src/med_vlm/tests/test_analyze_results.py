import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from med_vlm.analyze_results import (  # noqa: E402
    _load_results,
    _save_chart,
    _save_csv,
    _save_latex,
    main,
)

_SAMPLE_DATA = {
    "model": "Salesforce/blip2-opt-2.7b",
    "peft": "lora",
    "metrics": {
        "accuracy": 0.75,
        "bleu": 0.50,
        "rouge1": 0.60,
        "rouge2": 0.40,
        "rougeL": 0.55,
        "meteor": 0.65,
    },
}


@pytest.fixture
def results_dir(tmp_path):
    (tmp_path / "exp_blip2.json").write_text(json.dumps(_SAMPLE_DATA))
    return tmp_path


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "run": "exp_blip2",
                "model": "BLIP-2",
                "peft": "LoRA",
                "accuracy": 0.75,
                "bleu": 0.50,
                "rouge1": 0.60,
                "rouge2": 0.40,
                "rougeL": 0.55,
                "meteor": 0.65,
                "bert_score_f1": 0.70,
            }
        ]
    )


# ── _load_results ─────────────────────────────────────────────────────────────


def test_load_results_returns_dataframe(results_dir):
    df = _load_results(results_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["model"].iloc[0] == "BLIP-2"
    assert df["peft"].iloc[0] == "LoRA"
    assert df["accuracy"].iloc[0] == pytest.approx(0.75)


def test_load_results_unknown_model_kept_as_is(tmp_path):
    data = {"model": "unknown/model-xyz", "peft": "none", "metrics": {"accuracy": 0.5}}
    (tmp_path / "run.json").write_text(json.dumps(data))
    df = _load_results(tmp_path)
    assert df["model"].iloc[0] == "unknown/model-xyz"


def test_load_results_missing_metrics_key(tmp_path):
    data = {"model": "Salesforce/blip2-opt-2.7b", "peft": "none"}
    (tmp_path / "run.json").write_text(json.dumps(data))
    df = _load_results(tmp_path)
    assert pd.isna(df["accuracy"].iloc[0])


def test_load_results_raises_when_no_json(tmp_path):
    with pytest.raises(SystemExit):
        _load_results(tmp_path)


def test_load_results_multiple_files(tmp_path):
    for i in range(3):
        d = {**_SAMPLE_DATA, "model": "Salesforce/blip2-opt-2.7b"}
        (tmp_path / f"run_{i}.json").write_text(json.dumps(d))
    df = _load_results(tmp_path)
    assert len(df) == 3


# ── _save_csv ─────────────────────────────────────────────────────────────────


def test_save_csv_creates_file(sample_df, tmp_path):
    path = _save_csv(sample_df, tmp_path)
    assert path.exists()
    assert path.name == "results.csv"


def test_save_csv_content(sample_df, tmp_path):
    path = _save_csv(sample_df, tmp_path)
    loaded = pd.read_csv(path)
    assert len(loaded) == 1
    assert "accuracy" in loaded.columns
    assert loaded["accuracy"].iloc[0] == pytest.approx(0.75, abs=1e-3)


# ── _save_latex ───────────────────────────────────────────────────────────────


def test_save_latex_creates_file(sample_df, tmp_path):
    path = _save_latex(sample_df, tmp_path)
    assert path.exists()
    assert path.name == "results.tex"


def test_save_latex_content(sample_df, tmp_path):
    path = _save_latex(sample_df, tmp_path)
    content = path.read_text()
    assert r"\begin{table}" in content
    assert r"\toprule" in content
    assert r"\bottomrule" in content
    assert "BLIP-2" in content
    assert "LoRA" in content


def test_save_latex_nan_values(tmp_path):
    df = pd.DataFrame(
        [
            {
                "run": "r",
                "model": "M",
                "peft": "—",
                "accuracy": None,
                "bleu": None,
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "meteor": None,
                "bert_score_f1": None,
            }
        ]
    )
    path = _save_latex(df, tmp_path)
    content = path.read_text()
    assert "—" in content


# ── _save_chart ───────────────────────────────────────────────────────────────


def test_save_chart_creates_png(sample_df, tmp_path):
    path = _save_chart(sample_df, tmp_path)
    assert path.exists()
    # _save_chart produces both PNG and PDF; PDF is the returned/primary format
    assert (tmp_path / "metrics_bar.png").exists()


def test_save_chart_multiple_runs(tmp_path):
    df = pd.DataFrame(
        [
            {
                "run": f"run_{i}",
                "model": f"M{i}",
                "peft": "—",
                "accuracy": 0.5 + i * 0.1,
                "bleu": 0.3,
                "rouge1": 0.4,
                "rouge2": 0.2,
                "rougeL": 0.35,
                "meteor": 0.45,
                "bert_score_f1": 0.5 + i * 0.1,
            }
            for i in range(4)
        ]
    )
    path = _save_chart(df, tmp_path)
    assert path.exists()


# ── main ──────────────────────────────────────────────────────────────────────


def test_main_produces_all_outputs(results_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_results",
            "--outputs-dir",
            str(results_dir),
            "--out-dir",
            str(tmp_path),
        ],
    )
    main()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.tex").exists()
    assert (tmp_path / "metrics_bar.png").exists()


def test_main_creates_out_dir_if_missing(results_dir, tmp_path, monkeypatch):
    out_dir = tmp_path / "nested" / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_results",
            "--outputs-dir",
            str(results_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    main()
    assert out_dir.exists()
    assert (out_dir / "results.csv").exists()
