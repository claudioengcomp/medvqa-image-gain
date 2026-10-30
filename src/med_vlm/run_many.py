import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import yaml

from med_vlm.run_experiment import run_experiment


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    _metric_keys = ["accuracy", "bleu", "rouge1", "rouge2", "rougeL", "meteor"]

    flat_rows = []
    for r in rows:
        m = r.get("metrics", {})
        flat_rows.append(
            {
                "result_path": r.get("result_path"),
                "model": r.get("model"),
                "dataset": r.get("dataset"),
                "peft": r.get("peft"),
                **{k: m.get(k) for k in _metric_keys},
            }
        )

    fieldnames = list(flat_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="List of YAML config paths.")
    parser.add_argument("--config-dir", type=str, help="Directory with YAML configs.")
    parser.add_argument("--summary-csv", type=str, default="./outputs/summary.csv")
    args = parser.parse_args()

    paths: List[Path] = []
    if args.configs:
        paths.extend([Path(p) for p in args.configs])

    if args.config_dir:
        d = Path(args.config_dir)
        paths.extend(sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")))

    if not paths:
        raise SystemExit("Provide --configs and/or --config-dir")

    results = []
    for p in paths:
        cfg = load_yaml(p)
        print(f"\n=== Running: {p} ===")
        out = run_experiment(cfg)
        results.append(out)

    write_summary_csv(results, Path(args.summary_csv))
    print(f"\nWrote summary CSV to: {args.summary_csv}")


if __name__ == "__main__":
    main()

# TO RUN:
# CUDA_VISIBLE_DEVICES=3 poetry run python -m med_vlm.run_many --config-dir configs
