"""
Collect ablation_results.csv from multiple target subfolders into one summary table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output-root used by run_ablation_study_val_threshold_all.py")
    ap.add_argument("--out", default=None, help="output summary csv path")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    rows = []
    for csv_path in root.rglob("ablation_results.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df["source_csv"] = str(csv_path)
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No ablation_results.csv found under root.")

    out_df = pd.concat(rows, ignore_index=True)
    out_path = Path(args.out) if args.out else root / "ablation_results_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
