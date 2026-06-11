"""
Leave-one-project-out CPDP runner using cached H_file (Method C).

For each project as target, trains adaptation on remaining projects as source,
using src/ch4/train_cpdp_adapt_cached.py, then collects metrics into a summary CSV.

Example:
  python src/ch4/run_loocv_cpdp.py ^
    --dump-dir E:\project\WYP\LineDefStudy2.0\outputs\ch3_dump\best_w3_t256 ^
    --output-root outputs/ch4_runs/loocv ^
    --epochs 5 --batch-size 64 --lr 0.001 --lambda-dom 0.5
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import csv
import numpy as np


def infer_projects(meta_path: Path):
    projects = set()
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            projects.add(obj["project"])
    return sorted(projects)


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="dir containing repr.npz and meta.jsonl")
    ap.add_argument("--output-root", required=True, help="root dir to store per-target runs")
    ap.add_argument("--projects", default=None, help="comma-separated project list; if omitted, infer from meta.jsonl")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dom", type=float, default=0.5)
    ap.add_argument("--grl-gamma", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-intra", type=int, default=0)
    ap.add_argument("--feature-key", default="H_file")
    ap.add_argument("--use-bottleneck", type=int, default=0)
    ap.add_argument("--bottleneck-dim", type=int, default=128)
    ap.add_argument("--bottleneck-hidden", type=int, default=0)
    ap.add_argument("--bottleneck-dropout", type=float, default=0.0)
    ap.add_argument("--am-margin", type=float, default=0.0)
    ap.add_argument("--temp-scale", type=int, default=1)
    ap.add_argument("--temp-max-iter", type=int, default=200)
    ap.add_argument("--threshold-mode", default="mcc", choices=["mcc", "f1", "fbeta"])
    ap.add_argument("--f-beta", type=float, default=2.0)
    ap.add_argument("--use-percentile", type=int, default=1)
    ap.add_argument("--drift-thr", type=float, default=0.05)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    meta_path = dump_dir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    if args.projects:
        projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    else:
        projects = infer_projects(meta_path)
    if len(projects) < 2:
        raise ValueError("Need at least 2 projects for LOOCV")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    metrics_rows = []

    for tgt in projects:
        src_list = [p for p in projects if p != tgt]
        out_dir = output_root / f"target_{tgt}"
        cmd = [
            "python",
            str(train_py),
            "--dump-dir",
            str(dump_dir),
            "--source-projects",
            ",".join(src_list),
            "--target-project",
            tgt,
            "--output-dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lambda-dom",
            str(args.lambda_dom),
            "--grl-gamma",
            str(args.grl_gamma),
            "--seed",
            str(args.seed),
            "--dedup-intra",
            str(args.dedup_intra),
            "--feature-key",
            str(args.feature_key),
            "--use-bottleneck",
            str(args.use_bottleneck),
            "--bottleneck-dim",
            str(args.bottleneck_dim),
            "--bottleneck-hidden",
            str(args.bottleneck_hidden),
            "--bottleneck-dropout",
            str(args.bottleneck_dropout),
            "--am-margin",
            str(args.am_margin),
            "--temp-scale",
            str(args.temp_scale),
            "--temp-max-iter",
            str(args.temp_max_iter),
            "--threshold-mode",
            str(args.threshold_mode),
            "--f-beta",
            str(args.f_beta),
            "--use-percentile",
            str(args.use_percentile),
            "--drift-thr",
            str(args.drift_thr),
        ]
        run_cmd(cmd)

        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            m["run"] = f"target_{tgt}"
            preds_path = out_dir / "predictions.csv"
            if preds_path.exists():
                with preds_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    ys = [int(float(r["y_true"])) for r in reader if r]
                pos = sum(ys)
                neg = len(ys) - pos
                m["n_target_total"] = len(ys)
                m["n_target_pos"] = pos
                m["n_target_neg"] = neg
            metrics_rows.append(m)

    # write summary csv
    if metrics_rows:
        fieldnames = sorted({k for r in metrics_rows for k in r.keys()})
        summary_csv = output_root / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(metrics_rows)
            # mean/median rows for numeric columns
            numeric_cols = []
            for k in fieldnames:
                if k in ("run",):
                    continue
                vals = [r.get(k) for r in metrics_rows if isinstance(r.get(k), (int, float))]
                if vals:
                    numeric_cols.append(k)
            if numeric_cols:
                mean_row = {"run": "mean"}
                median_row = {"run": "median"}
                for k in numeric_cols:
                    vals = [r.get(k) for r in metrics_rows if isinstance(r.get(k), (int, float))]
                    if vals:
                        mean_row[k] = float(np.mean(vals))
                        median_row[k] = float(np.median(vals))
                w.writerow(mean_row)
                w.writerow(median_row)
        print(f"Saved LOOCV summary to {summary_csv}")
    else:
        print("No metrics found to summarize.")


if __name__ == "__main__":
    main()
