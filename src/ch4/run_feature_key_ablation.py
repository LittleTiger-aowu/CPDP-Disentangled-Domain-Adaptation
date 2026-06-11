"""
Feature-key ablation for Chapter 4: compare H_file vs Z_sh under identical settings.

Runs per target (each with its own dump dir):
  1) source-only @ H_file (lambda_dom=0)
  2) DANN @ H_file (lambda_dom>0)
  3) DANN @ Z_sh (lambda_dom>0)
  4) source-only @ Z_sh (lambda_dom=0)  [optional but included]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs/ch4_runs/feature_key_ablation")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dom", type=float, default=0.5)
    ap.add_argument("--grl-gamma", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-bottleneck", type=int, default=1)
    ap.add_argument("--bottleneck-dim", type=int, default=128)
    ap.add_argument("--am-margin", type=float, default=0.3)
    ap.add_argument("--temp-scale", type=int, default=1)
    ap.add_argument("--temp-max-iter", type=int, default=200)
    ap.add_argument("--threshold-mode", default="fbeta", choices=["mcc", "f1", "fbeta"])
    ap.add_argument("--f-beta", type=float, default=2.0)
    ap.add_argument("--use-percentile", type=int, default=1)
    ap.add_argument("--drift-thr", type=float, default=0.05)
    args = ap.parse_args()

    projects = [
        {
            "target": "Mylyn-3.1",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Mylyn-3.1\dump",
            "sources": "Eclipse_JDT_Core-3.4,Equinox-3.4,PDE_UI-3.4.1,lucene-2.4",
        },
        {
            "target": "Equinox-3.4",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Equinox-3.4\dump",
            "sources": "Eclipse_JDT_Core-3.4,Mylyn-3.1,PDE_UI-3.4.1,lucene-2.4",
        },
        {
            "target": "Eclipse_JDT_Core-3.4",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Eclipse_JDT_Core-3.4\dump",
            "sources": "Equinox-3.4,Mylyn-3.1,PDE_UI-3.4.1,lucene-2.4",
        },
        {
            "target": "PDE_UI-3.4.1",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_PDE_UI-3.4.1\dump",
            "sources": "Eclipse_JDT_Core-3.4,Equinox-3.4,Mylyn-3.1,lucene-2.4",
        },
        {
            "target": "lucene-2.4",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_lucene-2.4\dump",
            "sources": "Eclipse_JDT_Core-3.4,Equinox-3.4,Mylyn-3.1,PDE_UI-3.4.1",
        },
    ]

    variants = [
        ("source_only_hfile", "H_file", 0.0),
        ("dann_hfile", "H_file", args.lambda_dom),
        ("dann_zsh", "Z_sh", args.lambda_dom),
        ("source_only_zsh", "Z_sh", 0.0),
    ]

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    rows = []
    for p in projects:
        for tag, feat_key, lam in variants:
            out_dir = out_root / f"target_{p['target']}" / tag
            cmd = [
                "python",
                str(train_py),
                "--dump-dir",
                p["dump"],
                "--source-projects",
                p["sources"],
                "--target-project",
                p["target"],
                "--output-dir",
                str(out_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--lambda-dom",
                str(lam),
                "--grl-gamma",
                str(args.grl_gamma),
                "--seed",
                str(args.seed),
                "--feature-key",
                feat_key,
                "--use-bottleneck",
                str(args.use_bottleneck),
                "--bottleneck-dim",
                str(args.bottleneck_dim),
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
                flat = flatten(m)
                flat["target_project"] = p["target"]
                flat["variant"] = tag
                rows.append(flat)

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        out_csv = out_root / "summary_feature_key_ablation.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved summary to {out_csv}")
    else:
        print("No metrics found to summarize.")


if __name__ == "__main__":
    main()
