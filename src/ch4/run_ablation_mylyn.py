"""
Ablation runner for Mylyn-3.1 (or any target) using train_cpdp_adapt_cached.py.

Runs:
  - full (adversarial + bottleneck + margin)
  - w/o adversarial (lambda_dom=0)
  - w/o bottleneck (use_bottleneck=0)
  - w/o margin (am_margin=0)
  - input feature ablation (H_file vs Z_sh)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--source-projects",
        required=True,
        help="comma-separated source projects",
    )
    ap.add_argument("--target-project", default="Mylyn-3.1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dom", type=float, default=0.5)
    ap.add_argument("--grl-gamma", type=float, default=10.0)
    ap.add_argument("--use-bottleneck", type=int, default=1)
    ap.add_argument("--bottleneck-dim", type=int, default=128)
    ap.add_argument("--bottleneck-hidden", type=int, default=0)
    ap.add_argument("--bottleneck-dropout", type=float, default=0.0)
    ap.add_argument("--am-margin", type=float, default=0.3)
    ap.add_argument("--temp-scale", type=int, default=1)
    ap.add_argument("--temp-max-iter", type=int, default=200)
    ap.add_argument("--threshold-mode", default="mcc", choices=["mcc", "f1", "fbeta"])
    ap.add_argument("--f-beta", type=float, default=2.0)
    ap.add_argument("--use-percentile", type=int, default=1)
    ap.add_argument("--drift-thr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-intra", type=int, default=0)
    ap.add_argument("--feature-key", default="H_file")
    args = ap.parse_args()

    train_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base = {
        "dump_dir": args.dump_dir,
        "source_projects": args.source_projects,
        "target_project": args.target_project,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_dom": args.lambda_dom,
        "grl_gamma": args.grl_gamma,
        "use_bottleneck": args.use_bottleneck,
        "bottleneck_dim": args.bottleneck_dim,
        "bottleneck_hidden": args.bottleneck_hidden,
        "bottleneck_dropout": args.bottleneck_dropout,
        "am_margin": args.am_margin,
        "temp_scale": args.temp_scale,
        "temp_max_iter": args.temp_max_iter,
        "threshold_mode": args.threshold_mode,
        "f_beta": args.f_beta,
        "use_percentile": args.use_percentile,
        "drift_thr": args.drift_thr,
        "seed": args.seed,
        "dedup_intra": args.dedup_intra,
        "feature_key": args.feature_key,
    }

    runs = [
        ("full", base),
        ("wo_adversarial", {**base, "lambda_dom": 0.0}),
        ("wo_bottleneck", {**base, "use_bottleneck": 0}),
        ("wo_margin", {**base, "am_margin": 0.0}),
        ("feat_H_file", {**base, "feature_key": "H_file"}),
        ("feat_Z_sh", {**base, "feature_key": "Z_sh"}),
    ]

    summary_rows = []

    for tag, cfg in runs:
        out_dir = out_root / f"{tag}_target_{cfg['target_project']}"
        cmd = [
            "python",
            str(train_py),
            "--dump-dir",
            cfg["dump_dir"],
            "--source-projects",
            cfg["source_projects"],
            "--target-project",
            cfg["target_project"],
            "--output-dir",
            str(out_dir),
            "--epochs",
            str(cfg["epochs"]),
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            str(cfg["lr"]),
            "--lambda-dom",
            str(cfg["lambda_dom"]),
            "--grl-gamma",
            str(cfg["grl_gamma"]),
            "--seed",
            str(cfg["seed"]),
            "--dedup-intra",
            str(cfg["dedup_intra"]),
            "--feature-key",
            str(cfg["feature_key"]),
            "--use-bottleneck",
            str(cfg["use_bottleneck"]),
            "--bottleneck-dim",
            str(cfg["bottleneck_dim"]),
            "--bottleneck-hidden",
            str(cfg["bottleneck_hidden"]),
            "--bottleneck-dropout",
            str(cfg["bottleneck_dropout"]),
            "--am-margin",
            str(cfg["am_margin"]),
            "--temp-scale",
            str(cfg["temp_scale"]),
            "--temp-max-iter",
            str(cfg["temp_max_iter"]),
            "--threshold-mode",
            str(cfg["threshold_mode"]),
            "--f-beta",
            str(cfg["f_beta"]),
            "--use-percentile",
            str(cfg["use_percentile"]),
            "--drift-thr",
            str(cfg["drift_thr"]),
        ]
        run_cmd(cmd)
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "run": tag,
                "target_project": m.get("target_project"),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "f1_default": m.get("default_0p5", {}).get("f1"),
                "mcc_default": m.get("default_0p5", {}).get("mcc"),
                "recall_default": m.get("default_0p5", {}).get("recall"),
                "precision_default": m.get("default_0p5", {}).get("precision"),
                "pf_default": m.get("default_0p5", {}).get("pf"),
                "threshold_transfer": m.get("transfer_source", {}).get("threshold"),
                "f1_transfer": m.get("transfer_source", {}).get("f1"),
                "mcc_transfer": m.get("transfer_source", {}).get("mcc"),
                "recall_transfer": m.get("transfer_source", {}).get("recall"),
                "precision_transfer": m.get("transfer_source", {}).get("precision"),
                "pf_transfer": m.get("transfer_source", {}).get("pf"),
                "oracle_best_threshold": m.get("oracle_target", {}).get("best_threshold"),
                "oracle_best_f1": m.get("oracle_target", {}).get("best_f1"),
                "oracle_best_mcc": m.get("oracle_target", {}).get("best_mcc"),
                "final_domain_acc": m.get("final_domain_acc"),
                "feature_key": m.get("feature_key"),
                "use_bottleneck": m.get("use_bottleneck"),
                "am_margin": m.get("am_margin"),
                "lambda_dom": m.get("lambda_dom"),
            }
            summary_rows.append(row)

    if summary_rows:
        summary_path = out_root / f"ablation_summary_{args.target_project}.csv"
        fieldnames = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
