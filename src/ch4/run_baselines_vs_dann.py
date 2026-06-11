"""
Run traditional baselines (TCA/SA/CORAL) and DANN baseline, then summarize to CSV.
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
    ap.add_argument("--source-projects", required=True)
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--feature-key", default="H_file")
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dom", type=float, default=0.5)
    ap.add_argument("--grl-gamma", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-intra", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_py = Path(__file__).parent / "baselines" / "baseline_runner.py"
    dann_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    summary_rows = []

    # Traditional baselines
    for method in ["tca", "sa", "coral"]:
        out_json = out_root / f"{method}_metrics.json"
        cmd = [
            "python",
            str(baseline_py),
            "--dump-dir",
            args.dump_dir,
            "--source-projects",
            args.source_projects,
            "--target-project",
            args.target_project,
            "--feature-key",
            args.feature_key,
            "--method",
            method,
            "--n-components",
            str(args.n_components),
            "--kernel",
            str(args.kernel),
            "--gamma",
            str(args.gamma),
            "--mu",
            str(args.mu),
            "--seed",
            str(args.seed),
            "--dedup-intra",
            str(args.dedup_intra),
            "--output-json",
            str(out_json),
        ]
        run_cmd(cmd)
        m = json.loads(out_json.read_text(encoding="utf-8"))
        transfer = m.get("transfer_source", {})
        summary_rows.append(
            {
                "method": method,
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "f1": transfer.get("f1"),
                "mcc": transfer.get("mcc"),
                "pf": transfer.get("pf"),
                "recall": transfer.get("recall"),
                "precision": transfer.get("precision"),
                "acc": transfer.get("acc"),
                "threshold": transfer.get("threshold"),
                "feature_key": m.get("feature_key"),
                "domain_acc": None,
            }
        )

    # DANN baseline (no bottleneck, no margin)
    dann_out = out_root / "dann_base"
    cmd = [
        "python",
        str(dann_py),
        "--dump-dir",
        args.dump_dir,
        "--source-projects",
        args.source_projects,
        "--target-project",
        args.target_project,
        "--output-dir",
        str(dann_out),
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
        "--feature-key",
        args.feature_key,
        "--use-bottleneck",
        "0",
        "--am-margin",
        "0",
        "--seed",
        str(args.seed),
        "--dedup-intra",
        str(args.dedup_intra),
    ]
    run_cmd(cmd)

    dann_metrics = json.loads((dann_out / "metrics.json").read_text(encoding="utf-8"))
    summary_rows.append(
        {
            "method": "dann_base",
            "roc_auc": dann_metrics.get("roc_auc"),
            "pr_auc": dann_metrics.get("pr_auc"),
            "f1": dann_metrics.get("transfer_source", {}).get("f1"),
            "mcc": dann_metrics.get("transfer_source", {}).get("mcc"),
            "pf": dann_metrics.get("transfer_source", {}).get("pf"),
            "recall": dann_metrics.get("transfer_source", {}).get("recall"),
            "precision": dann_metrics.get("transfer_source", {}).get("precision"),
            "acc": None,
            "threshold": dann_metrics.get("transfer_source", {}).get("threshold"),
            "feature_key": dann_metrics.get("feature_key"),
            "domain_acc": dann_metrics.get("final_domain_acc"),
        }
    )

    summary_path = out_root / f"baseline_vs_dann_{args.target_project}.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
