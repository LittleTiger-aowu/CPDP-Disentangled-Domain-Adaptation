"""
Run ablation study for target project Mylyn-3.1 and summarize results.

For each variant:
  1) Train (exclude target)
  2) Dump representations + metrics/probes/plots (via run_direct_transfer.py)
  3) Evaluate direct transfer
Collect metrics into outputs/ch3_ablation/ablation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


VARIANTS = [
    {
        "name": "full_method",
        "desc": "Project supervision + orthogonality",
        "flags": ["--no-ortho", "0", "--no-pr-dom", "0"],
    },
    {
        "name": "w_o_orthogonality",
        "desc": "Keep project supervision, remove orthogonality",
        "flags": ["--no-ortho", "1", "--no-pr-dom", "0"],
    },
    {
        "name": "w_o_project_supervision",
        "desc": "Remove project supervision (no L_pr); ortho disabled",
        "flags": ["--no-ortho", "1", "--no-pr-dom", "1"],
    },
    {
        "name": "w_o_gcn",
        "desc": "Remove GCN (no deep structure)",
        "flags": ["--no-gcn", "1"],
    },
]


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--project-vocab", required=True)
    ap.add_argument("--codebert-path", required=True)
    ap.add_argument("--block-cache-dir", default=None)
    ap.add_argument("--target-project", default="Mylyn-3.1")
    ap.add_argument("--output-root", default="outputs/ch3_ablation")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tmax", type=int, default=256)
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--win-size-lines", type=int, default=20)
    ap.add_argument("--max-blocks-per-file", type=int, default=768)
    ap.add_argument("--max-total-blocks", type=int, default=0)
    ap.add_argument("--lambda-pr", type=float, default=1.0)
    ap.add_argument("--lambda-ortho", type=float, default=0.1)
    ap.add_argument("--lambda-ortho-warmup-epochs", type=int, default=0)
    ap.add_argument("--beta-bug-file", type=float, default=0.2)
    ap.add_argument("--freeze-encoder", type=int, default=1)
    ap.add_argument("--encoder-device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--run-metrics", type=int, default=1)
    ap.add_argument("--dump-max-files", type=int, default=None)
    ap.add_argument("--eval-use-bug-file", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).parent / "run_direct_transfer.py"
    probe_att_py = Path(__file__).parent / "probe_att_mean.py"
    rows = []
    probe_rows = []
    scorr_rows = []

    for v in VARIANTS:
        variant_root = out_root / v["name"]
        variant_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(run_py),
            "--data-parquet",
            args.data_parquet,
            "--project-vocab",
            args.project_vocab,
            "--codebert-path",
            args.codebert_path,
            "--target-project",
            args.target_project,
            "--output-root",
            str(variant_root),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--tmax",
            str(args.tmax),
            "--w",
            str(args.w),
            "--win-size-lines",
            str(args.win_size_lines),
            "--max-blocks-per-file",
            str(args.max_blocks_per_file),
            "--max-total-blocks",
            str(args.max_total_blocks),
            "--lambda-pr",
            str(args.lambda_pr),
            "--lambda-ortho",
            str(args.lambda_ortho),
            "--lambda-ortho-warmup-epochs",
            str(args.lambda_ortho_warmup_epochs),
            "--beta-bug-file",
            str(args.beta_bug_file),
            "--freeze-encoder",
            str(args.freeze_encoder),
            "--encoder-device",
            str(args.encoder_device),
            "--seed",
            str(args.seed),
            "--log-every",
            str(args.log_every),
            "--run-metrics",
            str(args.run_metrics),
            "--eval-use-bug-file",
            str(args.eval_use_bug_file),
        ] + v["flags"]

        if args.dump_max_files is not None:
            cmd += ["--dump-max-files", str(args.dump_max_files)]
        if args.block_cache_dir:
            cmd += ["--block-cache-dir", args.block_cache_dir]

        run(cmd)

        target_dir = variant_root / f"target_{args.target_project}"
        metrics_path = target_dir / "direct_eval" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for {v['name']}: {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        m05 = metrics.get("metrics_at_0p5") or {}
        rows.append(
            {
                "variant": v["name"],
                "desc": v["desc"],
                "target_project": metrics.get("target_project", args.target_project),
                "n_samples": metrics.get("n_samples"),
                "pos_ratio": metrics.get("pos_ratio"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "precision": m05.get("precision"),
                "recall": m05.get("recall"),
                "f1": m05.get("f1"),
                "acc": m05.get("acc"),
                "pf": m05.get("pf"),
                "mcc": m05.get("mcc"),
                "use_bug_file": metrics.get("use_bug_file"),
            }
        )

        # project probe table
        probe_path = target_dir / "domain_probe.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            probe_rows.append(
                {
                    "variant": v["name"],
                    "desc": v["desc"],
                    "h_file_ce_test": probe.get("h_file_ce_test"),
                    "z_sh_ce_test": probe.get("z_sh_ce_test"),
                    "z_pr_ce_test": probe.get("z_pr_ce_test"),
                    "h_file_acc_test": probe.get("h_file_acc_test"),
                    "z_sh_acc_test": probe.get("z_sh_acc_test"),
                    "z_pr_acc_test": probe.get("z_pr_acc_test"),
                }
            )

        # s_corr table
        repr_metrics_path = target_dir / "dump" / "repr_metrics.json"
        if repr_metrics_path.exists():
            repr_metrics = json.loads(repr_metrics_path.read_text(encoding="utf-8"))
            scorr_rows.append(
                {
                    "variant": v["name"],
                    "desc": v["desc"],
                    "s_corr": repr_metrics.get("s_corr"),
                }
            )

    # component ablation table
    out_csv = out_root / "ablation_results.csv"
    fieldnames = [
        "variant",
        "desc",
        "target_project",
        "n_samples",
        "pos_ratio",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "acc",
        "pf",
        "mcc",
        "use_bug_file",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # project probe table
    probe_csv = out_root / "project_probe_results.csv"
    probe_fields = [
        "variant",
        "desc",
        "h_file_ce_test",
        "z_sh_ce_test",
        "z_pr_ce_test",
        "h_file_acc_test",
        "z_sh_acc_test",
        "z_pr_acc_test",
    ]
    with probe_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=probe_fields)
        writer.writeheader()
        writer.writerows(probe_rows)

    # s_corr table
    scorr_csv = out_root / "s_corr_results.csv"
    with scorr_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "desc", "s_corr"])
        writer.writeheader()
        writer.writerows(scorr_rows)

    # attention vs mean pooling (use full_method dump)
    att_mean_rows = []
    full_dir = out_root / "full_method" / f"target_{args.target_project}"
    repr_path = full_dir / "dump" / "repr.npz"
    att_out = full_dir / "probe_att_mean.json"
    if repr_path.exists():
        run(
            [
                "python",
                str(probe_att_py),
                "--repr",
                str(repr_path),
                "--out",
                str(att_out),
            ]
        )
        if att_out.exists():
            att = json.loads(att_out.read_text(encoding="utf-8"))
            att_mean_rows = [
                {"pooling": "attention", "ce_test": att.get("att_ce_test")},
                {"pooling": "mean", "ce_test": att.get("mean_ce_test")},
            ]
    att_csv = out_root / "att_vs_mean.csv"
    with att_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pooling", "ce_test"])
        writer.writeheader()
        writer.writerows(att_mean_rows)

    # plots
    figs_dir = out_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Component metrics bar (ROC-AUC / PR-AUC / MCC)
    if rows:
        names = [r["variant"] for r in rows]
        roc = [r.get("roc_auc") or 0 for r in rows]
        pr = [r.get("pr_auc") or 0 for r in rows]
        mcc = [r.get("mcc") or 0 for r in rows]
        x = range(len(names))
        width = 0.25
        plt.figure(figsize=(8, 4))
        plt.bar([i - width for i in x], roc, width, label="ROC-AUC")
        plt.bar(x, pr, width, label="PR-AUC")
        plt.bar([i + width for i in x], mcc, width, label="MCC")
        plt.xticks(list(x), names, rotation=20, ha="right")
        plt.ylabel("score")
        plt.title("Component ablation metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / "ablation_metrics_bar.png", dpi=200)
        plt.close()

    # Project probe CE bar
    if probe_rows:
        names = [r["variant"] for r in probe_rows]
        h_ce = [r.get("h_file_ce_test") or 0 for r in probe_rows]
        zsh_ce = [r.get("z_sh_ce_test") or 0 for r in probe_rows]
        zpr_ce = [r.get("z_pr_ce_test") or 0 for r in probe_rows]
        x = range(len(names))
        width = 0.25
        plt.figure(figsize=(8, 4))
        plt.bar([i - width for i in x], h_ce, width, label="H_file CE")
        plt.bar(x, zsh_ce, width, label="Z_sh CE")
        plt.bar([i + width for i in x], zpr_ce, width, label="Z_pr CE")
        plt.xticks(list(x), names, rotation=20, ha="right")
        plt.ylabel("CE (lower is better)")
        plt.title("Project probe CE (source-only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / "project_probe_ce_bar.png", dpi=200)
        plt.close()

    # Attention vs mean CE bar
    if att_mean_rows:
        labels = [r["pooling"] for r in att_mean_rows]
        vals = [r.get("ce_test") or 0 for r in att_mean_rows]
        plt.figure(figsize=(4, 4))
        plt.bar(labels, vals, color=["#4c72b0", "#55a868"])
        plt.ylabel("CE (lower is better)")
        plt.title("Attention vs Mean pooling")
        plt.tight_layout()
        plt.savefig(figs_dir / "att_vs_mean_ce.png", dpi=200)
        plt.close()

    print(f"Saved ablation summary to {out_csv}")
    print(f"Saved project probe table to {probe_csv}")
    print(f"Saved s_corr table to {scorr_csv}")
    print(f"Saved attention vs mean table to {att_csv}")


if __name__ == "__main__":
    main()
