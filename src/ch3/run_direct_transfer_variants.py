"""
Run Chapter 3 direct-transfer variants for a single target project and summarize results.

Variants:
  A) codebert_only:        --no-struct 1 --no-gcn 1 --no-ortho 1 --no-pr-dom 1
  B) shallow_struct:       --no-struct 0 --no-gcn 1 --no-ortho 0 --no-pr-dom 0
  C) multimodal_no_disent: --no-struct 0 --no-gcn 0 --no-ortho 1 --no-pr-dom 1
  D) full:                 --no-struct 0 --no-gcn 0 --no-ortho 0 --no-pr-dom 0

Example:
  python src/ch3/run_direct_transfer_variants.py ^
    --data-parquet data/processed/ubd_class.parquet ^
    --project-vocab data/processed/ubd_project_vocab.json ^
    --codebert-path E:\\project\\WYP\\CPDP\\CodeBert ^
    --target-project Mylyn-3.1 ^
    --output-root outputs/ch3_direct_transfer_variants
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import csv


VARIANTS = [
    {
        "name": "A_codebert_only",
        "flags": ["--no-struct", "1", "--no-gcn", "1", "--no-ortho", "1", "--no-pr-dom", "1"],
        "desc": "CodeBERT-only (semantic only)",
    },
    {
        "name": "B_shallow_struct",
        "flags": ["--no-struct", "0", "--no-gcn", "1", "--no-ortho", "0", "--no-pr-dom", "0"],
        "desc": "CodeBERT + StructPrior (no GCN)",
    },
    {
        "name": "C_multimodal_no_disentangle",
        "flags": ["--no-struct", "0", "--no-gcn", "0", "--no-ortho", "1", "--no-pr-dom", "1"],
        "desc": "Multi-modal without disentanglement",
    },
    {
        "name": "D_full",
        "flags": ["--no-struct", "0", "--no-gcn", "0", "--no-ortho", "0", "--no-pr-dom", "0"],
        "desc": "Full method (multi-modal + disentangle)",
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
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--block-cache-dir", default=None)
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

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).parent / "run_direct_transfer.py"

    rows = []
    for v in VARIANTS:
        out_root = root / v["name"]
        out_root.mkdir(parents=True, exist_ok=True)

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
            str(out_root),
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

        metrics_path = (
            out_root / f"target_{args.target_project}" / "direct_eval" / "metrics.json"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for variant {v['name']}: {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "variant": v["name"],
            "desc": v["desc"],
            "target_project": metrics.get("target_project", args.target_project),
            "n_samples": metrics.get("n_samples"),
            "pos_ratio": metrics.get("pos_ratio"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "precision": (metrics.get("metrics_at_0p5") or {}).get("precision"),
            "recall": (metrics.get("metrics_at_0p5") or {}).get("recall"),
            "f1": (metrics.get("metrics_at_0p5") or {}).get("f1"),
            "acc": (metrics.get("metrics_at_0p5") or {}).get("acc"),
            "pf": (metrics.get("metrics_at_0p5") or {}).get("pf"),
            "mcc": (metrics.get("metrics_at_0p5") or {}).get("mcc"),
            "use_bug_file": metrics.get("use_bug_file"),
        }
        rows.append(row)

    # Summary outputs
    summary_csv = root / f"summary_{args.target_project}.csv"
    summary_md = root / f"summary_{args.target_project}.md"
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

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join([" --- "]*len(fieldnames)) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(k, "")) for k in fieldnames) + " |")
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved summary to {summary_csv} and {summary_md}")


if __name__ == "__main__":
    main()
