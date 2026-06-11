"""
Leave-one-project-out pretrain for Chapter 3 with target-unlabeled participation.

For each project as target:
  1) train_representation.py with ch3-objective/source_bug_only (or ssl_only)
  2) dump_representations.py
  3) optional metrics: compute_repr_metrics, probe_domain_ce, compute_coral, compute_effective_rank, plot_repr_figs, compute_alpha_uniform_gap

Adjust BASE paths to your workspace/caches.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--project-vocab", required=True)
    ap.add_argument("--codebert-path", required=True)
    ap.add_argument("--block-cache-dir", required=True)
    ap.add_argument("--projects", required=True, help="comma-separated project names")
    ap.add_argument("--out-root", required=True, help="root dir to store outputs/ch3_loocv/<target>/...")
    ap.add_argument("--objective", choices=["source_bug_only", "ssl_only", "bug_all"], default="source_bug_only")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tmax", type=int, default=256)
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--win-size-lines", type=int, default=20)
    ap.add_argument("--max-blocks-per-file", type=int, default=768)
    ap.add_argument("--max-total-blocks", type=int, default=4096)
    ap.add_argument("--lambda-ortho", type=float, default=0.1)
    ap.add_argument("--lambda-pr", type=float, default=1.0)
    ap.add_argument("--lambda-ssl", type=float, default=1.0)
    ap.add_argument("--lambda-var", type=float, default=25.0)
    ap.add_argument("--lambda-cov", type=float, default=1.0)
    ap.add_argument("--var-target", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.0)
    ap.add_argument("--adv-hidden", type=int, default=64)
    ap.add_argument("--adv-grl-gamma", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-metrics", type=int, default=1)
    args = ap.parse_args()

    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_py = Path(__file__).parent / "train_representation.py"
    dump_py = Path(__file__).parent / "dump_representations.py"
    metrics_py = Path(__file__).parent / "compute_repr_metrics.py"
    probe_py = Path(__file__).parent / "probe_domain_ce.py"
    coral_py = Path(__file__).parent / "compute_coral.py"
    rank_py = Path(__file__).parent / "compute_effective_rank.py"
    plot_py = Path(__file__).parent / "plot_repr_figs.py"
    alpha_py = Path(__file__).parent / "compute_alpha_uniform_gap.py"

    for tgt in projects:
        out_dir = out_root / tgt
        ckpt_dir = out_dir / "ckpt"
        dump_dir = out_dir / "dump"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dump_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            "python",
            str(train_py),
            "--data-parquet",
            args.data_parquet,
            "--project-vocab",
            args.project_vocab,
            "--codebert-path",
            args.codebert_path,
            "--local-files-only",
            "1",
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
            "--block-cache-dir",
            args.block_cache_dir,
            "--freeze-encoder",
            "1",
            "--encoder-device",
            "cuda",
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--balanced-batch",
            "1",
            "--projects-per-batch",
            "5",
            "--ensure-source",
            "1",
            "--ch3-objective",
            args.objective,
            "--target-project",
            tgt,
            "--include-target-unlabeled",
            "1",
            "--lambda-ortho",
            str(args.lambda_ortho),
            "--lambda-pr",
            str(args.lambda_pr),
            "--lambda-ssl",
            str(args.lambda_ssl),
            "--lambda-var",
            str(args.lambda_var),
            "--lambda-cov",
            str(args.lambda_cov),
            "--var-target",
            str(args.var_target),
            "--lambda-adv",
            str(args.lambda_adv),
            "--adv-hidden",
            str(args.adv_hidden),
            "--adv-grl-gamma",
            str(args.adv_grl_gamma),
            "--progress",
            "1",
            "--log-file",
            "1",
            "--metrics-file",
            "1",
            "--dump-config-every",
            "300",
            "--output-dir",
            str(ckpt_dir),
            "--seed",
            str(args.seed),
        ]
        run(train_cmd)

        ckpt_best = ckpt_dir / "best.pt"
        ckpt_use = ckpt_best if ckpt_best.exists() else ckpt_dir / "last.pt"

        dump_cmd = [
            "python",
            str(dump_py),
            "--data-parquet",
            args.data_parquet,
            "--ckpt",
            str(ckpt_use),
            "--outdir",
            str(dump_dir),
            "--batch-size",
            "4",
            "--tmax",
            str(args.tmax),
            "--w",
            str(args.w),
            "--win-size-lines",
            str(args.win_size_lines),
            "--max-blocks-per-file",
            str(args.max_blocks_per_file),
            "--codebert-path",
            args.codebert_path,
            "--local-files-only",
            "1",
        ]
        run(dump_cmd)

        if args.run_metrics:
            run(["python", str(metrics_py), "--dump-dir", str(dump_dir)])
            run(["python", str(probe_py), "--repr", str(dump_dir / "repr.npz"), "--out", str(out_dir / "domain_probe.json")])
            run(["python", str(coral_py), "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "coral.json")])
            run(["python", str(rank_py), "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "rank.json")])
            run(["python", str(alpha_py), "--repr-npz", str(dump_dir / "repr.npz")])
            run(["python", str(plot_py), "--dump-dir", str(dump_dir), "--out-dir", str(out_dir / "figs")])

    print("LOOCV ch3 pretrain completed.")


if __name__ == "__main__":
    main()
