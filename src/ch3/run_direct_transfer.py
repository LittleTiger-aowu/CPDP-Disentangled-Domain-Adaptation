"""
Run Chapter 3 direct-transfer pipeline for one target project.

Steps:
  1) Train Ch3 on source projects only (exclude target)
  2) Dump representations for plotting/metrics
  3) Compute repr metrics + probes + plots
  4) Direct-transfer evaluation on target project
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def load_projects(project_vocab_path: str) -> list[str]:
    vocab = json.load(open(project_vocab_path, "r", encoding="utf-8"))
    return [name for name, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--project-vocab", required=True)
    ap.add_argument("--codebert-path", required=True)
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--source-projects", default=None, help="comma-separated; default=all except target")
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
    ap.add_argument("--no-pr-dom", type=int, default=0)
    ap.add_argument("--no-ortho", type=int, default=0)
    ap.add_argument("--no-gcn", type=int, default=0)
    ap.add_argument("--no-struct", type=int, default=0)
    ap.add_argument("--freeze-encoder", type=int, default=1)
    ap.add_argument("--encoder-device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--run-metrics", type=int, default=1)
    ap.add_argument("--dump-max-files", type=int, default=None)
    ap.add_argument("--eval-use-bug-file", type=int, default=0)
    args = ap.parse_args()

    all_projects = load_projects(args.project_vocab)
    if args.target_project not in all_projects:
        raise ValueError(f"target project {args.target_project} not in project_vocab")

    if args.source_projects:
        source_projects = [p.strip() for p in args.source_projects.split(",") if p.strip()]
    else:
        source_projects = [p for p in all_projects if p != args.target_project]

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"target_{args.target_project}"
    ckpt_dir = out_dir / "ckpt"
    dump_dir = out_dir / "dump"
    figs_dir = out_dir / "figs"
    eval_dir = out_dir / "direct_eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_py = Path(__file__).parent / "train_representation.py"
    dump_py = Path(__file__).parent / "dump_representations.py"
    metrics_py = Path(__file__).parent / "compute_repr_metrics.py"
    probe_py = Path(__file__).parent / "probe_domain_ce.py"
    coral_py = Path(__file__).parent / "compute_coral.py"
    rank_py = Path(__file__).parent / "compute_effective_rank.py"
    alpha_py = Path(__file__).parent / "compute_alpha_uniform_gap.py"
    plot_py = Path(__file__).parent / "plot_repr_figs.py"
    eval_py = Path(__file__).parent / "eval_direct_transfer.py"

    train_cmd = [
        "python",
        str(train_py),
        "--data-parquet",
        args.data_parquet,
        "--project-vocab",
        args.project_vocab,
        "--target-project",
        args.target_project,
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
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--freeze-encoder",
        str(args.freeze_encoder),
        "--encoder-device",
        str(args.encoder_device),
        "--balanced-batch",
        "1",
        "--projects-per-batch",
        str(min(len(source_projects), 5)),
        "--lambda-pr",
        str(args.lambda_pr),
        "--lambda-ortho",
        str(args.lambda_ortho),
        "--lambda-ortho-warmup-epochs",
        str(args.lambda_ortho_warmup_epochs),
        "--beta-bug-file",
        str(args.beta_bug_file),
        "--no-pr-dom",
        str(args.no_pr_dom),
        "--no-ortho",
        str(args.no_ortho),
        "--no-gcn",
        str(args.no_gcn),
        "--no-struct",
        str(args.no_struct),
        "--max-total-blocks",
        str(args.max_total_blocks),
        "--log-every",
        str(args.log_every),
        "--log-file",
        "1",
        "--metrics-file",
        "1",
        "--output-dir",
        str(ckpt_dir),
        "--seed",
        str(args.seed),
    ]
    if args.block_cache_dir:
        train_cmd += ["--block-cache-dir", args.block_cache_dir]
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
    if args.dump_max_files is not None:
        dump_cmd += ["--max-files", str(args.dump_max_files)]
    if args.block_cache_dir:
        dump_cmd += ["--block-cache-dir", args.block_cache_dir]
    run(dump_cmd)

    if args.run_metrics:
        run(["python", str(metrics_py), "--dump-dir", str(dump_dir)])
        run(
            [
                "python",
                str(probe_py),
                "--repr",
                str(dump_dir / "repr.npz"),
                "--target-project",
                args.target_project,
                "--source-only",
                "1",
                "--out",
                str(out_dir / "domain_probe.json"),
            ]
        )
        run(["python", str(coral_py), "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "coral.json")])
        run(["python", str(rank_py), "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "rank.json")])
        run(["python", str(alpha_py), "--repr-npz", str(dump_dir / "repr.npz")])
        run(
            [
                "python",
                str(plot_py),
                "--dump-dir",
                str(dump_dir),
                "--out-dir",
                str(figs_dir),
                "--exclude-projects",
                args.target_project,
                "--target-project",
                args.target_project,
            ]
        )

    eval_cmd = [
        "python",
        str(eval_py),
        "--data-parquet",
        args.data_parquet,
        "--project-vocab",
        args.project_vocab,
        "--ckpt",
        str(ckpt_use),
        "--target-project",
        args.target_project,
        "--output-dir",
        str(eval_dir),
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
        "--use-bug-file",
        str(args.eval_use_bug_file),
    ]
    if args.block_cache_dir:
        eval_cmd += ["--block-cache-dir", args.block_cache_dir]
    run(eval_cmd)

    summary = {
        "target_project": args.target_project,
        "source_projects": source_projects,
        "ckpt": str(ckpt_use),
        "dump_dir": str(dump_dir),
        "eval_dir": str(eval_dir),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
