"""
Wrapper to run run_ablation_study_val_threshold.py sequentially for 5 target projects.
Order is fixed (do not rely on project_vocab.json).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


TARGETS = [
    "Eclipse_JDT_Core-3.4",
    "Equinox-3.4",
    "Mylyn-3.1",
    "PDE_UI-3.4.1",
    "lucene-2.4",
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
    ap.add_argument("--output-root", default="outputs/ch3_ablation_val_threshold")
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
    ap.add_argument("--dump-max-files", type=int, default=None)
    ap.add_argument("--eval-use-bug-file", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    run_py = Path(__file__).parent / "run_ablation_study_val_threshold.py"

    for target in TARGETS:
        target_root = Path(args.output_root) / f"target_{target}"
        target_root.mkdir(parents=True, exist_ok=True)
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
            target,
            "--output-root",
            str(target_root),
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
            "--eval-use-bug-file",
            str(args.eval_use_bug_file),
            "--val-ratio",
            str(args.val_ratio),
        ]
        if args.dump_max_files is not None:
            cmd += ["--dump-max-files", str(args.dump_max_files)]
        if args.block_cache_dir:
            cmd += ["--block-cache-dir", args.block_cache_dir]
        run(cmd)


if __name__ == "__main__":
    main()
