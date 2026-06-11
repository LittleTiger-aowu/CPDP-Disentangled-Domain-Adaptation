"""
Run Ch3 ablation suite with SSL/ADV always disabled.

For each run:
  1) train
  2) dump representations
  3) compute repr metrics + domain probe + CORAL + effective rank
Then collect results into a summary CSV/MD.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


# ---------- User-configurable base paths ----------
ROOT = Path(r"E:\project\WYP\LineDefStudy2.0")
DATA_PARQUET = ROOT / r"data\processed\ubd_class.parquet"
PROJECT_VOCAB = ROOT / r"data\processed\ubd_project_vocab.json"
CODEBERT = Path(r"E:\project\WYP\CPDP\CodeBert")
BLOCK_CACHE = ROOT / r"outputs\block_cache_w3_t256_b768"
TARGET_PROJECT = "Mylyn-3.1"

# Output root for ablations
RUNS_ROOT = ROOT / r"outputs\ch3_ablations_no_ssl_adv"

# Common training args (shared across runs)
COMMON_TRAIN = [
    "--data-parquet",
    str(DATA_PARQUET),
    "--project-vocab",
    str(PROJECT_VOCAB),
    "--codebert-path",
    str(CODEBERT),
    "--local-files-only",
    "1",
    "--tmax",
    "256",
    "--w",
    "3",
    "--win-size-lines",
    "20",
    "--max-blocks-per-file",
    "768",
    "--block-cache-dir",
    str(BLOCK_CACHE),
    "--epochs",
    "10",
    "--batch-size",
    "32",
    "--balanced-batch",
    "1",
    "--projects-per-batch",
    "5",
    "--ensure-source",
    "1",
    "--ch3-objective",
    "source_bug_only",
    "--target-project",
    TARGET_PROJECT,
    "--include-target-unlabeled",
    "1",
    "--lambda-pr",
    "1.0",
    "--lambda-adv",
    "0",
    "--lambda-ssl",
    "0",
    "--beta-bug-file",
    "0",
    "--max-total-blocks",
    "0",
    "--log-every",
    "50",
    "--metrics-file",
    "1",
    "--log-file",
    "1",
]

# Common dump args
DUMP_COMMON = [
    "--data-parquet",
    str(DATA_PARQUET),
    "--tmax",
    "256",
    "--w",
    "3",
    "--win-size-lines",
    "20",
    "--max-blocks-per-file",
    "768",
    "--block-cache-dir",
    str(BLOCK_CACHE),
    "--codebert-path",
    str(CODEBERT),
    "--local-files-only",
    "1",
    "--batch-size",
    "4",
]

# Ablation configurations (SSL/ADV always off)
RUNS = [
    {
        "name": "base_ortho0p4",
        "extra_train": ["--lambda-ortho", "0.4", "--lambda-ortho-warmup-epochs", "2"],
    },
    {
        "name": "no_ortho",
        "extra_train": ["--lambda-ortho", "0.0", "--no-ortho", "1"],
    },
    {
        "name": "ortho0p2",
        "extra_train": ["--lambda-ortho", "0.2", "--lambda-ortho-warmup-epochs", "2"],
    },
    {
        "name": "ortho0p6",
        "extra_train": ["--lambda-ortho", "0.6", "--lambda-ortho-warmup-epochs", "2"],
    },
    {
        "name": "no_pr_dom",
        "extra_train": ["--lambda-ortho", "0.4", "--lambda-ortho-warmup-epochs", "2", "--no-pr-dom", "1"],
    },
    {
        "name": "no_gcn",
        "extra_train": ["--lambda-ortho", "0.4", "--lambda-ortho-warmup-epochs", "2", "--no-gcn", "1"],
    },
    {
        "name": "no_struct",
        "extra_train": ["--lambda-ortho", "0.4", "--lambda-ortho-warmup-epochs", "2", "--no-struct", "1"],
    },
    {
        "name": "no_bug_head",
        "extra_train": ["--lambda-ortho", "0.4", "--lambda-ortho-warmup-epochs", "2", "--disable-bug-head", "1"],
    },
]


def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    train_py = str(ROOT / r"src\ch3\train_representation.py")
    dump_py = str(ROOT / r"src\ch3\dump_representations.py")
    metrics_py = str(ROOT / r"src\ch3\compute_repr_metrics.py")
    probe_py = str(ROOT / r"src\ch3\probe_domain_ce.py")
    coral_py = str(ROOT / r"src\ch3\compute_coral.py")
    rank_py = str(ROOT / r"src\ch3\compute_effective_rank.py")
    collect_py = str(ROOT / r"src\ch3\collect_ch3_results.py")

    for cfg in RUNS:
        run_name = cfg["name"]
        out_dir = RUNS_ROOT / run_name
        ckpt_dir = out_dir
        dump_dir = out_dir / "dump_best"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        train_cmd = ["python", train_py] + COMMON_TRAIN + cfg["extra_train"] + [
            "--output-dir",
            str(ckpt_dir),
        ]
        run(train_cmd)

        # 2) Dump (best if exists)
        ckpt_best = ckpt_dir / "best.pt"
        ckpt_use = ckpt_best if ckpt_best.exists() else ckpt_dir / "last.pt"
        dump_cmd = ["python", dump_py] + DUMP_COMMON + [
            "--ckpt",
            str(ckpt_use),
            "--outdir",
            str(dump_dir),
        ]
        run(dump_cmd)

        # 3) Metrics
        run(["python", metrics_py, "--dump-dir", str(dump_dir)])
        run(["python", probe_py, "--repr", str(dump_dir / "repr.npz"), "--out", str(out_dir / "domain_probe.json")])
        run(["python", coral_py, "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "coral.json")])
        run(["python", rank_py, "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "rank.json")])

    # Summary
    summary_csv = RUNS_ROOT / "ablation_summary.csv"
    summary_md = RUNS_ROOT / "ablation_summary.md"
    run(
        [
            "python",
            collect_py,
            "--runs-root",
            str(RUNS_ROOT),
            "--out-csv",
            str(summary_csv),
            "--out-md",
            str(summary_md),
        ]
    )
    print(f"All runs completed. Summary at {summary_csv}")


if __name__ == "__main__":
    main()
