"""
Run a small ablation suite end-to-end: train -> dump -> metrics/probe/coral/rank for each setting.

Edit the BASE paths below if your workspace differs.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


# ---------- User-configurable base paths ----------
ROOT = Path(r"E:\project\WYP\LineDefStudy2.0")
DATA_PARQUET = ROOT / r"data\processed\ubd_class.parquet"
PROJECT_VOCAB = ROOT / r"data\processed\ubd_project_vocab.json"
CODEBERT = Path(r"E:\project\WYP\CPDP\CodeBert")  # NOTE: updated to the actual CodeBert path
BLOCK_CACHE = ROOT / r"outputs\block_cache_w3_t256_b768"  # change if using another cache

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
    "--freeze-encoder",
    "1",
    "--encoder-device",
    "cuda",
    "--batch-size",
    "20",
    "--max-total-blocks",
    "2048",
    "--shuffle",
    "1",
    "--skip-template",
    "1",
    "--skip-missing-src",
    "1",
    "--epochs",
    "3",  # keep small for quick ablations; adjust as needed
    "--balanced-batch",
    "1",
    "--projects-per-batch",
    "5",
    "--lambda-ortho-warmup-epochs",
    "1",
    "--block-cache-dir",
    str(BLOCK_CACHE),
    "--progress",
    "1",
    "--log-file",
    "1",
    "--metrics-file",
    "1",
    "--dump-config-every",
    "300",
    "--log-every",
    "50",
    "--debug-ortho",
    "1",
    "--debug-ortho-every",
    "100",
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
    "--codebert-path",
    str(CODEBERT),
    "--local-files-only",
    "1",
    "--batch-size",
    "4",
]

# Ablation configurations
RUNS = [
    {
        "name": "full",
        "extra_train": ["--lambda-ortho", "0.1"],
    },
    {
        "name": "no_ortho",
        "extra_train": ["--lambda-ortho", "0.0", "--no-ortho", "1"],
    },
    {
        "name": "lambda_0p05",
        "extra_train": ["--lambda-ortho", "0.05"],
    },
    {
        "name": "lambda_0p2",
        "extra_train": ["--lambda-ortho", "0.2"],
    },
    {
        "name": "no_pr_dom",
        "extra_train": ["--lambda-ortho", "0.1", "--no-pr-dom", "1"],
    },
    {
        "name": "no_gcn",
        "extra_train": ["--lambda-ortho", "0.1", "--no-gcn", "1"],
    },
]


def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    train_py = str(ROOT / r"src\ch3\train_representation.py")
    dump_py = str(ROOT / r"src\ch3\dump_representations.py")
    metrics_py = str(ROOT / r"src\ch3\compute_repr_metrics.py")
    probe_py = str(ROOT / r"src\ch3\probe_domain_ce.py")
    coral_py = str(ROOT / r"src\ch3\compute_coral.py")
    rank_py = str(ROOT / r"src\ch3\compute_effective_rank.py")

    for cfg in RUNS:
        run_name = cfg["name"]
        out_dir = ROOT / "outputs" / "ch3_runs" / run_name
        ckpt_dir = out_dir
        dump_dir = out_dir / "dump_best"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        train_cmd = ["python", train_py] + COMMON_TRAIN + cfg["extra_train"] + [
            "--output-dir",
            str(ckpt_dir),
        ]
        run(train_cmd)

        # 2) Dump (use best.pt if exists, else last.pt)
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

        # 4) Domain probe (CE)
        run(["python", probe_py, "--repr", str(dump_dir / "repr.npz"), "--out", str(out_dir / "domain_probe.json")])

        # 5) CORAL
        run(["python", coral_py, "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "coral.json")])

        # 6) Effective rank
        run(["python", rank_py, "--repr-npz", str(dump_dir / "repr.npz"), "--out", str(out_dir / "rank.json")])

    print("All runs completed. Results under outputs/ch3_runs/<run_name>/")


if __name__ == "__main__":
    main()
