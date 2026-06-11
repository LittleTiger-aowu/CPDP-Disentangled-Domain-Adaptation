"""
Run ablation experiments for all projects (single seed) using existing Mylyn ablation runner.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs/ch4_runs/ablation_all_projects")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dom", type=float, default=0.5)
    ap.add_argument("--grl-gamma", type=float, default=10.0)
    ap.add_argument("--feature-key", default="H_file")
    ap.add_argument("--use-bottleneck", type=int, default=1)
    ap.add_argument("--bottleneck-dim", type=int, default=128)
    ap.add_argument("--am-margin", type=float, default=0.3)
    ap.add_argument("--temp-scale", type=int, default=1)
    ap.add_argument("--temp-max-iter", type=int, default=200)
    ap.add_argument("--threshold-mode", default="fbeta", choices=["mcc", "f1", "fbeta"])
    ap.add_argument("--f-beta", type=float, default=2.0)
    ap.add_argument("--use-percentile", type=int, default=1)
    ap.add_argument("--drift-thr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
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

    runner = Path(__file__).parent / "run_ablation_mylyn.py"
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for p in projects:
        out_dir = out_root / f"target_{p['target']}"
        cmd = [
            "python",
            str(runner),
            "--dump-dir",
            p["dump"],
            "--output-root",
            str(out_dir),
            "--source-projects",
            p["sources"],
            "--target-project",
            p["target"],
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
            str(args.feature_key),
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
            "--seed",
            str(args.seed),
        ]
        run_cmd(cmd)


if __name__ == "__main__":
    main()
