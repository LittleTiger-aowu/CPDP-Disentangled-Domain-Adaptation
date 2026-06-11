"""
Run lucene-2.4 ablation across multiple seeds and summarize into one CSV.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    seeds = [0, 1, 2, 3, 4]
    out_root = Path("outputs/ch4_runs/ablation_lucene_f2_pct")
    out_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "python",
        str(Path(__file__).parent / "run_ablation_mylyn.py"),
        "--dump-dir",
        r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_lucene-2.4\dump",
        "--source-projects",
        "Eclipse_JDT_Core-3.4,Equinox-3.4,Mylyn-3.1,PDE_UI-3.4.1",
        "--target-project",
        "lucene-2.4",
        "--epochs",
        "10",
        "--batch-size",
        "64",
        "--lr",
        "0.001",
        "--lambda-dom",
        "0.5",
        "--grl-gamma",
        "10",
        "--feature-key",
        "H_file",
        "--use-bottleneck",
        "1",
        "--bottleneck-dim",
        "128",
        "--am-margin",
        "0.3",
        "--temp-scale",
        "1",
        "--temp-max-iter",
        "200",
        "--threshold-mode",
        "fbeta",
        "--f-beta",
        "2.0",
        "--use-percentile",
        "1",
        "--drift-thr",
        "0.05",
    ]

    summary_rows = []
    for seed in seeds:
        seed_out = out_root / f"seed_{seed}"
        cmd = base_cmd + ["--output-root", str(seed_out), "--seed", str(seed)]
        run_cmd(cmd)
        summary_path = seed_out / "ablation_summary_lucene-2.4.csv"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r["seed"] = seed
                    summary_rows.append(r)

    if summary_rows:
        out_csv = out_root / "ablation_summary_all_seeds.csv"
        fieldnames = sorted({k for r in summary_rows for k in r.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Saved summary to {out_csv}")
    else:
        print("No summaries found.")


if __name__ == "__main__":
    main()
