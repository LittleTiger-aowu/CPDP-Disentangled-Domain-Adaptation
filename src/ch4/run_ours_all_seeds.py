"""
Run our CPDP method across multiple seeds on the same dump dir (LOOCV),
and summarize into a single CSV.
"""
from __future__ import annotations

import csv
import json
import json
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    out_root = Path("outputs/ch4_runs/ours_all_seeds")
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2, 3, 4]

    projects = [
        {
            "target": "Mylyn-3.1",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Mylyn-3.1\dump",
        },
        {
            "target": "Equinox-3.4",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Equinox-3.4\dump",
        },
        {
            "target": "Eclipse_JDT_Core-3.4",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Eclipse_JDT_Core-3.4\dump",
        },
        {
            "target": "PDE_UI-3.4.1",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_PDE_UI-3.4.1\dump",
        },
        {
            "target": "lucene-2.4",
            "dump": r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_lucene-2.4\dump",
        },
    ]

    base_args = [
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

    runner = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    for seed in seeds:
        for p in projects:
            seed_out = out_root / f"seed_{seed}" / f"target_{p['target']}"
            cmd = [
                "python",
                str(runner),
                "--dump-dir",
                p["dump"],
                "--source-projects",
                ",".join([proj["target"] for proj in projects if proj["target"] != p["target"]]),
                "--target-project",
                p["target"],
                "--output-dir",
                str(seed_out),
                "--seed",
                str(seed),
            ] + base_args
            run_cmd(cmd)

    # collect all summary.csv
    rows = []
    for seed in seeds:
        for p in projects:
            metrics_path = out_root / f"seed_{seed}" / f"target_{p['target']}" / "metrics.json"
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
                m["seed"] = seed
                m["target_project"] = p["target"]
                rows.append(m)

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        out_csv = out_root / "summary_all_projects_all_seeds.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved summary to {out_csv}")
    else:
        print("No summaries found.")


if __name__ == "__main__":
    main()
