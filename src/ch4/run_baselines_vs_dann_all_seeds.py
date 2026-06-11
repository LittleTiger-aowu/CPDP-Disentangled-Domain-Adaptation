"""
Run baselines vs DANN for all 5 projects across multiple seeds, then summarize.
"""
from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    base_dir = r"E:\project\WYP\LineDefStudy2.0"
    script = Path(__file__).parent / "run_baselines_vs_dann.py"
    base_args = [
        "--feature-key",
        "H_file",
        "--n-components",
        "64",
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
    ]

    projects = [
        {
            "target": "Mylyn-3.1",
            "dump": rf"{base_dir}\outputs\ch3_optimized_D\D_full\target_Mylyn-3.1\dump",
            "sources": "Eclipse_JDT_Core-3.4,Equinox-3.4,PDE_UI-3.4.1,lucene-2.4",
        },
        {
            "target": "Equinox-3.4",
            "dump": rf"{base_dir}\outputs\ch3_optimized_D\D_full\target_Equinox-3.4\dump",
            "sources": "Eclipse_JDT_Core-3.4,Mylyn-3.1,PDE_UI-3.4.1,lucene-2.4",
        },
        {
            "target": "Eclipse_JDT_Core-3.4",
            "dump": rf"{base_dir}\outputs\ch3_optimized_D\D_full\target_Eclipse_JDT_Core-3.4\dump",
            "sources": "Equinox-3.4,Mylyn-3.1,PDE_UI-3.4.1,lucene-2.4",
        },
        {
            "target": "PDE_UI-3.4.1",
            "dump": rf"{base_dir}\outputs\ch3_optimized_D\D_full\target_PDE_UI-3.4.1\dump",
            "sources": "Eclipse_JDT_Core-3.4,Equinox-3.4,Mylyn-3.1,lucene-2.4",
        },
        {
            "target": "lucene-2.4",
            "dump": rf"{base_dir}\outputs\ch3_optimized_D\D_full\target_lucene-2.4\dump",
            "sources": "Eclipse_JDT_Core-3.4,Equinox-3.4,Mylyn-3.1,PDE_UI-3.4.1",
        },
    ]

    seeds = [0, 1, 2, 3, 4]
    out_root = Path("outputs/ch4_runs/baselines_vs_dann_all_seeds")
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for p in projects:
            out_dir = out_root / f"seed_{seed}" / f"target_{p['target']}"
            cmd = [
                "python",
                str(script),
                "--dump-dir",
                p["dump"],
                "--output-root",
                str(out_dir),
                "--source-projects",
                p["sources"],
                "--target-project",
                p["target"],
                "--seed",
                str(seed),
            ] + base_args
            run_cmd(cmd)

    # collect all summary CSVs
    rows = []
    for seed in seeds:
        for p in projects:
            summary = out_root / f"seed_{seed}" / f"target_{p['target']}" / f"baseline_vs_dann_{p['target']}.csv"
            if not summary.exists():
                continue
            with summary.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r["seed"] = seed
                    r["target_project"] = p["target"]
                    rows.append(r)

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
