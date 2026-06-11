"""
Run baselines (TCA/SA/CORAL) + DANN baseline + our method across all projects and seeds.
Outputs a combined flattened summary CSV.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def main():
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

    seeds = [0, 1, 2, 3, 4]
    out_root = Path("outputs/ch4_runs/all_methods_all_seeds")
    out_root.mkdir(parents=True, exist_ok=True)

    baselines_py = Path(__file__).parent / "run_baselines_vs_dann.py"
    ours_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    baseline_args = [
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

    ours_args = [
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

    # Run experiments
    for seed in seeds:
        for p in projects:
            base_dir = out_root / f"seed_{seed}" / f"target_{p['target']}"
            base_dir.mkdir(parents=True, exist_ok=True)

            # Baselines + DANN
            run_cmd(
                [
                    "python",
                    str(baselines_py),
                    "--dump-dir",
                    p["dump"],
                    "--output-root",
                    str(base_dir / "baselines"),
                    "--source-projects",
                    p["sources"],
                    "--target-project",
                    p["target"],
                    "--seed",
                    str(seed),
                ]
                + baseline_args
            )

            # Our method (single-target LOOCV)
            run_cmd(
                [
                    "python",
                    str(ours_py),
                    "--dump-dir",
                    p["dump"],
                    "--source-projects",
                    p["sources"],
                    "--target-project",
                    p["target"],
                    "--output-dir",
                    str(base_dir / "ours"),
                    "--seed",
                    str(seed),
                ]
                + ours_args
            )

    # Collect metrics into one CSV
    rows = []
    for seed in seeds:
        for p in projects:
            base_dir = out_root / f"seed_{seed}" / f"target_{p['target']}"

            # Baselines
            for method in ["tca", "sa", "coral"]:
                mpath = base_dir / "baselines" / f"{method}_metrics.json"
                if mpath.exists():
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                    flat = flatten(m)
                    flat["seed"] = seed
                    flat["target_project"] = p["target"]
                    flat["method"] = method
                    rows.append(flat)

            # DANN baseline
            dann_path = base_dir / "baselines" / "dann_base" / "metrics.json"
            if dann_path.exists():
                m = json.loads(dann_path.read_text(encoding="utf-8"))
                flat = flatten(m)
                flat["seed"] = seed
                flat["target_project"] = p["target"]
                flat["method"] = "dann_base"
                rows.append(flat)

            # Our method
            ours_metrics = base_dir / "ours" / "metrics.json"
            if ours_metrics.exists():
                m = json.loads(ours_metrics.read_text(encoding="utf-8"))
                flat = flatten(m)
                flat["seed"] = seed
                flat["target_project"] = p["target"]
                flat["method"] = "ours"
                rows.append(flat)

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        out_csv = out_root / "summary_all_methods_all_seeds.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved summary to {out_csv}")
    else:
        print("No metrics found.")


if __name__ == "__main__":
    main()
