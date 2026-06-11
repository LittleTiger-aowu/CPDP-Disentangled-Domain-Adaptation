"""
Feature-key ablation across multiple seeds for all projects.
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

    variants = [
        # source-only
        ("source_only_hfile", "H_file", 0.0, 0, 0.0),
        ("source_only_zsh", "Z_sh", 0.0, 0, 0.0),
        # DANN baseline (no bottleneck, no margin)
        ("dann_hfile", "H_file", 0.5, 0, 0.0),
        ("dann_zsh", "Z_sh", 0.5, 0, 0.0),
        # Our method (bottleneck + margin)
        ("ours_hfile", "H_file", 0.5, 1, 0.3),
        ("ours_zsh", "Z_sh", 0.5, 1, 0.3),
    ]

    seeds = [0, 1, 2, 3, 4]
    out_root = Path("outputs/ch4_runs/feature_key_ablation_all_seeds")
    out_root.mkdir(parents=True, exist_ok=True)
    train_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    base_args = [
        "--epochs",
        "10",
        "--batch-size",
        "64",
        "--lr",
        "0.001",
        "--grl-gamma",
        "10",
        "--bottleneck-dim",
        "128",
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

    rows = []
    for seed in seeds:
        for p in projects:
            for tag, feat_key, lam, use_bn, am_margin in variants:
                out_dir = out_root / f"seed_{seed}" / f"target_{p['target']}" / tag
                cmd = [
                    "python",
                    str(train_py),
                    "--dump-dir",
                    p["dump"],
                    "--source-projects",
                    p["sources"],
                    "--target-project",
                    p["target"],
                    "--output-dir",
                    str(out_dir),
                    "--seed",
                    str(seed),
                    "--feature-key",
                    feat_key,
                    "--lambda-dom",
                    str(lam),
                    "--use-bottleneck",
                    str(use_bn),
                    "--am-margin",
                    str(am_margin),
                ] + base_args
                run_cmd(cmd)

                metrics_path = out_dir / "metrics.json"
                if metrics_path.exists():
                    m = json.loads(metrics_path.read_text(encoding="utf-8"))
                    flat = flatten(m)
                    flat["seed"] = seed
                    flat["target_project"] = p["target"]
                    flat["variant"] = tag
                    rows.append(flat)

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        out_csv = out_root / "summary_feature_key_ablation_all_seeds.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved summary to {out_csv}")

        # formatted core metrics
        core_fields = [
            "target_project",
            "seed",
            "variant",
            "roc_auc",
            "pr_auc",
            "final_on_target.f1",
            "final_on_target.mcc",
        ]
        core_csv = out_root / "summary_feature_key_ablation_all_seeds_core.csv"
        with core_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=core_fields)
            w.writeheader()
            for r in rows:
                row = {k: r.get(k) for k in core_fields}
                w.writerow(row)
        print(f"Saved core summary to {core_csv}")
    else:
        print("No metrics found to summarize.")


if __name__ == "__main__":
    main()
