"""
Collect ch3 run results into a CSV/MD summary (and optional simple plots).

Looks under runs_root/* for repr_metrics.json, domain_probe.json, coral.json, rank.json.
Outputs summary CSV/MD and, if requested, bar charts for key metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt


def load_json_safe(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--plot-prefix", default=None, help="If set, saves bar plots with this prefix (e.g., runs_summary)")
    args = ap.parse_args()

    root = Path(args.runs_root)
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        metrics = load_json_safe(run_dir / "repr_metrics.json")
        probe = load_json_safe(run_dir / "domain_probe.json")
        coral = load_json_safe(run_dir / "coral.json")
        rank = load_json_safe(run_dir / "rank.json")
        row = {"run": name}
        if metrics:
            row.update(
                {
                    "s_corr": metrics.get("s_corr"),
                    "sil_h_file": metrics.get("silhouette_h_file"),
                    "sil_z_sh": metrics.get("silhouette_z_sh"),
                    "sil_z_pr": metrics.get("silhouette_z_pr"),
                    "alpha_entropy": metrics.get("alpha_entropy_mean"),
                    "alpha_max_mean": metrics.get("alpha_max_mean"),
                }
            )
        if probe:
            row.update(
                {
                    "ce_h": probe.get("h_file_ce_test"),
                    "ce_z_sh": probe.get("z_sh_ce_test"),
                    "ce_z_pr": probe.get("z_pr_ce_test"),
                }
            )
        if coral and "stats" in coral:
            row["coral_z_sh"] = coral["stats"].get("Z_sh", {}).get("mean")
            row["coral_z_pr"] = coral["stats"].get("Z_pr", {}).get("mean")
        if rank:
            row["rank_z_sh"] = rank.get("Z_sh")
            row["rank_z_pr"] = rank.get("Z_pr")
        rows.append(row)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join([" --- "]*len(fieldnames)) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(k,"")) for k in fieldnames) + " |")
    Path(args.out_md).write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved summary to {args.out_csv} and {args.out_md}")

    if args.plot_prefix:
        def bar(metric, title):
            xs = [r["run"] for r in rows if metric in r]
            ys = [r.get(metric, 0) for r in rows if metric in r]
            if not xs:
                return
            plt.figure(figsize=(max(6, len(xs)*1.2), 4))
            plt.bar(xs, ys)
            plt.ylabel(metric)
            plt.title(title)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(f"{args.plot_prefix}_{metric}.png", dpi=200)
            plt.close()

        bar("s_corr", "S_corr (||C||_F)")
        bar("sil_z_sh", "Silhouette Z_sh (project)")
        bar("sil_z_pr", "Silhouette Z_pr (project)")
        bar("ce_z_sh", "Domain CE Z_sh")
        bar("ce_z_pr", "Domain CE Z_pr")
        bar("alpha_entropy", "Alpha entropy mean")
        bar("coral_z_sh", "CORAL mean Z_sh")
        bar("coral_z_pr", "CORAL mean Z_pr")
        print(f"Saved plots with prefix {args.plot_prefix}_*.png")


if __name__ == "__main__":
    main()
