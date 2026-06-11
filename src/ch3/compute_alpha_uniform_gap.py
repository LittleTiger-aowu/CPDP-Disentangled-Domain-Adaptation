"""
Compute how far attention weights are from uniform:
  - L1 distance to uniform (Du)
  - KL(alpha || uniform)
  - effective blocks Keff = exp(H)
  - Keff / K

Outputs a JSON with mean/p50/p90 stats and a CSV for Du vs K (mean/p50/p90).

Usage:
  python src/ch3/compute_alpha_uniform_gap.py --repr-npz <dump_dir>/repr.npz --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def stats(arr):
    if len(arr) == 0:
        return {"mean": 0, "p50": 0, "p90": 0, "min": 0, "max": 0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr-npz", required=True)
    ap.add_argument("--out", default=None, help="out json (default: repr dir / alpha_uniform_gap.json)")
    ap.add_argument("--csv", default=None, help="optional csv for Du vs K (default: repr dir / alpha_du_vs_k.csv)")
    args = ap.parse_args()

    data = np.load(args.repr_npz)
    alpha = data["alpha_values"]
    blk_ptr = data["blk_ptr"]

    du_list = []
    kl_list = []
    keff_list = []
    keff_over_k_list = []
    k_list = []

    # Per-K buckets for Du stats
    bucket = {}

    eps = 1e-8
    for i in range(len(blk_ptr) - 1):
        s = int(blk_ptr[i])
        e = int(blk_ptr[i + 1])
        seg = alpha[s:e]
        k = e - s
        if k <= 0:
            continue
        uni = 1.0 / k
        # L1 distance to uniform
        du = float(np.abs(seg - uni).mean())
        # KL(alpha || uniform)
        kl = float(np.sum(seg * (np.log(seg + eps) - math.log(uni + eps))))
        # effective blocks
        ent = -np.sum(seg * np.log(seg + eps))
        keff = float(np.exp(ent))
        keff_over_k = keff / k

        du_list.append(du)
        kl_list.append(kl)
        keff_list.append(keff)
        keff_over_k_list.append(keff_over_k)
        k_list.append(k)

        bucket.setdefault(k, []).append(du)

    out = {
        "n_files": len(du_list),
        "du": stats(du_list),
        "kl": stats(kl_list),
        "keff": stats(keff_list),
        "keff_over_k": stats(keff_over_k_list),
    }

    out_path = Path(args.out) if args.out else Path(args.repr_npz).with_name("alpha_uniform_gap.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")

    csv_path = Path(args.csv) if args.csv else Path(args.repr_npz).with_name("alpha_du_vs_k.csv")
    rows = []
    for k, vals in sorted(bucket.items()):
        rows.append(
            {
                "K": k,
                "du_mean": float(np.mean(vals)),
                "du_p50": float(np.percentile(vals, 50)),
                "du_p90": float(np.percentile(vals, 90)),
                "count": len(vals),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
