"""
Compute CORAL distance across projects for representations.

Outputs mean/min/max pairwise CORAL for H_file, Z_sh, Z_pr.
CORAL(A,B) = ||Cov(A) - Cov(B)||_F^2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def coral_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    diff = ca - cb
    return float(np.linalg.norm(diff, ord="fro") ** 2)


def compute_for_repr(x: np.ndarray, proj: np.ndarray):
    projs = np.unique(proj)
    n = len(projs)
    dists = []
    matrix = np.zeros((n, n), dtype=float)
    for i, pi in enumerate(projs):
        ai = x[proj == pi]
        for j, pj in enumerate(projs):
            if j <= i:
                continue
            bj = x[proj == pj]
            dist = coral_distance(ai, bj)
            matrix[i, j] = matrix[j, i] = dist
            dists.append(dist)
    stats = {
        "mean": float(np.mean(dists)) if dists else 0.0,
        "min": float(np.min(dists)) if dists else 0.0,
        "max": float(np.max(dists)) if dists else 0.0,
    }
    return stats, matrix.tolist(), projs.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr-npz", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.repr_npz)
    proj = data["proj_id"]

    out = {}
    matrices = {}
    for key in ["H_file", "Z_sh", "Z_pr"]:
        if key not in data:
            continue
        stats, mat, labels = compute_for_repr(data[key], proj)
        out[key] = stats
        matrices[key] = {"matrix": mat, "labels": labels}

    out_path = Path(args.out) if args.out else Path(args.repr_npz).with_name("coral.json")
    out_path.write_text(json.dumps({"stats": out, "matrices": matrices}, indent=2), encoding="utf-8")
    print(f"Saved CORAL metrics to {out_path}")


if __name__ == "__main__":
    main()
