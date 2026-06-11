"""
Compute effective rank (entropy of singular values) for H_file, Z_sh, Z_pr.

effective_rank(X) = exp( H(p) ), p = singular_values / sum(singular_values)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, v = np.linalg.svd(x, full_matrices=False)
    if s.size == 0:
        return 0.0
    p = s / (s.sum() + 1e-12)
    h = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr-npz", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.repr_npz)

    out = {}
    for key in ["H_file", "Z_sh", "Z_pr"]:
        if key not in data:
            continue
        out[key] = effective_rank(data[key])

    out_path = Path(args.out) if args.out else Path(args.repr_npz).with_name("rank.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved effective rank to {out_path}")


if __name__ == "__main__":
    main()
