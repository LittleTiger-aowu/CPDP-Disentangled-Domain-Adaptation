"""
Compute representation metrics from dump_representations outputs.

Example:
  python src/ch3/compute_repr_metrics.py --dump-dir outputs/ch3_dump/run1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def alpha_entropy(alpha: np.ndarray, start: int, end: int) -> float:
    seg = alpha[start:end]
    if seg.size == 0:
        return 0.0
    eps = 1e-8
    return float(-np.sum(seg * np.log(seg + eps)))


def silhouette_score_safe(x: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return float("nan")
    if x.shape[0] < 2:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(x, labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    npz_path = dump_dir / "repr.npz"
    data = np.load(npz_path)
    h_file = data["H_file"]
    z_sh = data["Z_sh"]
    z_pr = data["Z_pr"]
    proj_id = data["proj_id"]
    alpha_values = data["alpha_values"]
    blk_ptr = data["blk_ptr"]

    # alpha-derived stats
    alpha_max_list = []
    alpha_top5_sum = []
    alpha_norm_entropy = []
    k_list = []
    for i in range(len(blk_ptr) - 1):
        start = int(blk_ptr[i])
        end = int(blk_ptr[i + 1])
        k = end - start
        if k <= 0:
            continue
        k_list.append(k)
        seg = alpha_values[start:end]
        alpha_max_list.append(float(seg.max()))
        topk = min(5, seg.size)
        alpha_top5_sum.append(float(np.sort(seg)[-topk:].sum()))
        h = alpha_entropy(seg, 0, seg.size)
        alpha_norm_entropy.append(float(h / np.log(k)))

    # S_corr
    z_sh_norm = z_sh / (np.linalg.norm(z_sh, axis=1, keepdims=True) + 1e-12)
    z_pr_norm = z_pr / (np.linalg.norm(z_pr, axis=1, keepdims=True) + 1e-12)
    batch = max(z_sh_norm.shape[0], 1)
    c = (z_sh_norm.T @ z_pr_norm) / batch
    s_corr = float(np.linalg.norm(c, ord="fro"))

    # alpha entropy
    entropies = []
    for i in range(len(blk_ptr) - 1):
        start = int(blk_ptr[i])
        end = int(blk_ptr[i + 1])
        entropies.append(alpha_entropy(alpha_values, start, end))
    alpha_entropy_mean = float(np.mean(entropies)) if entropies else 0.0

    metrics = {
        "n_files": int(h_file.shape[0]),
        "n_projects": int(len(np.unique(proj_id))),
        "s_corr": s_corr,
        "silhouette_h_file": silhouette_score_safe(h_file, proj_id),
        "silhouette_z_pr": silhouette_score_safe(z_pr, proj_id),
        "silhouette_z_sh": silhouette_score_safe(z_sh, proj_id),
        "alpha_entropy_mean": alpha_entropy_mean,
        "alpha_max_mean": float(np.mean(alpha_max_list)) if alpha_max_list else 0.0,
        "alpha_max_p50": float(np.percentile(alpha_max_list, 50)) if alpha_max_list else 0.0,
        "alpha_max_p90": float(np.percentile(alpha_max_list, 90)) if alpha_max_list else 0.0,
        "alpha_top5_sum_mean": float(np.mean(alpha_top5_sum)) if alpha_top5_sum else 0.0,
        "alpha_top5_sum_p50": float(np.percentile(alpha_top5_sum, 50)) if alpha_top5_sum else 0.0,
        "alpha_top5_sum_p90": float(np.percentile(alpha_top5_sum, 90)) if alpha_top5_sum else 0.0,
        "alpha_norm_entropy_mean": float(np.mean(alpha_norm_entropy)) if alpha_norm_entropy else 0.0,
        "alpha_norm_entropy_p50": float(np.percentile(alpha_norm_entropy, 50)) if alpha_norm_entropy else 0.0,
        "alpha_norm_entropy_p90": float(np.percentile(alpha_norm_entropy, 90)) if alpha_norm_entropy else 0.0,
        "p_k_le_1": float(np.mean(np.array(k_list) <= 1)) if k_list else 0.0,
        "p_k_le_3": float(np.mean(np.array(k_list) <= 3)) if k_list else 0.0,
        "p_k_le_5": float(np.mean(np.array(k_list) <= 5)) if k_list else 0.0,
    }

    out_path = Path(args.out) if args.out else dump_dir / "repr_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
