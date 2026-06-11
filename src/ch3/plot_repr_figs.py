"""
Quick plotting helpers for representations: PCA scatter (H_file/Z_sh/Z_pr) and alpha vs num_blocks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_meta(meta_path: Path):
    projects = []
    num_blocks = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            projects.append(obj["project"])
            num_blocks.append(int(obj["num_blocks"]))
    return np.array(projects), np.array(num_blocks)


def scatter_pca(x, labels, title, out_path, target_project: str | None = None):
    pca = PCA(n_components=2)
    z = pca.fit_transform(x)
    uniq = np.unique(labels)
    plt.figure(figsize=(6, 5))
    for u in uniq:
        mask = labels == u
        plt.scatter(z[mask, 0], z[mask, 1], s=5, alpha=0.6, label=u)
    plt.legend(markerscale=3, fontsize=8)
    if target_project:
        plt.title(f"{title} (target={target_project})")
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_alpha_numblocks(alpha, blk_ptr, num_blocks, out_path, target_project: str | None = None):
    ent_list = []
    k_list = []
    for i in range(len(blk_ptr) - 1):
        start = int(blk_ptr[i])
        end = int(blk_ptr[i + 1])
        seg = alpha[start:end]
        k = end - start
        k_list.append(k)
        if k > 0:
            eps = 1e-8
            ent = -np.sum(seg * np.log(seg + eps))
        else:
            ent = 0.0
        ent_list.append(ent)
    plt.figure(figsize=(6, 5))
    plt.scatter(num_blocks, ent_list, s=4, alpha=0.4)
    # add theoretical upper bound y = log K
    k_unique = sorted(set(k_list))
    if k_unique:
        xs = np.array(k_unique, dtype=float)
        plt.plot(xs, np.log(xs), color="red", linewidth=1.0, alpha=0.7, label="log(K) upper bound")
        plt.legend(fontsize=8)
    plt.xlabel("num_blocks per file (K)")
    plt.ylabel("alpha entropy")
    if target_project:
        plt.title(f"alpha entropy vs num_blocks (target={target_project})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def alpha_diagnostics(alpha, blk_ptr, out_dir: Path, target_project: str | None = None):
    """Draw alpha-related reference plots: entropy histogram, top-k coverage vs K (binned), normalized entropy vs K (binned)."""
    eps = 1e-8
    records = []
    for i in range(len(blk_ptr) - 1):
        start = int(blk_ptr[i])
        end = int(blk_ptr[i + 1])
        seg = alpha[start:end]
        k = end - start
        if k == 0:
            continue
        ent = -np.sum(seg * np.log(seg + eps))
        norm_ent = ent / np.log(k) if k > 1 else 0.0
        seg_sorted = np.sort(seg)[::-1]
        top1 = seg_sorted[0]
        top3 = seg_sorted[: min(3, k)].sum()
        top5 = seg_sorted[: min(5, k)].sum()
        records.append((k, top1, top3, top5, norm_ent))

    if not records:
        return

    k_arr, top1_arr, top3_arr, top5_arr, norm_ent_arr = [np.array(x) for x in zip(*records)]

    # 1) Normalized entropy histogram
    plt.figure(figsize=(6, 4))
    plt.hist(norm_ent_arr, bins=40, color="#4c72b0", alpha=0.8)
    plt.xlabel("H(alpha)/log(K)")
    plt.ylabel("count")
    title = "Normalized attention entropy"
    if target_project:
        title = f"{title} (target={target_project})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_norm_entropy_hist.png", dpi=200)
    plt.close()

    # 2) Mean top1/3/5 vs K (binned)
    max_k = int(k_arr.max())
    xs = np.arange(1, max_k + 1)
    mean_top1 = []
    mean_top3 = []
    mean_top5 = []
    mean_norm_ent = []
    for k in xs:
        mask = k_arr == k
        if not mask.any():
            mean_top1.append(np.nan)
            mean_top3.append(np.nan)
            mean_top5.append(np.nan)
            mean_norm_ent.append(np.nan)
            continue
        mean_top1.append(top1_arr[mask].mean())
        mean_top3.append(top3_arr[mask].mean())
        mean_top5.append(top5_arr[mask].mean())
        mean_norm_ent.append(norm_ent_arr[mask].mean())

    plt.figure(figsize=(7, 4))
    plt.plot(xs, mean_top1, label="top1")
    plt.plot(xs, mean_top3, label="top3")
    plt.plot(xs, mean_top5, label="top5")
    plt.xscale("log")
    plt.xlabel("K (blocks per file, log scale)")
    plt.ylabel("mean cumulative weight")
    title = "Alpha coverage vs K (log-x)"
    if target_project:
        title = f"{title} (target={target_project})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_topk_vs_k.png", dpi=200)
    plt.close()

    # 3) Normalized entropy vs K (log-x)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, mean_norm_ent, label="H/log K")
    plt.xscale("log")
    plt.xlabel("K (blocks per file, log scale)")
    plt.ylabel("mean H/log K")
    title = "Normalized entropy vs K (log-x)"
    if target_project:
        title = f"{title} (target={target_project})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_norm_entropy_vs_k.png", dpi=200)
    plt.close()


def _parse_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def _filter_alpha(alpha: np.ndarray, blk_ptr: np.ndarray, keep_idx: np.ndarray):
    # Build a filtered alpha and blk_ptr for selected file indices.
    new_alpha = []
    new_blk_ptr = [0]
    for i in keep_idx:
        start = int(blk_ptr[i])
        end = int(blk_ptr[i + 1])
        seg = alpha[start:end]
        new_alpha.append(seg)
        new_blk_ptr.append(new_blk_ptr[-1] + len(seg))
    if new_alpha:
        new_alpha = np.concatenate(new_alpha, axis=0)
    else:
        new_alpha = np.zeros((0,), dtype=alpha.dtype)
    return new_alpha, np.array(new_blk_ptr, dtype=blk_ptr.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="Directory containing repr.npz and meta.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-projects", default=None, help="comma-separated project names to keep")
    ap.add_argument("--exclude-projects", default=None, help="comma-separated project names to drop")
    ap.add_argument("--target-project", default=None, help="target project name for plot titles")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dump_dir / "repr.npz")
    projects, num_blocks = load_meta(dump_dir / "meta.jsonl")
    project_ids = projects

    include_projects = set(_parse_list(args.include_projects))
    exclude_projects = set(_parse_list(args.exclude_projects))
    if include_projects:
        keep_mask = np.array([p in include_projects for p in projects], dtype=bool)
    else:
        keep_mask = np.ones(len(projects), dtype=bool)
    if exclude_projects:
        keep_mask &= np.array([p not in exclude_projects for p in projects], dtype=bool)

    if not keep_mask.any():
        raise ValueError("No samples left after include/exclude project filtering.")

    keep_idx = np.where(keep_mask)[0]

    for key, title in [("H_file", "PCA H_file"), ("Z_pr", "PCA Z_pr"), ("Z_sh", "PCA Z_sh")]:
        if key in data:
            scatter_pca(
                data[key][keep_idx],
                project_ids[keep_idx],
                title,
                out_dir / f"{key.lower()}_pca.png",
                target_project=args.target_project,
            )

    if "alpha_values" in data and "blk_ptr" in data:
        alpha_vals, blk_ptr = _filter_alpha(data["alpha_values"], data["blk_ptr"], keep_idx)
        plot_alpha_numblocks(
            alpha_vals,
            blk_ptr,
            num_blocks[keep_idx],
            out_dir / "alpha_vs_numblocks.png",
            target_project=args.target_project,
        )
        alpha_diagnostics(alpha_vals, blk_ptr, out_dir, target_project=args.target_project)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
