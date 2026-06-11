"""
Plot t-SNE for representations from dump_representations.py.

Example:
  python src/ch3/plot_tsne.py --dump-dir outputs/ch3_direct_transfer/target_Mylyn-3.1/dump --out-dir outputs/ch3_direct_transfer/target_Mylyn-3.1/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_meta(meta_path: Path):
    projects = []
    y = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            projects.append(obj["project"])
            y.append(int(obj["y"]))
    return np.array(projects), np.array(y)


def plot_tsne(x, labels, title, out_path, seed, perplexity, learning_rate, target_project: str | None = None):
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for t-SNE. Run: pip install scikit-learn") from exc

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=seed,
    )
    z = tsne.fit_transform(x)
    uniq = np.unique(labels)
    plt.figure(figsize=(6, 5))
    for u in uniq:
        mask = labels == u
        plt.scatter(z[mask, 0], z[mask, 1], s=5, alpha=0.6, label=str(u))
    plt.legend(markerscale=3, fontsize=8)
    if target_project:
        plt.title(f"{title} (target={target_project})")
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _parse_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keys", default="H_file,Z_sh,Z_pr", help="comma-separated keys from repr.npz")
    ap.add_argument("--color-by", choices=["project", "bug"], default="project")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--learning-rate", type=float, default=200.0)
    ap.add_argument("--include-projects", default=None, help="comma-separated project names to keep")
    ap.add_argument("--exclude-projects", default=None, help="comma-separated project names to drop")
    ap.add_argument("--target-project", default=None, help="target project name for plot titles")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dump_dir / "repr.npz")
    projects, y = load_meta(dump_dir / "meta.jsonl")

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

    if args.color_by == "project":
        labels = projects
    else:
        labels = y.astype(np.int64)

    labels = labels[keep_mask]
    n = len(labels)
    if args.max_samples and n > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.max_samples, replace=False)
    else:
        idx = np.arange(n)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    for key in keys:
        if key not in data:
            continue
        x = data[key][keep_mask][idx]
        lbl = labels[idx]
        title = f"t-SNE {key} by {args.color_by}"
        out_path = out_dir / f"tsne_{key}_by_{args.color_by}.png"
        plot_tsne(
            x,
            lbl,
            title,
            out_path,
            seed=args.seed,
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            target_project=args.target_project,
        )

    print(f"Saved t-SNE plots to {out_dir}")


if __name__ == "__main__":
    main()
