"""
Run minimal CPDP diagnosis suite:
  A) domain probe pre (source vs target) on dumped repr
  B) sweep lambda_dom values for DANN training
  C) compute effort-aware metrics on target predictions
     + domain probe post using dom_head

Outputs:
  - per-run outputs under --output-root/target_<target>_ld<lambda>
  - diag_summary.csv under --output-root
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run_cmd(cmd):
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def load_projects(meta_path: Path):
    projects = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            projects.append(obj["project"])
    return projects


def domain_probe_pre(dump_dir: Path, target_project: str, key: str, max_samples: int, seed: int) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for domain probe pre. Run: pip install scikit-learn") from exc

    data = np.load(dump_dir / "repr.npz")
    if key not in data:
        raise ValueError(f"repr.npz missing {key}")
    X = data[key]
    projects = load_projects(dump_dir / "meta.jsonl")
    y = np.array([1 if p == target_project else 0 for p in projects], dtype=np.int64)
    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return float(accuracy_score(y_te, pred))


def effort_metrics(pred_csv: Path, frac: float = 0.2):
    rows = []
    with pred_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    if not rows:
        return {"precision_at_effort": 0.0, "recall_at_effort": 0.0}
    y_true = np.array([int(float(r["y_true"])) for r in rows], dtype=np.int64)
    y_score = np.array([float(r["y_score"]) for r in rows], dtype=np.float32)
    order = np.argsort(-y_score)
    k = max(1, int(len(rows) * frac))
    top = y_true[order[:k]]
    tp = int(top.sum())
    pos_total = int(y_true.sum())
    recall = float(tp / pos_total) if pos_total > 0 else 0.0
    precision = float(tp / k) if k > 0 else 0.0
    return {"precision_at_effort": precision, "recall_at_effort": recall}


def domain_probe_post(dump_dir: Path, run_dir: Path) -> float:
    import torch
    from src.data.feature_cache import FeatureCache
    from src.ch4.train_cpdp_adapt_cached import Heads

    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    src_uids = splits["source_train"] + splits["source_val"]
    tgt_uids = splits["target_test"]
    if not src_uids or not tgt_uids:
        return 0.0
    cache = FeatureCache(dump_dir)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    cfg = ckpt.get("config", {})
    feature_key = cfg.get("feature_key", "H_file")
    H_src, _, _ = cache.get_features(src_uids, key=feature_key)
    H_tgt, _, _ = cache.get_features(tgt_uids, key=feature_key)
    heads = Heads(
        d_in=H_src.shape[1],
        d_sh=128,
        dom_hidden=64,
        use_bottleneck=bool(cfg.get("use_bottleneck", 0)),
        bottleneck_dim=int(cfg.get("bottleneck_dim", 128)),
        bottleneck_hidden=int(cfg.get("bottleneck_hidden", 0)),
        bottleneck_dropout=float(cfg.get("bottleneck_dropout", 0.0)),
    )
    heads.load_state_dict(ckpt["state_dict"])
    heads.eval()
    with torch.no_grad():
        logits_src = heads(H_src, lambd_grl=0.0)["logits_dom"]
        logits_tgt = heads(H_tgt, lambd_grl=0.0)["logits_dom"]
        preds = torch.argmax(torch.cat([logits_src, logits_tgt], 0), dim=1)
        labels = torch.cat(
            [
                torch.zeros(len(logits_src), dtype=torch.long),
                torch.ones(len(logits_tgt), dtype=torch.long),
            ],
            0,
        )
        acc = (preds == labels).float().mean().item()
    return float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--source-projects", required=True)
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--lambda-dom-list", default="0.1,0.3,0.5")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grl-gamma", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-key", default="Z_sh")
    ap.add_argument("--probe-max-samples", type=int, default=5000)
    ap.add_argument("--effort-frac", type=float, default=0.2)
    ap.add_argument("--feature-key", default="H_file")
    ap.add_argument("--use-bottleneck", type=int, default=0)
    ap.add_argument("--bottleneck-dim", type=int, default=128)
    ap.add_argument("--bottleneck-hidden", type=int, default=0)
    ap.add_argument("--bottleneck-dropout", type=float, default=0.0)
    ap.add_argument("--am-margin", type=float, default=0.0)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pre_acc = domain_probe_pre(dump_dir, args.target_project, args.probe_key, args.probe_max_samples, args.seed)

    lambda_list = [float(x) for x in args.lambda_dom_list.split(",") if x.strip()]
    train_py = Path(__file__).parent / "train_cpdp_adapt_cached.py"

    rows = []
    for ld in lambda_list:
        run_dir = out_root / f"target_{args.target_project}_ld{ld}"
        cmd = [
            "python",
            str(train_py),
            "--dump-dir",
            str(dump_dir),
            "--source-projects",
            args.source_projects,
            "--target-project",
            args.target_project,
            "--output-dir",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lambda-dom",
            str(ld),
            "--grl-gamma",
            str(args.grl_gamma),
            "--seed",
            str(args.seed),
            "--feature-key",
            str(args.feature_key),
            "--use-bottleneck",
            str(args.use_bottleneck),
            "--bottleneck-dim",
            str(args.bottleneck_dim),
            "--bottleneck-hidden",
            str(args.bottleneck_hidden),
            "--bottleneck-dropout",
            str(args.bottleneck_dropout),
            "--am-margin",
            str(args.am_margin),
        ]
        run_cmd(cmd)

        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        effort = effort_metrics(run_dir / "predictions.csv", frac=args.effort_frac)
        post_acc = domain_probe_post(dump_dir, run_dir)

        row = {
            "lambda_dom": ld,
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "best_mcc_val": metrics.get("best_mcc_val"),
            "best_threshold": metrics.get("best_threshold"),
            "threshold_from": metrics.get("threshold_from"),
            "precision_at_effort": effort["precision_at_effort"],
            "recall_at_effort": effort["recall_at_effort"],
            "domain_acc_pre": pre_acc,
            "domain_acc_post": post_acc,
            "target_project": args.target_project,
        }
        rows.append(row)

    summary_csv = out_root / "diag_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
