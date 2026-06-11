"""
Baseline runner for TCA / SA / CORAL on cached repr.npz.

Loads source/target features via FeatureCache + CPDP splits,
applies the chosen transform, trains LogisticRegression on source,
evaluates on target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.feature_cache import FeatureCache
from src.utils.cpdp_tasks import build_cpdp_task
from src.ch4.baselines.tca import TCA
from src.ch4.baselines.sa import SA
from src.ch4.baselines.coral import CORAL


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.5):
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            accuracy_score,
            matthews_corrcoef,
        )
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for metrics. Run: pip install scikit-learn") from exc

    y_pred = (y_score >= thresh).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    pf = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "pf": pf,
        "threshold": float(thresh),
    }
    return out


def best_threshold(scores: np.ndarray, labels: np.ndarray, mode: str = "mcc"):
    thresholds = np.unique(scores)
    best_t = 0.5
    best_val = -1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = float(((pred == 1) & (labels == 1)).sum())
        tn = float(((pred == 0) & (labels == 0)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        if mode == "f1":
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            val = ((tp * tn - fp * fn) / np.sqrt(denom + 1e-12)) if denom > 0 else 0.0
        if val > best_val:
            best_val = val
            best_t = float(t)
    return best_t, float(best_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--source-projects", required=True)
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--feature-key", default="H_file", help="H_file, Z_sh, Z_pr")
    ap.add_argument("--method", required=True, choices=["tca", "sa", "coral"])
    ap.add_argument("--n-components", type=int, default=20)
    ap.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-intra", type=int, default=0)
    ap.add_argument("--standardize", type=int, default=1)
    ap.add_argument("--threshold-mode", default="mcc", choices=["mcc", "f1"])
    ap.add_argument("--lr-max-iter", type=int, default=5000)
    ap.add_argument("--lr-solver", default="lbfgs")
    ap.add_argument("--lr-class-weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)

    splits = build_cpdp_task(
        dump_dir / "meta.jsonl",
        source_projects=args.source_projects.split(","),
        target_project=args.target_project,
        train_ratio=0.8,
        seed=args.seed,
        dedup_intra_project=bool(args.dedup_intra),
    )

    cache = FeatureCache(dump_dir)
    Xs_tr, ys_tr, _ = cache.get_features(splits.source_train.uids, key=args.feature_key)
    Xs_val, ys_val, _ = cache.get_features(splits.source_val.uids, key=args.feature_key)
    Xt, yt, _ = cache.get_features(splits.target_test.uids, key=args.feature_key)

    Xs_tr = Xs_tr.numpy()
    Xs_val = Xs_val.numpy()
    Xt = Xt.numpy()
    ys_tr = ys_tr.numpy().astype(int)
    ys_val = ys_val.numpy().astype(int)
    yt = yt.numpy().astype(int)

    if args.standardize:
        try:
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for standardization. Run: pip install scikit-learn") from exc
        scaler = StandardScaler()
        X_all = np.vstack([Xs_tr, Xs_val, Xt])
        scaler.fit(X_all)
        Xs_tr = scaler.transform(Xs_tr)
        Xs_val = scaler.transform(Xs_val)
        Xt = scaler.transform(Xt)

    if args.method == "tca":
        transformer = TCA(n_components=args.n_components, kernel=args.kernel, gamma=args.gamma, mu=args.mu)
    elif args.method == "sa":
        transformer = SA(n_components=args.n_components, random_state=args.seed)
    elif args.method == "coral":
        transformer = CORAL()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    Xs_new, Xt_new = transformer.fit_transform(np.vstack([Xs_tr, Xs_val]), Xt)
    Xs_tr_new = Xs_new[: len(Xs_tr)]
    Xs_val_new = Xs_new[len(Xs_tr) :]

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for LogisticRegression. Run: pip install scikit-learn") from exc

    class_weight = None if args.lr_class_weight == "none" else "balanced"
    clf = LogisticRegression(max_iter=args.lr_max_iter, n_jobs=-1, solver=args.lr_solver, class_weight=class_weight)
    clf.fit(Xs_tr_new, ys_tr)
    val_score = clf.predict_proba(Xs_val_new)[:, 1] if len(Xs_val_new) > 0 else np.array([])
    if val_score.size > 0:
        best_t, best_val = best_threshold(val_score, ys_val, mode=args.threshold_mode)
    else:
        best_t, best_val = 0.5, 0.0
    y_score = clf.predict_proba(Xt_new)[:, 1]

    default_metrics = compute_metrics(yt, y_score, thresh=0.5)
    transfer_metrics = compute_metrics(yt, y_score, thresh=best_t)
    metrics = {
        "roc_auc": default_metrics["roc_auc"],
        "pr_auc": default_metrics["pr_auc"],
        "default_0p5": default_metrics,
        "transfer_source": {
            **transfer_metrics,
            "threshold": float(best_t),
            "threshold_from": f"source_val_{args.threshold_mode}",
            "best_val": float(best_val),
        },
    }
    metrics.update(
        {
            "method": args.method,
            "feature_key": args.feature_key,
            "source_projects": args.source_projects.split(","),
            "target_project": args.target_project,
            "n_components": args.n_components,
            "kernel": args.kernel,
            "gamma": args.gamma,
            "mu": args.mu,
            "lr_max_iter": args.lr_max_iter,
            "lr_solver": args.lr_solver,
            "lr_class_weight": args.lr_class_weight,
            "threshold_mode": args.threshold_mode,
        }
    )

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
