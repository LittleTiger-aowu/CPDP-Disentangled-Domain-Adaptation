"""
Run CK-metrics baselines (LR/RF) for class-level CSVs in zero-shot (source-only) setting.

Example:
  python src/ch3/run_ck_lr_rf_baselines.py ^
    --csv-dir Dataset/BugPrediction/csv-class ^
    --target-project Eclipse_JDT_Core ^
    --out-dir outputs/ch3_ck_baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


CK_FEATURES = ["WMC", "DIT", "NOC", "CBO", "RFC", "LCOM5"]


def parse_project_version(csv_path: Path) -> tuple[str, str]:
    stem = csv_path.stem
    for suffix in ("-Unified", "-unified"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "-" in stem:
        project, version = stem.rsplit("-", 1)
    else:
        project, version = stem, "unknown"
    return project, version


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y_true)) < 2:
        return None, None
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def bin_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= thresh).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = float((tp + tn) / max(1, tp + tn + fp + fn))
    pf = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom > 0:
        mcc = float((tp * tn - fp * fn) / np.sqrt(denom))
    else:
        mcc = 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "acc": acc, "pf": pf, "mcc": mcc}


def load_csvs(csv_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        project, version = parse_project_version(csv_path)
        df = pd.read_csv(csv_path)
        df["project"] = project
        df["version"] = version
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="Directory containing *-Unified.csv files")
    ap.add_argument("--target-project", default=None, help="Project name (e.g., Eclipse_JDT_Core)")
    ap.add_argument(
        "--all-projects",
        type=int,
        default=0,
        help="1: run zero-shot for every project found in csv-dir",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(csv_dir)
    if "bug" not in df.columns:
        raise ValueError("Missing 'bug' column in CSVs.")

    missing = [c for c in CK_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CK feature columns: {missing}")

    df = df.copy()
    bug_raw = pd.to_numeric(df["bug"], errors="coerce")
    # Drop rows where bug label is missing (do not include in testing)
    df = df[~bug_raw.isna()].copy()
    df["bug"] = (bug_raw[~bug_raw.isna()] > 0).astype(int)
    for c in CK_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    projects = sorted(df["project"].unique().tolist())
    if args.all_projects:
        targets = projects
    else:
        if not args.target_project:
            raise ValueError("Either --target-project or --all-projects 1 is required.")
        targets = [args.target_project]

    all_results: list[dict] = []
    for target in targets:
        train_df = df[df["project"] != target].reset_index(drop=True)
        test_df = df[df["project"] == target].reset_index(drop=True)
        if train_df.empty:
            raise ValueError(f"No source samples found after excluding target {target}")
        if test_df.empty:
            raise ValueError(f"No target samples found for {target}")

        X_train = train_df[CK_FEATURES].to_numpy(dtype=np.float32)
        y_train = train_df["bug"].to_numpy(dtype=np.int64)
        X_test = test_df[CK_FEATURES].to_numpy(dtype=np.float32)
        y_test = test_df["bug"].to_numpy(dtype=np.int64)

        results = []

        lr = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=args.seed),
        )
        lr.fit(X_train, y_train)
        lr_prob = lr.predict_proba(X_test)[:, 1]
        roc_auc, pr_auc = safe_auc(y_test, lr_prob)
        row = {
            "model": "CK+LR",
            "target_project": target,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_samples": int(len(y_test)),
            "pos_ratio": float(y_test.mean()) if len(y_test) else 0.0,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "use_bug_file": None,
        }
        row.update(bin_metrics(y_test, lr_prob))
        results.append(row)

        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        roc_auc, pr_auc = safe_auc(y_test, rf_prob)
        row = {
            "model": "CK+RF",
            "target_project": target,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_samples": int(len(y_test)),
            "pos_ratio": float(y_test.mean()) if len(y_test) else 0.0,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "use_bug_file": None,
        }
        row.update(bin_metrics(y_test, rf_prob))
        results.append(row)

        json_path = out_dir / f"ck_baselines_{target}.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        all_results.extend(results)

    csv_path = out_dir / "ck_baselines_summary.csv"
    fieldnames = [
        "model",
        "target_project",
        "n_train",
        "n_test",
        "n_samples",
        "pos_ratio",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "acc",
        "pf",
        "mcc",
        "use_bug_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved CK baselines summary to {csv_path}")


if __name__ == "__main__":
    main()
