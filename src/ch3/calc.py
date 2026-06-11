import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score
from pathlib import Path
import re

# ============ 配置区域 ============
ROOT_DIR = r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_ablation"
OUT_CSV = r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_ablation_summary.csv"
# =================================

TARGET_RE = re.compile(r"target_(.+)")

def parse_variant_target(path: Path):
    parts = path.parts
    variant = None
    target = None
    for i, p in enumerate(parts):
        if p.startswith("target_"):
            target = p.replace("target_", "")
            if i > 0:
                variant = parts[i - 1]
            break
    return variant, target

def metrics_at_threshold(y_true, y_prob, t):
    y_pred = (y_prob >= t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return prec, rec, f1, mcc

def calc_best_metrics(file_path, n_thresholds=100):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return {"file": str(file_path), "error": "file not found"}

    try:
        y_true = df.iloc[:, 1].astype(int).values
        y_prob = df.iloc[:, 2].values
    except Exception as e:
        return {"file": str(file_path), "error": f"parse error: {e}"}

    n_samples = len(y_true)
    n_pos = int(np.sum(y_true))
    pos_ratio = n_pos / n_samples if n_samples else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    # 默认阈值 0.5
    default_prec, default_rec, default_f1, default_mcc = metrics_at_threshold(y_true, y_prob, 0.5)

    # 搜索最佳阈值
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1, best_f1_thresh = -1.0, 0.5
    best_mcc, best_mcc_thresh = -1.0, 0.5

    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        mcc = matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_f1_thresh = f1, t
        if mcc > best_mcc:
            best_mcc, best_mcc_thresh = mcc, t

    # 计算“最佳阈值”下的完整指标
    bf_prec, bf_rec, bf_f1, bf_mcc = metrics_at_threshold(y_true, y_prob, best_f1_thresh)
    bm_prec, bm_rec, bm_f1, bm_mcc = metrics_at_threshold(y_true, y_prob, best_mcc_thresh)

    variant, target = parse_variant_target(Path(file_path))
    return {
        "variant": variant,
        "target_project": target,
        "file": str(file_path),
        "n_samples": n_samples,
        "pos_ratio": pos_ratio,
        "roc_auc": auc,

        "default_prec@0.5": default_prec,
        "default_rec@0.5": default_rec,
        "default_f1@0.5": default_f1,
        "default_mcc@0.5": default_mcc,

        "best_f1": best_f1,
        "best_f1_thresh": round(best_f1_thresh, 4),
        "best_f1_prec": bf_prec,
        "best_f1_rec": bf_rec,
        "best_f1_mcc": bf_mcc,

        "best_mcc": best_mcc,
        "best_mcc_thresh": round(best_mcc_thresh, 4),
        "best_mcc_prec": bm_prec,
        "best_mcc_rec": bm_rec,
        "best_mcc_f1": bm_f1,
    }

def main():
    preds = list(Path(ROOT_DIR).rglob("predictions.csv"))
    if not preds:
        print("❌ 未找到 predictions.csv")
        return

    rows = [calc_best_metrics(p) for p in preds]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(df)
    print(f"\n✅ 汇总完成: {OUT_CSV}")

if __name__ == "__main__":
    main()
