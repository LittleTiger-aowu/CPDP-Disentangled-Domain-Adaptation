"""
CPDP adversarial adaptation using cached H_file (Method C, DANN style).

Workflow:
  - Build CPDP task (source projects -> target project) from meta.jsonl
  - Load cached H_file via FeatureCache
  - Train lightweight generator head f_sh + bug classifier + domain discriminator (GRL)
  - Loss: L = L_cls(source) + lambda_dom * L_dom(source+target)
  - Class imbalance: source-train uses WeightedRandomSampler; bug loss uses pos_weight
  - Threshold chosen on source val (max MCC); test on target with fixed threshold

Inputs:
  --dump-dir: directory containing repr.npz + meta.jsonl from dump_representations.py
  --source-projects, --target-project (comma-separated)

Outputs (under --output-dir):
  - splits.json (uids)
  - checkpoint.pt (heads only + config + threshold)
  - predictions.csv (uid,y_true,y_score,y_pred)
  - metrics.json
  - train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import time
import torch.nn.functional as F

# ensure project root on path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.cpdp_tasks import build_cpdp_task, save_splits
from src.data.feature_cache import FeatureCache
from src.models.grl import GradientReversal
from src.models.domain_heads import SharedDomainDisc


def build_sampler(y: torch.Tensor):
    # y: 0/1
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    # avoid zero
    pos = max(1, pos)
    neg = max(1, neg)
    w_pos = neg / (pos + neg)
    w_neg = pos / (pos + neg)
    weights = torch.where(y == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float)
    return sampler, pos_weight


def best_threshold(scores: torch.Tensor, labels: torch.Tensor, mode: str = "mcc", beta: float = 1.0):
    # scores: float tensor, labels: 0/1
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    thresholds = np.unique(scores_np)
    best_val = -1.0
    best_t = 0.5
    beta2 = float(beta) ** 2
    for t in thresholds:
        pred = (scores_np >= t).astype(int)
        tp = float(((pred == 1) & (labels_np == 1)).sum())
        tn = float(((pred == 0) & (labels_np == 0)).sum())
        fp = float(((pred == 1) & (labels_np == 0)).sum())
        fn = float(((pred == 0) & (labels_np == 1)).sum())
        if mode in ("f1", "fbeta"):
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = beta2 * prec + rec
            val = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
        else:
            denom_term = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if denom_term <= 0:
                continue
            denom = math.sqrt(denom_term + 1e-12)
            val = (tp * tn - fp * fn) / denom
        if val > best_val:
            best_val = val
            best_t = t
    return float(best_t), float(best_val)


def roc_pr(scores, labels):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except Exception:
        return None, None
    y_true = labels.cpu().numpy()
    y_score = scores.cpu().numpy()
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def bin_metrics(scores, labels, thresh: float):
    """Compute precision/recall/f1/acc at a given threshold."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef

    y_true = labels.cpu().numpy()
    y_pred = (scores.cpu().numpy() >= thresh).astype(int)
    # confusion parts for pf
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    pf = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "pf": pf,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def score_stats(scores: torch.Tensor):
    if scores is None or scores.numel() == 0:
        return {
            "mean": 0.0,
            "var": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    s = scores.detach().cpu().numpy()
    return {
        "mean": float(np.mean(s)),
        "var": float(np.var(s)),
        "p10": float(np.percentile(s, 10)),
        "p25": float(np.percentile(s, 25)),
        "p50": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p90": float(np.percentile(s, 90)),
    }


def percentile_transfer_threshold(source_scores: torch.Tensor, target_scores: torch.Tensor, tau_src: float):
    if source_scores is None or target_scores is None:
        return tau_src, 0.0
    if source_scores.numel() == 0 or target_scores.numel() == 0:
        return tau_src, 0.0
    s = source_scores.detach().cpu().numpy()
    t = target_scores.detach().cpu().numpy()
    q = float(np.mean(s <= tau_src))
    tau_tgt = float(np.quantile(t, q))
    return tau_tgt, q


def should_use_percentile(stats_src: dict, stats_tgt: dict, thr: float = 0.05):
    d75 = abs(stats_src["p75"] - stats_tgt["p75"])
    d90 = abs(stats_src["p90"] - stats_tgt["p90"])
    dmu = abs(stats_src["mean"] - stats_tgt["mean"])
    return (d75 >= thr) or (d90 >= thr) or (dmu >= thr), {"d75": d75, "d90": d90, "dmu": dmu}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    # Optimize T>0 on source-val logits only
    logits = logits.detach().float()
    labels = labels.detach().float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(logits / t, labels)
        loss.backward()
        return loss

    opt.step(_closure)
    t_final = float(torch.exp(log_t).item())
    if not np.isfinite(t_final) or t_final <= 0:
        return 1.0
    return t_final


class Heads(nn.Module):
    """Lightweight bottleneck + generator head + bug classifier + domain discriminator."""

    def __init__(
        self,
        d_in: int,
        d_sh: int,
        dom_hidden: int = 64,
        use_bottleneck: bool = False,
        bottleneck_dim: int = 128,
        bottleneck_hidden: int = 0,
        bottleneck_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_bottleneck = bool(use_bottleneck)
        if self.use_bottleneck:
            if bottleneck_hidden > 0:
                self.bottleneck = nn.Sequential(
                    nn.Linear(d_in, bottleneck_hidden),
                    nn.ReLU(),
                    nn.Dropout(bottleneck_dropout),
                    nn.Linear(bottleneck_hidden, bottleneck_dim),
                )
            else:
                self.bottleneck = nn.Sequential(
                    nn.Linear(d_in, bottleneck_dim),
                    nn.Dropout(bottleneck_dropout),
                )
            f_sh_in = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            f_sh_in = d_in
        self.f_sh = nn.Sequential(nn.Linear(f_sh_in, d_sh), nn.ReLU(), nn.Linear(d_sh, d_sh))
        self.bug_head = nn.Linear(d_sh, 1)
        self.dom_head = SharedDomainDisc(d_sh, num_domains=2, hidden=dom_hidden)
        self.grl = GradientReversal(1.0)

    def forward(self, h_file, lambd_grl=1.0):
        h_in = self.bottleneck(h_file)
        z_sh = self.f_sh(h_in)
        logits_bug = self.bug_head(z_sh).squeeze(-1)
        self.grl.lambd = float(lambd_grl)
        logits_dom = self.dom_head(self.grl(z_sh))
        return {
            "Z_sh": z_sh,
            "logits_bug": logits_bug,
            "logits_dom": logits_dom,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="Path to dump dir containing repr.npz and meta.jsonl")
    ap.add_argument("--source-projects", required=True, help="comma-separated source projects")
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dom", type=float, default=0.5)
    ap.add_argument("--grl-gamma", type=float, default=10.0, help="sigmoid schedule rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-intra", type=int, default=0)
    ap.add_argument("--feature-key", default="H_file", help="feature key from repr.npz: H_file, Z_sh, Z_pr")
    ap.add_argument("--use-bottleneck", type=int, default=0)
    ap.add_argument("--bottleneck-dim", type=int, default=128)
    ap.add_argument("--bottleneck-hidden", type=int, default=0)
    ap.add_argument("--bottleneck-dropout", type=float, default=0.0)
    ap.add_argument("--am-margin", type=float, default=0.0, help="additive margin for positive logits in BCE")
    ap.add_argument("--export-embeddings", type=int, default=1)
    ap.add_argument("--embeddings-max", type=int, default=500)
    ap.add_argument("--temp-scale", type=int, default=1)
    ap.add_argument("--temp-max-iter", type=int, default=200)
    ap.add_argument("--threshold-mode", default="mcc", choices=["mcc", "f1", "fbeta"])
    ap.add_argument("--f-beta", type=float, default=2.0)
    ap.add_argument("--use-percentile", type=int, default=1)
    ap.add_argument("--drift-thr", type=float, default=0.05)
    # legacy args (unused, kept for compatibility)
    ap.add_argument("--lambda-pr", type=float, default=0.0)
    ap.add_argument("--lambda-ortho", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build task splits
    source_projects = args.source_projects.split(",")
    splits = build_cpdp_task(
        dump_dir / "meta.jsonl",
        source_projects=source_projects,
        target_project=args.target_project,
        train_ratio=0.8,
        seed=args.seed,
        dedup_intra_project=bool(args.dedup_intra),
    )
    try:
        save_splits(splits, out_dir / "splits.json", meta_path=dump_dir / "meta.jsonl")
    except TypeError:
        # backward compatibility if save_splits signature lacks meta_path
        save_splits(splits, out_dir / "splits.json")

    # Feature cache
    cache = FeatureCache(dump_dir)

    # Prepare tensors
    H_src_tr, y_src_tr, _ = cache.get_features(splits.source_train.uids, key=args.feature_key)
    H_src_val, y_src_val, _ = cache.get_features(splits.source_val.uids, key=args.feature_key)
    H_tgt_te, y_tgt_te, _ = cache.get_features(splits.target_test.uids, key=args.feature_key)

    # Sampler & pos_weight
    sampler, pos_weight = build_sampler(y_src_tr)

    source_ds = TensorDataset(H_src_tr, y_src_tr)
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, sampler=sampler)

    val_ds = TensorDataset(H_src_val, y_src_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    target_ds = TensorDataset(H_tgt_te)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True) if len(target_ds) > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = Heads(
        d_in=H_src_tr.shape[1],
        d_sh=128,
        dom_hidden=64,
        use_bottleneck=bool(args.use_bottleneck),
        bottleneck_dim=args.bottleneck_dim,
        bottleneck_hidden=args.bottleneck_hidden,
        bottleneck_dropout=args.bottleneck_dropout,
    ).to(device)

    opt = torch.optim.Adam(heads.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    ce = nn.CrossEntropyLoss()

    def grl_lambda(p):
        # p in [0,1]
        return 2.0 / (1 + math.exp(-args.grl_gamma * p)) - 1.0

    steps_per_epoch = max(len(source_loader), 1)
    total_steps = max(args.epochs * steps_per_epoch, 1)

    # Training loop
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        batch_times = []
        loss_cls_acc = []
        loss_dom_acc = []
        heads.train()
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader) if target_loader is not None else None
        for step in range(steps_per_epoch):
            t0 = time.perf_counter()
            try:
                h_src, y_src = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                h_src, y_src = next(src_iter)
            if target_loader is not None:
                try:
                    (h_tgt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(target_loader)
                    (h_tgt,) = next(tgt_iter)
            else:
                h_tgt = None

            h_src, y_src = h_src.to(device), y_src.to(device)
            if h_tgt is not None:
                h_tgt = h_tgt.to(device)

            p = (epoch * steps_per_epoch + step + 1) / total_steps
            lambda_val = grl_lambda(p) * float(args.lambda_dom)

            out_src = heads(h_src, lambda_val)
            out_tgt = heads(h_tgt, lambda_val) if h_tgt is not None else None

            logits_for_loss = out_src["logits_bug"]
            if args.am_margin > 0.0:
                logits_for_loss = logits_for_loss - y_src * float(args.am_margin)
            loss_cls = bce(logits_for_loss, y_src)

            if out_tgt is not None and out_tgt["logits_dom"].numel() > 0:
                dom_logits = torch.cat([out_src["logits_dom"], out_tgt["logits_dom"]], dim=0)
                dom_labels = torch.cat(
                    [
                        torch.zeros(out_src["logits_dom"].size(0), dtype=torch.long, device=device),
                        torch.ones(out_tgt["logits_dom"].size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                loss_dom = ce(dom_logits, dom_labels)
            else:
                loss_dom = torch.zeros((), device=device)

            loss = loss_cls + lambda_val * loss_dom

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_times.append(time.perf_counter() - t0)
            loss_cls_acc.append(loss_cls.item())
            loss_dom_acc.append(loss_dom.item())

        # val threshold selection
        heads.eval()
        with torch.no_grad():
            val_scores = []
            val_labels = []
            for h, y in val_loader:
                h, y = h.to(device), y.to(device)
                out = heads(h, lambd_grl=0.0)
                score = torch.sigmoid(out["logits_bug"])
                val_scores.append(score.cpu())
                val_labels.append(y.cpu())
            val_scores = torch.cat(val_scores) if val_scores else torch.zeros(0)
            val_labels = torch.cat(val_labels) if val_labels else torch.zeros(0)
            if val_scores.numel() > 0:
                best_t, best_mcc = best_threshold(val_scores, val_labels, mode="mcc")
            else:
                best_t, best_mcc = 0.5, 0.0
        roc_val, pr_val = roc_pr(val_scores, val_labels) if val_scores.numel() > 0 else (None, None)
        val_bin = bin_metrics(val_scores, val_labels, best_t) if val_scores.numel() > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0}
        epoch_time = time.perf_counter() - epoch_start
        lr_cur = opt.param_groups[0]["lr"]
        p_epoch = (epoch + 1) / max(args.epochs, 1)
        lambda_epoch = grl_lambda(p_epoch) * float(args.lambda_dom)
        domain_acc = None
        if H_src_val.numel() > 0 and H_tgt_te.numel() > 0:
            with torch.no_grad():
                logits_src = heads(H_src_val.to(device), lambd_grl=0.0)["logits_dom"]
                logits_tgt = heads(H_tgt_te.to(device), lambd_grl=0.0)["logits_dom"]
                preds = torch.argmax(torch.cat([logits_src, logits_tgt], 0), dim=1)
                labels = torch.cat(
                    [
                        torch.zeros(len(logits_src), dtype=torch.long),
                        torch.ones(len(logits_tgt), dtype=torch.long),
                    ],
                    0,
                )
                domain_acc = float((preds.cpu() == labels).float().mean().item())
        auc_str = f"{roc_val:.4f}" if roc_val is not None else "na"
        prauc_str = f"{pr_val:.4f}" if pr_val is not None else "na"
        print(
            f"Epoch {epoch+1}: best threshold={best_t:.4f} MCC={best_mcc:.4f} "
            f"AUC={auc_str} PR-AUC={prauc_str} "
            f"Prec={val_bin['precision']:.4f} Rec={val_bin['recall']:.4f} F1={val_bin['f1']:.4f} "
            f"Acc={val_bin['acc']:.4f} "
            f"Losses cls={np.mean(loss_cls_acc):.4f} dom={np.mean(loss_dom_acc):.4f} "
            f"lr={lr_cur:.6f} epoch_time={epoch_time:.2f}s batch_time_avg={np.mean(batch_times):.4f}s"
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "loss_cls": float(np.mean(loss_cls_acc)),
                        "loss_dom": float(np.mean(loss_dom_acc)),
                        "best_threshold": float(best_t),
                        "best_mcc_val": float(best_mcc),
                        "roc_auc": roc_val,
                        "pr_auc": pr_val,
                        "epoch_time_sec": float(epoch_time),
                        "domain_acc": domain_acc,
                        "lr": float(lr_cur),
                        "lambda_dom": float(lambda_epoch),
                    }
                )
                + "\n"
            )

    # Final threshold on source val
    heads.eval()
    with torch.no_grad():
        val_scores = []
        val_labels = []
        val_logits = []
        for h, y in val_loader:
            h, y = h.to(device), y.to(device)
            out = heads(h, lambd_grl=0.0)
            logits = out["logits_bug"]
            score = torch.sigmoid(logits)
            val_scores.append(score.cpu())
            val_logits.append(logits.cpu())
            val_labels.append(y.cpu())
        val_scores = torch.cat(val_scores) if val_scores else torch.zeros(0)
        val_logits = torch.cat(val_logits) if val_logits else torch.zeros(0)
        val_labels = torch.cat(val_labels) if val_labels else torch.zeros(0)
        if val_scores.numel() > 0:
            best_t, best_mcc = best_threshold(val_scores, val_labels, mode="mcc")
        else:
            best_t, best_mcc = 0.5, 0.0

    # Temperature scaling on source-val logits
    temp_T = 1.0
    if args.temp_scale and val_logits.numel() > 0:
        temp_T = fit_temperature(val_logits, val_labels, max_iter=args.temp_max_iter)
    val_scores_cal = torch.sigmoid(val_logits / temp_T) if val_logits.numel() > 0 else val_scores
    if val_scores_cal.numel() > 0:
        best_t, best_val = best_threshold(
            val_scores_cal,
            val_labels,
            mode=args.threshold_mode,
            beta=args.f_beta,
        )
    else:
        best_t, best_val = 0.5, 0.0
    srcval_bin_at_tau = bin_metrics(val_scores_cal, val_labels, best_t) if val_scores_cal.numel() > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0, "mcc": 0.0, "pf": 0.0, "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    val_score_stats = score_stats(val_scores_cal)

    # Test on target
    # Save logits for 0.5 threshold metrics
    preds = []
    scores_all = []
    logits_all = []
    labels_all = []
    uids_all = splits.target_test.uids
    tgt_loader_eval = DataLoader(TensorDataset(H_tgt_te, y_tgt_te), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for (h, y) in tgt_loader_eval:
            h = h.to(device)
            out = heads(h, lambd_grl=0.0)
            logits = out["logits_bug"]
            score = torch.sigmoid(logits / temp_T)
            scores_all.append(score.cpu())
            logits_all.append(logits.cpu())
            labels_all.append(y)
    scores_all = torch.cat(scores_all) if scores_all else torch.zeros(0)
    logits_all = torch.cat(logits_all) if logits_all else torch.zeros(0)
    labels_all = torch.cat(labels_all) if labels_all else torch.zeros(0)
    roc, pr = roc_pr(scores_all, labels_all) if scores_all.numel() > 0 else (None, None)
    tgt_score_stats = score_stats(scores_all)

    # Metrics at default threshold 0.5
    default_bin = bin_metrics(scores_all, labels_all, 0.5) if scores_all.numel() > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0, "mcc": 0.0, "pf": 0.0, "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    # Metrics at transfer (source-val) threshold
    transfer_bin = bin_metrics(scores_all, labels_all, best_t) if scores_all.numel() > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0, "mcc": 0.0, "pf": 0.0, "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}

    # Percentile transfer threshold (source-val quantile -> target quantile)
    use_pct, drift = should_use_percentile(val_score_stats, tgt_score_stats, thr=args.drift_thr)
    if not args.use_percentile:
        use_pct = False
    tau_tgt, q_src = percentile_transfer_threshold(val_scores_cal, scores_all, best_t) if use_pct else (best_t, 0.0)
    transfer_pct = (
        bin_metrics(scores_all, labels_all, tau_tgt)
        if scores_all.numel() > 0
        else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0, "mcc": 0.0, "pf": 0.0, "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    )
    threshold_policy = f"{args.threshold_mode}"
    if args.threshold_mode == "fbeta":
        threshold_policy = f"f{args.f_beta:g}"
    if use_pct:
        threshold_policy = f"{threshold_policy}_percentile"

    y_pred = (scores_all >= tau_tgt).int()

    # Oracle thresholds on target
    best_f1 = 0.0
    best_f1_t = 0.5
    best_mcc_oracle = -1.0
    best_mcc_t = 0.5
    if scores_all.numel() > 0:
        scores_np = scores_all.cpu().numpy()
        labels_np = labels_all.cpu().numpy()
        thresholds = np.unique(scores_np)
        for t in thresholds:
            pred = (scores_np >= t).astype(int)
            tp = float(((pred == 1) & (labels_np == 1)).sum())
            tn = float(((pred == 0) & (labels_np == 0)).sum())
            fp = float(((pred == 1) & (labels_np == 0)).sum())
            fn = float(((pred == 0) & (labels_np == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc = ((tp * tn - fp * fn) / np.sqrt(denom + 1e-12)) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_f1_t = float(t)
            if mcc > best_mcc_oracle:
                best_mcc_oracle = mcc
                best_mcc_t = float(t)

    # Final domain discriminator accuracy (source train+val vs target test)
    final_domain_acc = 0.0
    with torch.no_grad():
        H_src_all = torch.cat([H_src_tr, H_src_val], dim=0) if len(H_src_tr) > 0 or len(H_src_val) > 0 else torch.zeros((0, H_src_tr.shape[1]))
        if H_src_all.numel() > 0 and H_tgt_te.numel() > 0:
            logits_src = heads(H_src_all.to(device), lambd_grl=0.0)["logits_dom"]
            logits_tgt = heads(H_tgt_te.to(device), lambd_grl=0.0)["logits_dom"]
            preds = torch.argmax(torch.cat([logits_src, logits_tgt], 0), dim=1)
            labels = torch.cat(
                [
                    torch.zeros(len(logits_src), dtype=torch.long),
                    torch.ones(len(logits_tgt), dtype=torch.long),
                ],
                0,
            )
            final_domain_acc = float((preds.cpu() == labels).float().mean().item())

    # save predictions
    import pandas as pd

    df = pd.DataFrame(
        {
            "uid": uids_all,
            "target_project": [args.target_project] * len(uids_all),
            "y_true": labels_all.numpy() if labels_all.numel() > 0 else np.array([]),
            "y_score": scores_all.numpy() if scores_all.numel() > 0 else np.array([]),
            "y_pred": y_pred.numpy() if y_pred.numel() > 0 else np.array([]),
            "threshold_used": [float(tau_tgt)] * len(uids_all),
            "threshold_policy": [threshold_policy] * len(uids_all),
            "threshold_mode": [args.threshold_mode] * len(uids_all),
            "f_beta": [float(args.f_beta)] * len(uids_all),
        }
    )
    df.to_csv(out_dir / "predictions.csv", index=False)

    # Export embeddings for visualization (z_adapt from bottleneck output)
    if args.export_embeddings:
        rng = np.random.default_rng(args.seed)
        src_uids = splits.source_train.uids + splits.source_val.uids
        tgt_uids = splits.target_test.uids
        src_idx = np.arange(len(src_uids))
        tgt_idx = np.arange(len(tgt_uids))
        n_each = args.embeddings_max // 2 if args.embeddings_max > 1 else 0
        if n_each > 0:
            if len(src_idx) > n_each:
                src_idx = rng.choice(src_idx, size=n_each, replace=False)
            if len(tgt_idx) > n_each:
                tgt_idx = rng.choice(tgt_idx, size=n_each, replace=False)
        sel_src_uids = [src_uids[i] for i in src_idx]
        sel_tgt_uids = [tgt_uids[i] for i in tgt_idx]
        H_src_sel, _, _ = cache.get_features(sel_src_uids, key=args.feature_key)
        H_tgt_sel, _, _ = cache.get_features(sel_tgt_uids, key=args.feature_key)
        heads.eval()
        with torch.no_grad():
            z_src = heads.bottleneck(H_src_sel.to(device)).cpu().numpy() if H_src_sel.numel() > 0 else np.zeros((0, 0))
            z_tgt = heads.bottleneck(H_tgt_sel.to(device)).cpu().numpy() if H_tgt_sel.numel() > 0 else np.zeros((0, 0))
        z_all = np.vstack([z_src, z_tgt]) if z_src.size and z_tgt.size else (z_src if z_src.size else z_tgt)
        meta = {
            "uids": sel_src_uids + sel_tgt_uids,
            "domain": [0] * len(sel_src_uids) + [1] * len(sel_tgt_uids),
            "feature_key": args.feature_key,
            "note": "z_adapt from bottleneck output",
        }
        np.save(out_dir / "embeddings.npy", z_all)
        (out_dir / "embeddings_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics = {
        "target_project": args.target_project,
        "roc_auc": roc,
        "pr_auc": pr,
        "temp_scale": {
            "enabled": bool(args.temp_scale),
            "T": float(temp_T),
        },
        "score_stats": {
            "source_val": val_score_stats,
            "target": tgt_score_stats,
        },
        "default_0p5": {
            "f1": float(default_bin["f1"]),
            "mcc": float(default_bin["mcc"]),
            "recall": float(default_bin["recall"]),
            "precision": float(default_bin["precision"]),
            "pf": float(default_bin["pf"]),
        },
        "transfer_source": {
            "threshold": float(best_t),
            "f1": float(transfer_bin["f1"]),
            "mcc": float(transfer_bin["mcc"]),
            "recall": float(transfer_bin["recall"]),
            "precision": float(transfer_bin["precision"]),
            "pf": float(transfer_bin["pf"]),
        },
        "transfer_percentile": {
            "q_source": float(q_src),
            "threshold_target": float(tau_tgt),
            "f1": float(transfer_pct["f1"]),
            "mcc": float(transfer_pct["mcc"]),
            "recall": float(transfer_pct["recall"]),
            "precision": float(transfer_pct["precision"]),
            "pf": float(transfer_pct["pf"]),
        },
        "threshold_policy": threshold_policy,
        "beta": float(args.f_beta),
        "drift": drift,
        "use_percentile": bool(use_pct),
        "tau_select_srcval": {
            "threshold": float(best_t),
            "mode": args.threshold_mode,
            "best_val": float(best_val),
            "f1": float(srcval_bin_at_tau["f1"]),
            "mcc": float(srcval_bin_at_tau["mcc"]),
            "recall": float(srcval_bin_at_tau["recall"]),
            "precision": float(srcval_bin_at_tau["precision"]),
            "pf": float(srcval_bin_at_tau["pf"]),
        },
        "final_on_target": {
            "threshold": float(tau_tgt),
            "f1": float(transfer_pct["f1"]),
            "mcc": float(transfer_pct["mcc"]),
            "recall": float(transfer_pct["recall"]),
            "precision": float(transfer_pct["precision"]),
            "pf": float(transfer_pct["pf"]),
        },
        "oracle_target": {
            "best_threshold": float(best_f1_t),
            "best_f1": float(best_f1),
            "best_mcc": float(best_mcc_oracle),
            "best_threshold_mcc": float(best_mcc_t),
        },
        "final_domain_acc": float(final_domain_acc),
        "threshold_from": f"source_val_{args.threshold_mode}",
        "best_mcc_val": float(srcval_bin_at_tau["mcc"]),
        "source_projects": source_projects,
        "lambda_dom": float(args.lambda_dom),
        "feature_key": args.feature_key,
        "use_bottleneck": bool(args.use_bottleneck),
        "bottleneck_dim": int(args.bottleneck_dim),
        "bottleneck_hidden": int(args.bottleneck_hidden),
        "bottleneck_dropout": float(args.bottleneck_dropout),
        "am_margin": float(args.am_margin),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    ckpt = {
        "state_dict": heads.state_dict(),
        "config": vars(args),
        "threshold": float(best_t),
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    print(f"Saved to {out_dir}")
    print(f"PR-AUC={pr:.4f}" if pr is not None else "PR-AUC=na")
    print(f"Transfer F1={metrics['transfer_source']['f1']:.4f}")


if __name__ == "__main__":
    main()
