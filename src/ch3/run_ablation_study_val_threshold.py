"""
Run Ch3 ablation study with source-val threshold selection (maximize MCC),
then evaluate on target with fixed threshold.

Steps per variant:
  1) Split source (all except target) into train/val by sha1 (grouped).
  2) Train on source-train only.
  3) Score source-val, scan threshold to maximize MCC.
  4) Score target, report ROC-AUC/PR-AUC and MCC/F1 at best threshold.

Outputs: outputs/ch3_ablation_val_threshold/ablation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rep.collate import collate_batch
from src.rep.dataset import FileDataset
from src.rep.encoder import CodeBertBlockEncoder, build_tokenizer
from src.rep.model import RepresentationModel
from src.rep.struct_prior import StatsMLP, TypeEmbedding, num_block_types


VARIANTS = [
    {
        "name": "full_method",
        "desc": "Project supervision + orthogonality",
        "flags": ["--no-ortho", "0", "--no-pr-dom", "0"],
    },
    {
        "name": "w_o_orthogonality",
        "desc": "Keep project supervision, remove orthogonality",
        "flags": ["--no-ortho", "1", "--no-pr-dom", "0"],
    },
    {
        "name": "w_o_project_supervision",
        "desc": "Remove project supervision (no L_pr); ortho disabled",
        "flags": ["--no-ortho", "1", "--no-pr-dom", "1"],
    },
    {
        "name": "w_o_gcn",
        "desc": "Remove GCN (no deep structure)",
        "flags": ["--no-gcn", "1"],
    },
]


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def split_source_by_sha1(df: pd.DataFrame, target_project: str, val_ratio: float, seed: int):
    source_df = df[df["project"] != target_project].copy()
    sha1s = source_df["sha1"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(sha1s)
    n_val = int(len(sha1s) * val_ratio)
    val_sha1 = set(sha1s[:n_val])
    train_sha1 = set(sha1s[n_val:])
    train_df = source_df[source_df["sha1"].isin(train_sha1)].copy()
    val_df = source_df[source_df["sha1"].isin(val_sha1)].copy()
    return train_df, val_df


def build_block_cache(cache_dir: str | None):
    if not cache_dir:
        return None
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    return {uid: str(cache_dir / path) for uid, path in index.items()}


def score_parquet(
    parquet_path: Path,
    project_vocab: dict,
    ckpt_path: Path,
    codebert_path: str,
    tmax: int,
    w: int,
    win_size_lines: int,
    max_blocks_per_file: int,
    batch_size: int,
    block_cache_dir: str | None,
    use_bug_file: bool,
):
    dataset = FileDataset(str(parquet_path), project_vocab, dedup_by_sha1="within_project")
    tokenizer = build_tokenizer(codebert_path, local_files_only=True)
    block_cache = build_block_cache(block_cache_dir)

    def _collate(batch):
        return collate_batch(
            batch,
            tokenizer=tokenizer,
            tmax=tmax,
            win_size_lines=win_size_lines,
            window=w,
            max_blocks_per_file=max_blocks_per_file,
            block_cache=block_cache,
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CodeBertBlockEncoder(codebert_path, local_files_only=True)
    type_embed = TypeEmbedding(num_block_types(), config.get("d_t", 32))
    stats_mlp = StatsMLP(3, config.get("d_p", 32))
    num_projects = ckpt["model"]["d_pr.weight"].shape[0]
    model = RepresentationModel(
        d_h=config.get("d_h", 256),
        d_sh=config.get("d_sh", 128),
        d_pr=config.get("d_pr", 128),
        num_projects=num_projects,
        input_dim=768 + config.get("d_t", 32) + config.get("d_p", 32),
    )

    encoder.load_state_dict(ckpt["encoder"])
    type_embed.load_state_dict(ckpt["type_embed"])
    stats_mlp.load_state_dict(ckpt["stats_mlp"])
    model.load_state_dict(ckpt["model"])

    encoder.to(device)
    type_embed.to(device)
    stats_mlp.to(device)
    model.to(device)

    encoder.eval()
    type_embed.eval()
    stats_mlp.eval()
    model.eval()

    scores_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            meta = batch["meta"]
            blk_ptr = batch["blk_ptr"].to(device)
            struct_type_ids = batch["struct_type_ids"].to(device)
            struct_stats = batch["struct_stats"].to(device)
            edge_indices = batch["edge_indices"]
            y = torch.tensor(meta["y"], dtype=torch.float, device=device)

            if batch["h_sem"] is not None:
                h_sem = batch["h_sem"].to(device)
            else:
                input_ids = batch["flat_input_ids"].to(device)
                attention_mask = batch["flat_attention_mask"].to(device)
                h_sem = encoder(input_ids, attention_mask)

            type_emb = type_embed(struct_type_ids)
            stats_emb = stats_mlp(struct_stats)
            e_struct = torch.cat([type_emb, stats_emb], dim=1)
            outputs = model(h_sem, e_struct, blk_ptr, edge_indices=edge_indices)
            if use_bug_file:
                logits = outputs["logit_bug_file"].squeeze(-1)
            else:
                logits = outputs["logit_bug_sh"].squeeze(-1)
            scores = torch.sigmoid(logits)
            scores_all.append(scores.detach().cpu())
            labels_all.append(y.detach().cpu())

    scores_all = torch.cat(scores_all) if scores_all else torch.zeros(0)
    labels_all = torch.cat(labels_all) if labels_all else torch.zeros(0)
    return scores_all.numpy(), labels_all.numpy()


def scan_best_threshold(scores: np.ndarray, labels: np.ndarray, n_steps: int = 100):
    best_mcc = -1.0
    best_t = 0.5
    thresholds = np.linspace(0.01, 0.99, n_steps)
    for t in thresholds:
        preds = (scores >= t).astype(int)
        mcc = matthews_corrcoef(labels, preds) if labels.size else 0.0
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return best_t, best_mcc


def compute_metrics(scores: np.ndarray, labels: np.ndarray, thresh: float):
    preds = (scores >= thresh).astype(int)
    try:
        roc = float(roc_auc_score(labels, scores)) if labels.size else None
    except Exception:
        roc = None
    try:
        pr = float(average_precision_score(labels, scores)) if labels.size else None
    except Exception:
        pr = None
    if labels.size:
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        pf = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        precision = 0.0
        recall = 0.0
        pf = 0.0
    f1 = float(f1_score(labels, preds, zero_division=0)) if labels.size else 0.0
    mcc = float(matthews_corrcoef(labels, preds)) if labels.size else 0.0
    return roc, pr, f1, mcc, precision, recall, pf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--project-vocab", required=True)
    ap.add_argument("--codebert-path", required=True)
    ap.add_argument("--block-cache-dir", default=None)
    ap.add_argument("--target-project", default="Mylyn-3.1")
    ap.add_argument("--output-root", default="outputs/ch3_ablation_val_threshold")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tmax", type=int, default=256)
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--win-size-lines", type=int, default=20)
    ap.add_argument("--max-blocks-per-file", type=int, default=768)
    ap.add_argument("--max-total-blocks", type=int, default=0)
    ap.add_argument("--lambda-pr", type=float, default=1.0)
    ap.add_argument("--lambda-ortho", type=float, default=0.1)
    ap.add_argument("--lambda-ortho-warmup-epochs", type=int, default=0)
    ap.add_argument("--beta-bug-file", type=float, default=0.2)
    ap.add_argument("--freeze-encoder", type=int, default=1)
    ap.add_argument("--encoder-device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dump-max-files", type=int, default=None)
    ap.add_argument("--eval-use-bug-file", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_parquet)
    if "project" not in df.columns or "sha1" not in df.columns:
        raise ValueError("data parquet must include 'project' and 'sha1' columns.")

    project_vocab = json.load(open(args.project_vocab, "r", encoding="utf-8"))
    run_py = Path(__file__).parent / "train_representation.py"

    rows = []
    for v in VARIANTS:
        variant_root = out_root / v["name"]
        variant_root.mkdir(parents=True, exist_ok=True)

        train_df, val_df = split_source_by_sha1(df, args.target_project, args.val_ratio, args.seed)
        target_df = df[df["project"] == args.target_project].copy()
        if target_df.empty:
            raise ValueError(f"No target samples for {args.target_project}")

        train_parquet = variant_root / "source_train.parquet"
        val_parquet = variant_root / "source_val.parquet"
        target_parquet = variant_root / "target.parquet"
        train_df.to_parquet(train_parquet, index=False)
        val_df.to_parquet(val_parquet, index=False)
        target_df.to_parquet(target_parquet, index=False)

        train_cmd = [
            "python",
            str(run_py),
            "--data-parquet",
            str(train_parquet),
            "--project-vocab",
            args.project_vocab,
            "--target-project",
            args.target_project,
            "--codebert-path",
            args.codebert_path,
            "--local-files-only",
            "1",
            "--tmax",
            str(args.tmax),
            "--w",
            str(args.w),
            "--win-size-lines",
            str(args.win_size_lines),
            "--max-blocks-per-file",
            str(args.max_blocks_per_file),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--freeze-encoder",
            str(args.freeze_encoder),
            "--encoder-device",
            str(args.encoder_device),
            "--balanced-batch",
            "1",
            "--projects-per-batch",
            "4",
            "--lambda-pr",
            str(args.lambda_pr),
            "--lambda-ortho",
            str(args.lambda_ortho),
            "--lambda-ortho-warmup-epochs",
            str(args.lambda_ortho_warmup_epochs),
            "--beta-bug-file",
            str(args.beta_bug_file),
            "--no-pr-dom",
            "0",
            "--no-ortho",
            "0",
            "--no-gcn",
            "0",
            "--no-struct",
            "0",
            "--max-total-blocks",
            str(args.max_total_blocks),
            "--log-every",
            str(args.log_every),
            "--log-file",
            "1",
            "--metrics-file",
            "1",
            "--output-dir",
            str(variant_root / "ckpt"),
            "--seed",
            str(args.seed),
        ] + v["flags"]
        if args.block_cache_dir:
            train_cmd += ["--block-cache-dir", args.block_cache_dir]
        run(train_cmd)

        ckpt_dir = variant_root / "ckpt"
        ckpt_best = ckpt_dir / "best.pt"
        ckpt_use = ckpt_best if ckpt_best.exists() else ckpt_dir / "last.pt"

        val_scores, val_labels = score_parquet(
            val_parquet,
            project_vocab,
            ckpt_use,
            args.codebert_path,
            args.tmax,
            args.w,
            args.win_size_lines,
            args.max_blocks_per_file,
            args.batch_size,
            args.block_cache_dir,
            bool(args.eval_use_bug_file),
        )
        best_t, best_val_mcc = scan_best_threshold(val_scores, val_labels)

        tgt_scores, tgt_labels = score_parquet(
            target_parquet,
            project_vocab,
            ckpt_use,
            args.codebert_path,
            args.tmax,
            args.w,
            args.win_size_lines,
            args.max_blocks_per_file,
            args.batch_size,
            args.block_cache_dir,
            bool(args.eval_use_bug_file),
        )
        roc, pr, f1, mcc, precision, recall, pf = compute_metrics(tgt_scores, tgt_labels, best_t)

        rows.append(
            {
                "variant": v["name"],
                "desc": v["desc"],
                "target_project": args.target_project,
                "val_ratio": args.val_ratio,
                "best_threshold": best_t,
                "val_mcc": best_val_mcc,
                "roc_auc": roc,
                "pr_auc": pr,
                "mcc": mcc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "pf": pf,
                "n_target": int(len(tgt_labels)),
                "pos_ratio_target": float(tgt_labels.mean()) if tgt_labels.size else 0.0,
            }
        )

    out_csv = out_root / "ablation_results.csv"
    fields = [
        "variant",
        "desc",
        "target_project",
        "val_ratio",
        "best_threshold",
        "val_mcc",
        "roc_auc",
        "pr_auc",
        "mcc",
        "f1",
        "precision",
        "recall",
        "pf",
        "n_target",
        "pos_ratio_target",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ablation results to {out_csv}")


if __name__ == "__main__":
    main()
