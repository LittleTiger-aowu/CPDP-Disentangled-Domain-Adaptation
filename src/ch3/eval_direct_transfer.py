"""
Direct transfer evaluation for Chapter 3.

Evaluate a trained Ch3 checkpoint on a single target project (no adaptation).
Outputs: predictions.csv + metrics.json (ROC-AUC / PR-AUC if sklearn available).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rep.collate import collate_batch
from src.rep.dataset import FileDataset
from src.rep.encoder import CodeBertBlockEncoder, build_tokenizer
from src.rep.model import RepresentationModel
from src.rep.struct_prior import StatsMLP, TypeEmbedding, num_block_types


def roc_pr(scores: torch.Tensor, labels: torch.Tensor):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except Exception:
        return None, None
    y_true = labels.cpu().numpy()
    y_score = scores.cpu().numpy()
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def bin_metrics(scores: torch.Tensor, labels: torch.Tensor, thresh: float):
    y_true = labels.cpu().numpy().astype(np.int64)
    y_pred = (scores.cpu().numpy() >= thresh).astype(np.int64)
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
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "pf": pf,
        "mcc": mcc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--project-vocab", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target-project", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--win-size-lines", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--dedup-by-sha1", default="within_project")
    ap.add_argument("--skip-template", type=int, default=0)
    ap.add_argument("--skip-missing-src", type=int, default=0)
    ap.add_argument("--codebert-path", required=True)
    ap.add_argument("--local-files-only", type=int, default=1)
    ap.add_argument("--max-blocks-per-file", type=int, default=None)
    ap.add_argument("--block-cache-dir", default=None)
    ap.add_argument("--use-bug-file", type=int, default=0, help="1: use logit_bug_file; 0: use logit_bug_sh")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt.get("config", {})

    if args.tmax is None:
        args.tmax = int(config.get("tmax", 128))
    if args.w is None:
        args.w = int(config.get("w", 2))
    if args.win_size_lines is None:
        args.win_size_lines = int(config.get("win_size_lines", 20))
    if args.max_blocks_per_file is None:
        args.max_blocks_per_file = int(config.get("max_blocks_per_file", 128))

    project_vocab = json.load(open(args.project_vocab, "r", encoding="utf-8"))
    if args.target_project not in project_vocab:
        raise ValueError(f"target project {args.target_project} not in project_vocab")

    dataset = FileDataset(
        args.data_parquet,
        project_vocab,
        dedup_by_sha1=args.dedup_by_sha1,
        max_files=args.max_files,
    )
    dataset.df = dataset.df[dataset.df["project"] == args.target_project].reset_index(drop=True)
    if len(dataset) == 0:
        raise ValueError(f"No samples found for target project {args.target_project}")

    tokenizer = build_tokenizer(args.codebert_path, local_files_only=bool(args.local_files_only))
    block_cache = None
    if args.block_cache_dir:
        cache_dir = Path(args.block_cache_dir)
        index_path = cache_dir / "index.json"
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        block_cache = {uid: str(cache_dir / path) for uid, path in index.items()}

    def _collate(batch):
        return collate_batch(
            batch,
            tokenizer=tokenizer,
            tmax=args.tmax,
            win_size_lines=args.win_size_lines,
            window=args.w,
            max_blocks_per_file=args.max_blocks_per_file,
            skip_template=bool(args.skip_template),
            skip_missing_src=bool(args.skip_missing_src),
            block_cache=block_cache,
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CodeBertBlockEncoder(args.codebert_path, local_files_only=bool(args.local_files_only))
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
    uids_all = []
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
            if args.use_bug_file:
                logits = outputs["logit_bug_file"].squeeze(-1)
            else:
                logits = outputs["logit_bug_sh"].squeeze(-1)
            scores = torch.sigmoid(logits)
            scores_all.append(scores.detach().cpu())
            labels_all.append(y.detach().cpu())
            uids_all.extend(meta["uid"])

    scores_all = torch.cat(scores_all) if scores_all else torch.zeros(0)
    labels_all = torch.cat(labels_all) if labels_all else torch.zeros(0)

    roc_auc, pr_auc = roc_pr(scores_all, labels_all)
    bin_05 = bin_metrics(scores_all, labels_all, 0.5) if scores_all.numel() > 0 else {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save predictions
    import pandas as pd

    df = pd.DataFrame(
        {
            "uid": uids_all,
            "y_true": labels_all.numpy() if labels_all.numel() > 0 else np.array([]),
            "y_score": scores_all.numpy() if scores_all.numel() > 0 else np.array([]),
            "y_pred_0p5": (scores_all >= 0.5).int().numpy() if scores_all.numel() > 0 else np.array([]),
        }
    )
    df.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "target_project": args.target_project,
        "n_samples": int(labels_all.numel()),
        "pos_ratio": float(labels_all.mean().item()) if labels_all.numel() > 0 else 0.0,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "metrics_at_0p5": bin_05,
        "use_bug_file": bool(args.use_bug_file),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved direct transfer metrics to {out_dir}")


if __name__ == "__main__":
    main()
