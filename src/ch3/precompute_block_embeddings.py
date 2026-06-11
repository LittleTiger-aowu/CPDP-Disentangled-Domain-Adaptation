"""
Precompute CodeBERT block embeddings for TPSM blocks.

Example:
  python src/ch3/precompute_block_embeddings.py --data-parquet data/processed/ubd_class.parquet \
    --project-vocab data/processed/ubd_project_vocab.json --out-dir outputs/block_cache \
    --codebert-path E:\\project\\WYP\\CPDP\\CodeBert --tmax 128 --win-size-lines 20 --max-blocks-per-file 256
"""
from __future__ import annotations

import argparse
import hashlib
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


def uid_to_fname(uid: str) -> str:
    return hashlib.sha1(uid.encode("utf-8")).hexdigest() + ".npy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parquet", required=True)
    parser.add_argument("--project-vocab", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=128)
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--win-size-lines", type=int, default=20)
    parser.add_argument("--max-blocks-per-file", type=int, default=256)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--codebert-path", default="E:\\project\\WYP\\CPDP\\CodeBert")
    parser.add_argument("--local-files-only", type=int, default=1)
    parser.add_argument("--encoder-device", default="cuda")
    parser.add_argument("--dedup-by-sha1", default="within_project")
    args = parser.parse_args()

    with open(args.project_vocab, "r", encoding="utf-8") as f:
        project_vocab = json.load(f)

    dataset = FileDataset(
        args.data_parquet,
        project_vocab,
        dedup_by_sha1=args.dedup_by_sha1,
        max_files=args.max_files,
    )

    tokenizer = build_tokenizer(args.codebert_path, local_files_only=bool(args.local_files_only))

    def _collate(batch):
        return collate_batch(
            batch,
            tokenizer=tokenizer,
            tmax=args.tmax,
            win_size_lines=args.win_size_lines,
            window=args.w,
            max_blocks_per_file=args.max_blocks_per_file,
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
    device = torch.device(args.encoder_device)

    encoder = CodeBertBlockEncoder(args.codebert_path, local_files_only=bool(args.local_files_only))
    encoder.eval()
    encoder.to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["flat_input_ids"].to(device)
            attention_mask = batch["flat_attention_mask"].to(device)
            blk_ptr = batch["blk_ptr"].cpu().numpy().tolist()
            meta = batch["meta"]

            h_sem = encoder(input_ids, attention_mask).detach().cpu().numpy()

            for i, uid in enumerate(meta["uid"]):
                start = int(blk_ptr[i])
                end = int(blk_ptr[i + 1])
                emb = h_sem[start:end]
                fname = uid_to_fname(uid)
                np.save(out_dir / fname, emb)
                index[uid] = fname

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved {len(index)} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
