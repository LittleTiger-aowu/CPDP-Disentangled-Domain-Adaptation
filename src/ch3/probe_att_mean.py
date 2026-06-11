"""
Compare attention-pooled H_file (already in repr.npz) vs mean-pooled H_mean
for domain prediction (project_id). Outputs cross-entropy on a holdout split.

Usage:
  python src/ch3/probe_att_mean.py --repr <dump_dir>/repr.npz --epochs 10 --lr 1e-3 --batch-size 256

If H_blk is absent in repr.npz, the base will exit with an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def split_dataset(x: torch.Tensor, y: torch.Tensor, split: float = 0.8):
    n = x.size(0)
    n_train = int(n * split)
    perm = torch.randperm(n)
    return (x[perm[:n_train]], y[perm[:n_train]]), (x[perm[n_train:]], y[perm[n_train:]])


class Probe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def train_probe(x, y, num_classes, epochs, lr, batch_size, hidden):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_tr, y_tr), (x_te, y_te) = split_dataset(x, y)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss()

    probe = Probe(x.shape[1], num_classes, hidden).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        tr = ce(probe(x_tr.to(device)), y_tr.to(device)).item()
        te = ce(probe(x_te.to(device)), y_te.to(device)).item()
    return tr, te


def compute_mean_pool(h_blk: np.ndarray, blk_ptr: np.ndarray) -> np.ndarray:
    h = []
    for i in range(len(blk_ptr) - 1):
        s = int(blk_ptr[i])
        e = int(blk_ptr[i + 1])
        seg = h_blk[s:e]
        if seg.shape[0] == 0:
            h.append(np.zeros((h_blk.shape[1],), dtype=np.float32))
        else:
            h.append(seg.mean(axis=0))
    return np.stack(h, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", required=True, help="Path to repr.npz from dump_representations.py")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output json; default: repr dir / probe_att_mean.json")
    args = ap.parse_args()

    data = np.load(args.repr)
    required = ["H_file", "proj_id", "H_blk", "blk_ptr"]
    for k in required:
        if k not in data:
            raise FileNotFoundError(f"{k} missing in {args.repr}. Re-run dump with H_blk saved.")

    proj_id = torch.from_numpy(data["proj_id"]).long()
    num_classes = int(proj_id.max().item() + 1)

    h_file = torch.from_numpy(data["H_file"]).float()
    h_mean = torch.from_numpy(compute_mean_pool(data["H_blk"], data["blk_ptr"])).float()

    out = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "num_classes": num_classes,
    }

    for name, x in [("att", h_file), ("mean", h_mean)]:
        tr, te = train_probe(x, proj_id, num_classes, args.epochs, args.lr, args.batch_size, args.hidden)
        out[f"{name}_ce_train"] = float(tr)
        out[f"{name}_ce_test"] = float(te)

    out_path = Path(args.out) if args.out else Path(args.repr).with_name("probe_att_mean.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved probe att vs mean metrics to {out_path}")


if __name__ == "__main__":
    main()
