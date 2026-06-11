"""
Domain probe on dumped representations.

Reads repr.npz from dump_representations.py, trains tiny probes on Z_sh / Z_pr (and optionally H_file)
to predict project_id, reports cross-entropy (lower is more domain information).

Example:
  python src/ch3/probe_domain.py --repr outputs/ch3_dump/best_w3_t256/repr.npz --epochs 10 --lr 1e-3 --batch-size 256
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
    x = x[perm]
    y = y[perm]
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


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
    optim = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb)
            loss = ce(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

    probe.eval()
    with torch.no_grad():
        logits_tr = probe(x_tr.to(device))
        logits_te = probe(x_te.to(device))
        loss_tr = ce(logits_tr, y_tr.to(device)).item()
        loss_te = ce(logits_te, y_te.to(device)).item()
    return loss_tr, loss_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", required=True, help="Path to repr.npz from dump_representations.py")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=0, help="Hidden dim for probe; 0 => linear")
    ap.add_argument("--out", default=None, help="Output json; default: repr dir / probe_metrics.json")
    args = ap.parse_args()

    data = np.load(args.repr)
    proj_id = torch.from_numpy(data["proj_id"]).long()
    num_classes = int(proj_id.max().item() + 1)

    out = {
        "num_classes": num_classes,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
    }

    for key in ["Z_sh", "Z_pr", "H_file"]:
        if key not in data:
            continue
        x = torch.from_numpy(data[key]).float()
        tr_loss, te_loss = train_probe(x, proj_id, num_classes, args.epochs, args.lr, args.batch_size, args.hidden)
        out[f"{key.lower()}_ce_train"] = float(tr_loss)
        out[f"{key.lower()}_ce_test"] = float(te_loss)

    out_path = Path(args.out) if args.out else Path(args.repr).with_name("probe_metrics.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved probe metrics to {out_path}")


if __name__ == "__main__":
    main()
