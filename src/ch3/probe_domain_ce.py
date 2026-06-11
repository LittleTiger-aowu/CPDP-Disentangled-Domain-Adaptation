"""
Domain probe with cross-entropy only (no accuracy), for 3.4 representation validation.

JSON-based mode (no ckpt dependency):
  - Read project_vocab.json
  - Build source_project_vocab ON THE FLY = all projects except target_project
    (sorted by global id, then re-indexed to 0..K-1).
  - If --source-only=1: filter to SOURCE samples only, and probe over SOURCE classes only.

Inputs:
  - repr.npz from dump_representations.py (expects Z_sh, Z_pr, H_file, proj_id)
  - project_vocab.json (default: in repr.npz directory)
Outputs:
  - JSON with CE_train / CE_test for each repr.

Example:
  python src/ch3/probe_domain_ce.py ^
    --repr outputs/ch3_direct_transfer/target_Mylyn-3.1/dump/repr.npz ^
    --target-project Mylyn-3.1 ^
    --source-only 1 ^
    --out outputs/ch3_direct_transfer/target_Mylyn-3.1/domain_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def split_dataset(x: torch.Tensor, y: torch.Tensor, split: float = 0.8, seed: int = 0):
    n = x.size(0)
    n_train = int(n * split)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    x = x[perm]
    y = y[perm]
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


class Probe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def train_probe(x, y, num_classes, epochs, lr, batch_size, hidden, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_tr, y_tr), (x_te, y_te) = split_dataset(x, y, split=0.8, seed=seed)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss()
    probe = Probe(x.shape[1], num_classes, hidden).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(probe(xb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

    probe.eval()
    with torch.no_grad():
        logits_tr = probe(x_tr.to(device))
        logits_te = probe(x_te.to(device))
        loss_tr = ce(logits_tr, y_tr.to(device)).item()
        loss_te = ce(logits_te, y_te.to(device)).item()
        acc_tr = float((logits_tr.argmax(dim=1) == y_tr.to(device)).float().mean().item())
        acc_te = float((logits_te.argmax(dim=1) == y_te.to(device)).float().mean().item())
    return loss_tr, loss_te, acc_tr, acc_te


def _load_json(path: Path, name: str) -> dict:
    if not path.exists():
        raise ValueError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_source_project_vocab(project_vocab: dict, target_project: str) -> dict:
    """
    project_vocab: name -> global_id
    returns: source_project_vocab: name -> source_local_id (0..K-1)
    Rule: source = all projects except target_project, sorted by global_id asc.
    """
    if target_project not in project_vocab:
        raise ValueError(f"target_project '{target_project}' not found in project_vocab.json")

    items = [(name, int(pid)) for name, pid in project_vocab.items() if name != target_project]
    items.sort(key=lambda kv: kv[1])  # sort by global id
    return {name: i for i, (name, _pid) in enumerate(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", required=True, help="path to repr.npz")
    ap.add_argument("--target-project", required=True, help="target project name (excluded from source vocab)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)

    ap.add_argument(
        "--source-only",
        type=int,
        default=1,
        help="1: only probe on source projects (recommended for CPDP); 0: all projects",
    )
    ap.add_argument(
        "--project-vocab",
        default=None,
        help="path to project_vocab.json (default: repr dir / project_vocab.json)",
    )
    ap.add_argument(
        "--dump-source-vocab",
        type=int,
        default=1,
        help="1: also write constructed source_project_vocab.json next to output json; 0: do not write",
    )
    args = ap.parse_args()

    data = np.load(args.repr)
    if "proj_id" not in data:
        raise ValueError("repr.npz missing required field: proj_id")
    proj_id_global = torch.from_numpy(data["proj_id"]).long()

    repr_dir = Path(args.repr).parent
    project_vocab_path = Path(args.project_vocab) if args.project_vocab else (repr_dir / "project_vocab.json")

    # name -> global_id
    project_vocab = _load_json(project_vocab_path, "project_vocab.json")
    # build source vocab from (project_vocab - target)
    source_project_vocab = build_source_project_vocab(project_vocab, args.target_project)

    inv_project_vocab = {int(v): k for k, v in project_vocab.items()}

    # Map global proj_id -> project name
    proj_names = [inv_project_vocab.get(int(pid), None) for pid in proj_id_global.tolist()]
    if any(n is None for n in proj_names):
        bad = sum(n is None for n in proj_names)
        raise ValueError(f"repr contains {bad} proj_id not found in project_vocab.json")

    if bool(args.source_only):
        # name -> source local id, filter out target (=-1)
        proj_id_src = torch.tensor([source_project_vocab.get(name, -1) for name in proj_names], dtype=torch.long)
        mask = proj_id_src >= 0
        if not bool(mask.any()):
            raise ValueError("No source samples found after source-only filter (check target-project spelling)")
        y = proj_id_src[mask]
        num_classes = int(len(source_project_vocab))  # fixed K (e.g., 4)
        # sanity
        if int(y.min().item()) < 0 or int(y.max().item()) >= num_classes:
            raise ValueError("Constructed source_project_vocab produced out-of-range labels.")
    else:
        # probe on all projects using global ids (may include target)
        mask = torch.ones_like(proj_id_global, dtype=torch.bool)
        y = proj_id_global
        num_classes = int(y.max().item() + 1)

    out = {
        "target_project": args.target_project,
        "num_classes": int(num_classes),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "seed": args.seed,
        "source_only": bool(args.source_only),
        "num_samples_total": int(proj_id_global.numel()),
        "num_samples_used": int(mask.sum().item()),
        "repr": str(args.repr),
        "project_vocab": str(project_vocab_path),
        "source_vocab_built_from": "project_vocab_minus_target_sorted_by_global_id",
        "source_projects": [k for k, _ in sorted(source_project_vocab.items(), key=lambda kv: kv[1])],
    }

    for key in ["Z_sh", "Z_pr", "H_file"]:
        if key not in data:
            continue
        x = torch.from_numpy(data[key]).float()
        x = x[mask]
        tr, te, acc_tr, acc_te = train_probe(
            x, y, num_classes, args.epochs, args.lr, args.batch_size, args.hidden, args.seed
        )
        out[f"{key.lower()}_ce_train"] = float(tr)
        out[f"{key.lower()}_ce_test"] = float(te)
        out[f"{key.lower()}_acc_train"] = float(acc_tr)
        out[f"{key.lower()}_acc_test"] = float(acc_te)

    out_path = Path(args.out) if args.out else Path(args.repr).with_name("domain_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved probe metrics to {out_path}")

    # optionally dump constructed source_project_vocab.json next to output
    if bool(args.dump_source_vocab) and bool(args.source_only):
        src_path = out_path.with_name("source_project_vocab.json")
        src_path.write_text(json.dumps(source_project_vocab, indent=2), encoding="utf-8")
        print(f"Saved constructed source_project_vocab to {src_path}")


if __name__ == "__main__":
    main()
