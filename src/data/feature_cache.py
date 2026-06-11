"""
Feature cache for cached H_file representations.

Assumes dump_representations.py produced:
  - meta.jsonl (one object per file, includes uid, y, project, version, sha1, file_path, etc.)
  - repr.npz containing H_file (N x d), proj_id, y (optional)

This cache:
  - loads meta and repr.npz
  - validates length and uid uniqueness
  - builds uid -> row_idx mapping
  - provides get(uids) -> (H_file tensor, y tensor, proj_id tensor)
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch


class FeatureCache:
    def __init__(self, dump_dir: Path):
        dump_dir = Path(dump_dir)
        meta_path = dump_dir / "meta.jsonl"
        npz_path = dump_dir / "repr.npz"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        self.meta = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        self.uid_to_idx = {}
        for i, m in enumerate(self.meta):
            uid = m["uid"]
            if uid in self.uid_to_idx:
                raise ValueError(f"Duplicate uid in meta: {uid}")
            self.uid_to_idx[uid] = i

        data = np.load(npz_path)
        self.H = torch.from_numpy(data["H_file"]).float()
        self.Z_sh = torch.from_numpy(data["Z_sh"]).float() if "Z_sh" in data else None
        self.Z_pr = torch.from_numpy(data["Z_pr"]).float() if "Z_pr" in data else None
        self.proj_id = torch.from_numpy(data["proj_id"]).long()
        if "y" in data:
            self.y_npz = torch.from_numpy(data["y"]).float()
        else:
            self.y_npz = None

        if len(self.meta) != self.H.shape[0]:
            raise ValueError("Length mismatch between meta and H_file")

        # sanity: meta y vs npz y (if present)
        meta_y = torch.tensor([float(m["y"]) for m in self.meta], dtype=torch.float)
        if self.y_npz is not None:
            if self.y_npz.numel() != meta_y.numel():
                raise ValueError("y length mismatch")
            if not torch.allclose(self.y_npz, meta_y):
                raise ValueError("meta y and npz y mismatch")
        self.y = meta_y

    def get(self, uids):
        idx = [self.uid_to_idx[u] for u in uids]
        return self.H[idx], self.y[idx], self.proj_id[idx]

    def get_features(self, uids, key: str = "H_file"):
        idx = [self.uid_to_idx[u] for u in uids]
        if key == "H_file":
            feats = self.H
        elif key == "Z_sh":
            if self.Z_sh is None:
                raise ValueError("Z_sh not found in repr.npz")
            feats = self.Z_sh
        elif key == "Z_pr":
            if self.Z_pr is None:
                raise ValueError("Z_pr not found in repr.npz")
            feats = self.Z_pr
        else:
            raise ValueError(f"Unknown feature key: {key}")
        return feats[idx], self.y[idx], self.proj_id[idx]

    def __len__(self):
        return self.H.shape[0]
