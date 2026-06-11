"""
Utilities for constructing strict CPDP tasks using cached representations.

Key rules:
  - Projects are treated as domains. Each project may have multiple versions; they are merged.
  - Source = one or more projects; Target = a single project.
  - Source is split by sha1 groups: 80% train / 20% val (grouped so same sha1 stays in one split).
  - Target is test only.
  - Leakage removal: any target sample whose sha1 appears in source (train+val) is dropped.
  - Optional intra-project dedup (by sha1, y = max(y)) can be enabled, but it must not reorder meta/repr files;
    dedup is done on the uid list selection only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
from collections import defaultdict


@dataclass
class Split:
    uids: List[str]


@dataclass
class TaskSplits:
    source_train: Split
    source_val: Split
    target_test: Split
    info: Dict


def _load_meta(meta_path: Path) -> List[dict]:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _group_by_sha(rows: List[dict]) -> Dict[str, List[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[r["sha1"]].append(r)
    return g


def _dedup_by_sha(rows: List[dict]) -> List[dict]:
    g = _group_by_sha(rows)
    deduped = []
    for sha, items in g.items():
        # keep max y; uid choose first
        y_max = max(int(it["y"]) for it in items)
        keep = items[0].copy()
        keep["y"] = y_max
        deduped.append(keep)
    return deduped


def build_cpdp_task(
    meta_path: Path,
    source_projects: List[str],
    target_project: str,
    train_ratio: float = 0.8,
    seed: int = 0,
    dedup_intra_project: bool = False,
) -> TaskSplits:
    rows = _load_meta(meta_path)
    rng = random.Random(seed)

    src_rows = [r for r in rows if r["project"] in source_projects]
    tgt_rows = [r for r in rows if r["project"] == target_project]

    if dedup_intra_project:
        src_rows = _dedup_by_sha(src_rows)
        tgt_rows = _dedup_by_sha(tgt_rows)

    # group source by sha1 for grouped split
    sha_groups = list(_group_by_sha(src_rows).values())
    rng.shuffle(sha_groups)
    n_train = int(len(sha_groups) * train_ratio)
    train_groups = sha_groups[:n_train]
    val_groups = sha_groups[n_train:]

    source_train = [u for g in train_groups for u in g]
    source_val = [u for g in val_groups for u in g]

    # leakage removal: drop any target sample whose sha1 appears in source train+val
    source_sha = {r["sha1"] for r in source_train + source_val}
    target_clean = [r for r in tgt_rows if r["sha1"] not in source_sha]

    splits = TaskSplits(
        source_train=Split([r["uid"] for r in source_train]),
        source_val=Split([r["uid"] for r in source_val]),
        target_test=Split([r["uid"] for r in target_clean]),
        info={
            "source_projects": source_projects,
            "target_project": target_project,
            "train_ratio": train_ratio,
            "seed": seed,
            "dedup_intra_project": dedup_intra_project,
            "n_source_train": len(source_train),
            "n_source_val": len(source_val),
            "n_target_test": len(target_clean),
            "n_leak_dropped": len(tgt_rows) - len(target_clean),
        },
    )
    return splits


def save_splits(task: TaskSplits, out_path: Path) -> None:
    out = {
        "source_train": task.source_train.uids,
        "source_val": task.source_val.uids,
        "target_test": task.target_test.uids,
        "info": task.info,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
