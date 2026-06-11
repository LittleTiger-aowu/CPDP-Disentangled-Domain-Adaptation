"""
Prepare UBD class-level parquet data with strict class-level source slices.

Example:
  python src/ch3/prepare_ubd_class_parquet.py --ubd-root data/UBD \
    --out-parquet data/processed/ubd_class.parquet \
    --out-project-vocab data/processed/ubd_project_vocab.json \
    --out-baseline-csv-dir outputs/ubd_baseline_csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


def canonicalize_src(src: str) -> str:
    if not isinstance(src, str):
        return ""
    normalized = src.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join([line.rstrip() for line in normalized.split("\n")])


def sha1_of_src(src: str) -> str:
    canon = canonicalize_src(src)
    try:
        return hashlib.sha1(canon.encode("utf-8"), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def extract_all_zips(src_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = []
    for zp in sorted(src_dir.glob("*.zip")):
        dst = out_dir / zp.stem
        if not dst.exists():
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(dst)
        roots.append(dst)
    return roots


def _add_index_entry(
    idx: dict[str, tuple[Path, str]],
    rel: str,
    abs_path: Path,
    project: str,
) -> None:
    if rel not in idx:
        idx[rel] = (abs_path, project)


def norm_rel(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./").casefold()


def candidate_rels(rel_path: str) -> list[str]:
    p = norm_rel(rel_path)
    cands = [p]
    if p.startswith("src/"):
        cands.append(p[len("src/") :])
    else:
        cands.append(f"src/{p}")
    seen = set()
    out = []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def choose_src_root(project_root: Path) -> Path:
    inner = project_root / project_root.name
    if inner.exists() and inner.is_dir():
        return inner
    return project_root


def build_java_index(src_root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in src_root.rglob("*.java"):
        rel = p.relative_to(src_root).as_posix()
        rel_norm = norm_rel(rel)
        idx.setdefault(rel_norm, p)
    return idx


def build_project_indices(roots: list[Path]) -> tuple[dict[str, dict[str, Path]], dict[str, Path]]:
    index_by_project: dict[str, dict[str, Path]] = {}
    src_root_by_project: dict[str, Path] = {}
    for root in roots:
        project = root.name
        src_root = choose_src_root(root)
        src_root_by_project[project] = src_root
        index_by_project[project] = build_java_index(src_root)
    return index_by_project, src_root_by_project


def build_global_index(index_by_project: dict[str, dict[str, Path]]) -> dict[str, tuple[Path, str]]:
    global_index: dict[str, tuple[Path, str]] = {}
    for project, idx in index_by_project.items():
        for rel_norm, path in idx.items():
            global_index.setdefault(rel_norm, (path, project))
    return global_index


def suffix_match_path(target: str, index: dict[str, Path]) -> Optional[Path]:
    best = None
    best_len = None
    for rel_norm, path in index.items():
        if not rel_norm.endswith(target):
            continue
        cur_len = len(rel_norm)
        if best is None or cur_len < best_len:
            best = path
            best_len = cur_len
    return best


def suffix_match_global(target: str, index: dict[str, tuple[Path, str]]) -> Optional[tuple[Path, str]]:
    best = None
    best_len = None
    for rel_norm, payload in index.items():
        if not rel_norm.endswith(target):
            continue
        cur_len = len(rel_norm)
        if best is None or cur_len < best_len:
            best = payload
            best_len = cur_len
    return best


def is_template_placeholder(rel_path: str) -> bool:
    # Skip template placeholders based on filename like "$actionClass$.java".
    if not rel_path:
        return False
    filename = rel_path.replace("\\", "/").split("/")[-1]
    return bool(re.match(r"^\$[^$]+\$\.java$", filename))


def candidate_paths(rel_path: str) -> list[str]:
    p = (rel_path or "").replace("\\", "/").lstrip("./")
    cands = [p]
    if p.startswith("src/"):
        cands.append(p[len("src/") :])
    return cands


def _should_log(current_len: int, max_samples: int) -> bool:
    return max_samples < 0 or current_len < max_samples


def extract_span_by_line(
    java_text: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> Optional[tuple[str, int]]:
    """
    Return (snippet, base_line) from [start_line, end_line] (1-based, inclusive).
    base_line is 1-based line index of snippet in original file.
    """
    if not java_text:
        return None
    if start_line is None or end_line is None:
        return None
    try:
        start = int(start_line)
        end = int(end_line)
    except Exception:
        return None
    if start <= 0 or end <= 0 or end < start:
        return None
    lines = java_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return None
    if start > len(lines):
        return None
    end = min(end, len(lines))
    snippet = "\n".join(lines[start - 1 : end])
    return snippet, start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubd-root", required=True, help="Path to UBD/BugPrediction root.")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--csv-glob", default="csv-class/*-Unified.csv")
    parser.add_argument("--src-zip-dir", default="src")
    parser.add_argument("--extract-class", type=int, default=1)
    parser.add_argument("--out-parquet", required=True)
    parser.add_argument("--out-project-vocab", required=True)
    parser.add_argument("--out-baseline-csv-dir", default=None)
    parser.add_argument("--log-missing", type=int, default=1)
    parser.add_argument("--log-dir", default="outputs/ubd_logs")
    parser.add_argument("--max-miss-samples", type=int, default=50)
    args = parser.parse_args()

    ubd_root = Path(args.ubd_root)
    src_dir = ubd_root / args.src_zip_dir
    extracted_dir = ubd_root / "src_extracted"

    if args.csv:
        csv_paths = [ubd_root / args.csv]
    else:
        csv_paths = [p for p in ubd_root.glob(args.csv_glob)]
    if not csv_paths:
        raise FileNotFoundError("No CSV files found for the given csv or csv_glob.")

    roots = extract_all_zips(src_dir, extracted_dir)
    index_by_project, _ = build_project_indices(roots)
    global_index = build_global_index(index_by_project)

    rows = []
    miss_path = 0
    miss_span = 0
    template_count = 0
    miss_path_samples = []
    miss_span_samples = []
    template_skip_samples = []
    keep_ids = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "Type" in df.columns:
            df = df[df["Type"].astype(str).str.lower() == "class"].copy()

        stem = csv_path.stem
        project_hint = stem
        for suffix in ("-Unified", "-unified"):
            if project_hint.endswith(suffix):
                project_hint = project_hint[: -len(suffix)]
                break
        project_hint = project_hint.strip()
        hint_index = index_by_project.get(project_hint)

        for _, r in df.iterrows():
            rel_path = str(r.get("Path", "")).replace("\\", "/")
            name = str(r.get("Name", ""))
            longname = str(r.get("LongName", ""))
            is_template = is_template_placeholder(rel_path)
            if is_template:
                template_count += 1
                if args.log_missing and _should_log(len(template_skip_samples), args.max_miss_samples):
                    template_skip_samples.append(
                        {
                            "csv": str(csv_path),
                            "rel_path": rel_path,
                            "id": r.get("ID", ""),
                            "name": name,
                            "classname": r.get("classname", ""),
                            "longname": longname,
                        }
                    )
            hit = None
            rel_norms = candidate_rels(rel_path)
            if hint_index is not None:
                for cand in rel_norms:
                    hit = hint_index.get(cand)
                    if hit is not None:
                        hit = (hit, project_hint)
                        break
            if hit is None:
                for cand in rel_norms:
                    hit = global_index.get(cand)
                    if hit is not None:
                        break
            if hit is None:
                target = norm_rel(rel_path)
                if hint_index is not None:
                    path = suffix_match_path(target, hint_index)
                    if path is not None:
                        hit = (path, project_hint)
                if hit is None:
                    hit = suffix_match_global(target, global_index)
            missing_src = False
            if hit is not None:
                abs_java, project = hit
                if not abs_java.exists():
                    abs_java = None
            else:
                abs_java, project = None, project_hint
            if abs_java is None:
                miss_path += 1
                if args.log_missing and _should_log(len(miss_path_samples), args.max_miss_samples):
                    miss_path_samples.append(
                        {
                            "csv": str(csv_path),
                            "rel_path": rel_path,
                            "id": r.get("ID", ""),
                            "name": name,
                            "classname": r.get("classname", ""),
                            "longname": longname,
                        }
                    )
                missing_src = True

            text = ""
            if not missing_src:
                text = abs_java.read_text(encoding="utf-8", errors="ignore")

            bug_val = r.get("bug", 0)
            try:
                y = 1 if float(bug_val) > 0 else 0
            except Exception:
                y = 0

            base_line = 1
            src_snip = text
            if bool(args.extract_class):
                if not missing_src:
                    got = extract_span_by_line(text, r.get("Line", None), r.get("EndLine", None))
                    if got is not None:
                        src_snip, base_line = got
                    else:
                        miss_span += 1
                        missing_src = True
                        src_snip = ""
                        if args.log_missing and _should_log(len(miss_span_samples), args.max_miss_samples):
                            miss_span_samples.append(
                                {
                                    "csv": str(csv_path),
                                    "rel_path": rel_path,
                                    "id": r.get("ID", ""),
                                    "classname": r.get("classname", ""),
                                    "longname": r.get("LongName", ""),
                                    "line": r.get("Line", ""),
                                    "end_line": r.get("EndLine", ""),
                                }
                            )

            uid = f"{project}::unknown::{r.get('ID', rel_path)}"
            keep_ids.append(r.get("ID", rel_path))
            rows.append(
                {
                    "uid": uid,
                    "project": project,
                    "version": "unknown",
                    "file_path": str(abs_java.resolve()) if abs_java is not None else rel_path,
                    "class_name": str(r.get("classname", r.get("LongName", ""))),
                    "base_line": int(base_line),
                    "y": int(y),
                    "src": src_snip,
                    "sha1": sha1_of_src(src_snip),
                    "is_template": int(is_template),
                    "missing_src": int(missing_src),
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    projects = sorted(out_df["project"].unique().tolist())
    project_vocab = {p: i for i, p in enumerate(projects)}
    vocab_path = Path(args.out_project_vocab)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(project_vocab, indent=2), encoding="utf-8")

    if args.out_baseline_csv_dir:
        out_dir = Path(args.out_baseline_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        proj_by_path = {rel: proj for rel, (_, proj) in global_index.items()}
        for csv_path in csv_paths:
            df2 = pd.read_csv(csv_path)
            if "Type" in df2.columns:
                df2 = df2[df2["Type"].astype(str).str.lower() == "class"].copy()
            rel_paths = df2["Path"].astype(str).str.replace("\\", "/")
            df2["project"] = rel_paths.map(proj_by_path)
            df2["y"] = pd.to_numeric(df2.get("bug", 0), errors="coerce").fillna(0).gt(0).astype(int)
            df2 = df2[~df2["project"].isna()]
            for project, g in df2.groupby("project"):
                out_csv = out_dir / f"{project}.csv"
                if out_csv.exists():
                    g.to_csv(out_csv, index=False, mode="a", header=False)
                else:
                    g.to_csv(out_csv, index=False)

    if args.log_missing:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        miss_path_log = log_dir / "missing_path_samples.json"
        miss_span_log = log_dir / "missing_span_samples.json"
        template_skip_log = log_dir / "template_skip_samples.json"
        keep_ids_log = log_dir / "keep_ids.json"
        miss_path_log.write_text(
            json.dumps(miss_path_samples, indent=2),
            encoding="utf-8",
        )
        miss_span_log.write_text(
            json.dumps(miss_span_samples, indent=2),
            encoding="utf-8",
        )
        template_skip_log.write_text(
            json.dumps(template_skip_samples, indent=2),
            encoding="utf-8",
        )
        keep_ids_log.write_text(
            json.dumps(keep_ids, indent=2),
            encoding="utf-8",
        )

    print(f"Saved {len(out_df)} rows to {out_path}")
    print(f"Saved project vocab with {len(project_vocab)} entries to {vocab_path}")
    print(f"Missing path: {miss_path} | Missing span: {miss_span} | Template count: {template_count}")
    if args.log_missing:
        print(
            f"Logged missing samples: path={len(miss_path_samples)} span={len(miss_span_samples)} "
            f"to {miss_path_log} and {miss_span_log}"
        )
        print(f"Logged template skips: {len(template_skip_samples)} to {template_skip_log}")
        print(f"Saved keep IDs to {keep_ids_log}")


if __name__ == "__main__":
    main()
