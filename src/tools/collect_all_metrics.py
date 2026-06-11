# save as tools/collect_all_metrics.py or run with python - << 'PY'
import json, csv, glob, os

root = "outputs/ch4_runs/loocv_f2_pct"  # 改成你的结果根目录
paths = glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True)

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

rows = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    run_dir = os.path.basename(os.path.dirname(p))
    m["run"] = run_dir
    if "target_project" not in m:
        if "target_" in run_dir:
            m["target_project"] = run_dir.split("target_", 1)[1]
    rows.append(flatten(m))

if not rows:
    print("No metrics.json found")
    raise SystemExit(0)

# collect all fields
fieldnames = sorted({k for r in rows for k in r.keys()})
out_csv = os.path.join(root, "summary_all_fields.csv")

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("Saved:", out_csv)
