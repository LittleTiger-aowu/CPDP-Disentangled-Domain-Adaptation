import json
import pandas as pd
from pathlib import Path

# ====== 配置 ======
ROOT_DIR = r"E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D"
OUT_CSV = r"E:\project\WYP\LineDefStudy2.0\outputs\domain_probe_summary.csv"
# ================

def parse_variant_target(path: Path):
    parts = path.parts
    variant = None
    target = None
    for i, p in enumerate(parts):
        if p.startswith("target_"):
            target = p.replace("target_", "")
            if i > 0:
                variant = parts[i - 1]
            break
    return variant, target

def main():
    files = list(Path(ROOT_DIR).rglob("domain_probe.json"))
    if not files:
        print("❌ 未找到 domain_probe.json")
        return

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        variant, target = parse_variant_target(fp)
        rows.append({
            "variant": variant,
            "target_project": target,
            "path": str(fp),
            "num_classes": data.get("num_classes"),
            "num_samples_total": data.get("num_samples_total"),
            "num_samples_used": data.get("num_samples_used"),
            "z_sh_ce_test": data.get("z_sh_ce_test"),
            "z_pr_ce_test": data.get("z_pr_ce_test"),
            "h_file_ce_test": data.get("h_file_ce_test"),
            "z_sh_ce_train": data.get("z_sh_ce_train"),
            "z_pr_ce_train": data.get("z_pr_ce_train"),
            "h_file_ce_train": data.get("h_file_ce_train"),
            "z_sh_acc_test": data.get("z_sh_acc_test"),
            "z_pr_acc_test": data.get("z_pr_acc_test"),
            "h_file_acc_test": data.get("h_file_acc_test"),
            "source_projects": ",".join(data.get("source_projects", [])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(df)
    print(f"\n✅ 汇总完成: {OUT_CSV}")

if __name__ == "__main__":
    main()
