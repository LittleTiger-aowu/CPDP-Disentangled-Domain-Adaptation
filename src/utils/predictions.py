import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

# =========================
# 0) 你 predictions.csv 的列名（按需修改）
# =========================
Y_COL = "y_true"
S_COL = "y_score"

# =========================
# 1) 你的“完整表格”（原样粘贴）
#    字段说明（按你当前导出最合理的语义命名）：
#    - precision_fix/recall_fix/f1_fix/mcc_fix：某固定阈值下（你之前提到过）计算的指标
#    - best_f1, tau_f1, prec@tau_f1, rec@tau_f1, f1@tau_f1
#    - best_mcc, tau_src (=tau_mcc), prec@tau_src, rec@tau_src, f1@tau_src
#    - roc_auc 作为阈值无关指标
# =========================
RAW = r"""
variant,target_project,pred_path,n_samples,pos_ratio,roc_auc,precision_fix,recall_fix,f1_fix,mcc_fix,best_f1,tau_f1,prec_at_tau_f1,rec_at_tau_f1,f1_at_tau_f1,best_mcc,tau_src,prec_at_tau_src,rec_at_tau_src,f1_at_tau_src
A_codebert_only,Eclipse_JDT_Core-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\A_codebert_only\target_Eclipse_JDT_Core-3.4\direct_eval\predictions.csv,2080,0.09903846153846153,0.5929350022277253,0.0,0.0,0.0,0.0,0.2142857142857143,0.3466,0.12520458265139117,0.7427184466019418,0.10453849363690432,0.12008585566242579,0.307,0.11353467561521252,0.9854368932038835,0.20361083249749248
A_codebert_only,Equinox-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\A_codebert_only\target_Equinox-3.4\direct_eval\predictions.csv,600,0.21,0.7802056124840935,0.0,0.0,0.0,0.0,0.5,0.3565,0.3944954128440367,0.6825396825396826,0.3421834061195057,0.3421834061195057,0.3565,0.3944954128440367,0.6825396825396826,0.5
A_codebert_only,lucene-2.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\A_codebert_only\target_lucene-2.4\direct_eval\predictions.csv,1442,0.043689320388349516,0.4921498210113148,0.0,0.0,0.0,0.0,0.09990009990009989,0.3664,0.053304904051172705,0.7936507936507936,0.0641762969913203,0.0641762969913203,0.3664,0.053304904051172705,0.7936507936507936,0.09990009990009989
A_codebert_only,Mylyn-3.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\A_codebert_only\target_Mylyn-3.1\direct_eval\predictions.csv,3028,0.0690224570673712,0.533278963153312,0.0,0.0,0.0,0.0,0.15692821368948248,0.3961,0.08596250571559214,0.8995215311004785,0.10776462863744957,0.10776462863744957,0.3961,0.08596250571559214,0.8995215311004785,0.15692821368948248
A_codebert_only,PDE_UI-3.4.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\A_codebert_only\target_PDE_UI-3.4.1\direct_eval\predictions.csv,2957,0.07034156239431856,0.47733616419956915,0.0,0.0,0.0,0.0,0.1523487092678798,0.3961,0.08352668213457076,0.8653846153846154,0.08451870193677956,0.08451870193677956,0.3961,0.08352668213457076,0.8653846153846154,0.1523487092678798
B_shallow_struct,Eclipse_JDT_Core-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\B_shallow_struct\target_Eclipse_JDT_Core-3.4\direct_eval\predictions.csv,2080,0.09903846153846153,0.6253406347462983,0.14042126379137412,0.6796116504854369,0.2327514546965919,0.1329224012086367,0.23802395209580837,0.4852,0.1407079646017699,0.7718446601941747,0.15213904010990367,0.15213904010990367,0.4852,0.1407079646017699,0.7718446601941747,0.23802395209580837
B_shallow_struct,Equinox-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\B_shallow_struct\target_Equinox-3.4\direct_eval\predictions.csv,600,0.21,0.5892103676913804,0.0,0.0,0.0,0.0,0.4029038112522686,0.2674,0.2611764705882353,0.8809523809523809,0.19580445956991477,0.19580445956991477,0.2674,0.2611764705882353,0.8809523809523809,0.4029038112522686
B_shallow_struct,lucene-2.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\B_shallow_struct\target_lucene-2.4\direct_eval\predictions.csv,1442,0.043689320388349516,0.7257386880302036,0.0,0.0,0.0,0.0,0.15649452269170577,0.2971,0.08680555555555555,0.7936507936507936,0.17203070941974963,0.18600723032597347,0.2773,0.08160442600276625,0.9365079365079365,0.15012722646310434
B_shallow_struct,Mylyn-3.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\B_shallow_struct\target_Mylyn-3.1\direct_eval\predictions.csv,3028,0.0690224570673712,0.6449825262954219,0.08421052631578947,0.11483253588516747,0.09716599190283401,0.019312890788550568,0.17963386727688788,0.4159,0.10201429499675113,0.7511961722488039,0.13231639535069128,0.13926000444124298,0.3664,0.09188935771214252,0.937799043062201,0.16737830913748933
B_shallow_struct,PDE_UI-3.4.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\B_shallow_struct\target_PDE_UI-3.4.1\direct_eval\predictions.csv,2957,0.07034156239431856,0.6897473207040322,0.0,0.0,0.0,-0.008765963932734915,0.2020997375328084,0.3862,0.11702127659574468,0.7403846153846154,0.16346851961378633,0.16346851961378633,0.3862,0.11702127659574468,0.7403846153846154,0.2020997375328084
C_multimodal_no_disentangle,Eclipse_JDT_Core-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\C_multimodal_no_disentangle\target_Eclipse_JDT_Core-3.4\direct_eval\predictions.csv,2080,0.09903846153846153,0.843497114318575,0.5631067961165048,0.2815533980582524,0.37540453074433655,0.35460278715666294,0.4625850340136055,0.3763,0.4340425531914894,0.49514563106796117,0.4002494914824754,0.40361032755038223,0.406,0.46766169154228854,0.4563106796116505,0.4619164619164619
C_multimodal_no_disentangle,Equinox-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\C_multimodal_no_disentangle\target_Equinox-3.4\direct_eval\predictions.csv,600,0.21,0.7960116536065903,1.0,0.007936507936507936,0.015748031496062992,0.07924839714841088,0.5241730279898218,0.01,0.3857677902621723,0.8174603174603174,0.38641111810612416,0.38641111810612416,0.01,0.3857677902621723,0.8174603174603174,0.5241730279898218
C_multimodal_no_disentangle,lucene-2.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\C_multimodal_no_disentangle\target_lucene-2.4\direct_eval\predictions.csv,1442,0.043689320388349516,0.8433762675967171,0.5,0.047619047619047616,0.08695652173913042,0.14430184737580107,0.2887700534759359,0.109,0.21774193548387097,0.42857142857142855,0.26118398442092067,0.26118398442092067,0.109,0.21774193548387097,0.42857142857142855,0.2887700534759359
C_multimodal_no_disentangle,Mylyn-3.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\C_multimodal_no_disentangle\target_Mylyn-3.1\direct_eval\predictions.csv,3028,0.0690224570673712,0.810396302601452,0.39622641509433965,0.10047846889952153,0.1603053435114504,0.1722852859645139,0.29090909090909095,0.2179,0.2517482517482518,0.3444976076555024,0.232800731440201,0.25114645763206445,0.0595,0.14,0.9043062200956937,0.24246311738293777
C_multimodal_no_disentangle,PDE_UI-3.4.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\C_multimodal_no_disentangle\target_PDE_UI-3.4.1\direct_eval\predictions.csv,2957,0.07034156239431856,0.8184339759912695,0.17142857142857143,0.028846153846153848,0.04938271604938272,0.04326348999221601,0.3087885985748218,0.0991,0.20504731861198738,0.625,0.2751936861064496,0.2836471667006404,0.0595,0.18,0.7788461538461539,0.2924187725631769
D_full,Eclipse_JDT_Core-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Eclipse_JDT_Core-3.4\direct_eval\predictions.csv,2080,0.09903846153846153,0.8431720218420697,0.41365461847389556,0.5,0.4527472527472527,0.38840199745258375,0.4773869346733668,0.6138,0.4947916666666667,0.46116504854368934,0.42249248817085283,0.42296962908402236,0.6435,0.5082872928176796,0.44660194174757284,0.47545219638242897
D_full,Equinox-3.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Equinox-3.4\direct_eval\predictions.csv,600,0.21,0.7961288594199986,0.6666666666666666,0.047619047619047616,0.08888888888888889,0.1383577955197987,0.5223529411764706,0.01,0.3712374581939799,0.8809523809523809,0.39454368498412956,0.39454368498412956,0.01,0.3712374581939799,0.8809523809523809,0.5223529411764706
D_full,lucene-2.4,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_lucene-2.4\direct_eval\predictions.csv,1442,0.043689320388349516,0.7599594829471552,0.16666666666666666,0.031746031746031744,0.05333333333333334,0.05511382768417566,0.17079889807162532,0.0199,0.10333333333333333,0.49206349206349204,0.1495571855510362,0.15750606066330602,0.01,0.09172259507829977,0.6507936507936508,0.1607843137254902
D_full,Mylyn-3.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_Mylyn-3.1\direct_eval\predictions.csv,3028,0.0690224570673712,0.8114028015635528,0.2682926829268293,0.05263157894736842,0.088,0.09209836491522022,0.3156089193825043,0.2377,0.24598930481283424,0.44019138755980863,0.2620670660872739,0.2784998492998758,0.0694,0.1826809015421115,0.7368421052631579,0.29277566539923955
D_full,PDE_UI-3.4.1,E:\project\WYP\LineDefStudy2.0\outputs\ch3_optimized_D\D_full\target_PDE_UI-3.4.1\direct_eval\predictions.csv,2957,0.07034156239431856,0.8038945980356493,0.20869565217391303,0.23076923076923078,0.21917808219178084,0.15712512553566424,0.2829827915869981,0.2278,0.1766109785202864,0.7115384615384616,0.26133472009254677,0.2647999369355069,0.208,0.17062634989200864,0.7596153846153846,0.27865961199294537
""".strip()

df = pd.read_csv(StringIO(RAW))

# 目标项目显示名（论文里短写）
TARGET_ORDER = [
    "Eclipse_JDT_Core-3.4",
    "Equinox-3.4",
    "Mylyn-3.1",
    "PDE_UI-3.4.1",
    "lucene-2.4",
]
TARGET_ALIAS = {
    "Eclipse_JDT_Core-3.4": "JDT",
    "Equinox-3.4": "EQ",
    "Mylyn-3.1": "MY",
    "PDE_UI-3.4.1": "PDE",
    "lucene-2.4": "LC",
}
METHOD_ORDER = [
    "A_codebert_only",
    "B_shallow_struct",
    "C_multimodal_no_disentangle",
    "D_full",
]
METHOD_ALIAS = {
    "A_codebert_only": "CodeBERT-only",
    "B_shallow_struct": "Shallow-Struct",
    "C_multimodal_no_disentangle": "Multi-modal (no disent.)",
    "D_full": "Ours (full)",
}

# =========================
# 2) 从 predictions.csv 读分数，bootstrap 置信区间
# =========================
def load_preds(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictions.csv not found: {path}")
    x = pd.read_csv(path)
    if Y_COL not in x.columns or S_COL not in x.columns:
        raise ValueError(
            f"Missing columns in {path}. Need '{Y_COL}' and '{S_COL}'. "
            f"Got columns: {list(x.columns)[:20]}"
        )
    y = x[Y_COL].astype(int).to_numpy()
    s = x[S_COL].astype(float).to_numpy()
    return pd.DataFrame({"y": y, "s": s})

def metrics_at_tau(y, s, tau):
    y_pred = (s >= tau).astype(int)
    # 注意：roc_auc_score 要求两类都存在，否则会报错
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan
    mcc = matthews_corrcoef(y, y_pred) if len(np.unique(y_pred)) == 2 else 0.0
    return auc, ap, mcc

def bootstrap_ci(y, s, tau, n_boot=800, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, mccs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # bootstrap resample
        yy, ss = y[idx], s[idx]
        auc, _, mcc = metrics_at_tau(yy, ss, tau)
        if not np.isnan(auc):
            aucs.append(auc)
        mccs.append(mcc)
    # 95% CI
    def ci(arr):
        arr = np.asarray(arr, dtype=float)
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)
    return ci(aucs), ci(mccs)

records = []
# 如果你不想从 predictions.csv 重算（只用表格里的点估计），把 RECOMPUTE=False
RECOMPUTE = True
N_BOOT = 800

for _, r in df.iterrows():
    method = r["variant"]
    tgt = r["target_project"]
    tau = float(r["tau_src"])
    if RECOMPUTE:
        pr = load_preds(r["pred_path"])
        y, s = pr["y"].to_numpy(), pr["s"].to_numpy()
        auc, ap, mcc = metrics_at_tau(y, s, tau)
        (auc_lo, auc_hi), (mcc_lo, mcc_hi) = bootstrap_ci(y, s, tau, n_boot=N_BOOT, seed=123)
    else:
        auc = float(r["roc_auc"])
        mcc = float(r["best_mcc"])
        auc_lo, auc_hi = np.nan, np.nan
        mcc_lo, mcc_hi = np.nan, np.nan

    records.append({
        "method": method,
        "target": tgt,
        "tau_src": tau,
        "roc_auc": auc,
        "roc_auc_lo": auc_lo,
        "roc_auc_hi": auc_hi,
        "mcc": mcc,
        "mcc_lo": mcc_lo,
        "mcc_hi": mcc_hi,
        "pos_ratio": float(r["pos_ratio"]),
        "n": int(r["n_samples"]),
    })

res = pd.DataFrame(records)

# =========================
# 3) 画“折线 + CI 阴影”：AUC（上）+ MCC（下）
# =========================
os.makedirs("figs", exist_ok=True)

# x 轴顺序固定
x_targets = TARGET_ORDER
x = np.arange(len(x_targets))
x_labels = [TARGET_ALIAS[t] for t in x_targets]

fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)

def plot_metric(ax, metric, lo, hi, title, ylim=None):
    for method in METHOD_ORDER:
        sub = res[(res["method"] == method)].set_index("target").reindex(x_targets)
        y = sub[metric].to_numpy(dtype=float)
        ylo = sub[lo].to_numpy(dtype=float)
        yhi = sub[hi].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", linewidth=1.8, label=METHOD_ALIAS.get(method, method))
        if np.all(np.isfinite(ylo)) and np.all(np.isfinite(yhi)):
            ax.fill_between(x, ylo, yhi, alpha=0.18)

    ax.set_ylabel(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

plot_metric(
    axes[0],
    metric="roc_auc",
    lo="roc_auc_lo",
    hi="roc_auc_hi",
    title="ROC-AUC (target, 95% CI)",
    ylim=(0.40, 0.90),
)

plot_metric(
    axes[1],
    metric="mcc",
    lo="mcc_lo",
    hi="mcc_hi",
    title=r"MCC@$\tau_{\mathrm{src}}$ (target, 95% CI)",
    ylim=(-0.05, 0.50),
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels)
axes[0].legend(ncol=2, frameon=False, loc="lower left")

# 在 x 轴下方补充每个 target 的缺陷比例，方便读者理解 PR/MCC 波动
pos_text = []
for t in x_targets:
    pr = res[(res["target"] == t)].iloc[0]["pos_ratio"]
    pos_text.append(f"{pr*100:.1f}%")
axes[1].set_xlabel("Target project (Bug ratio shown below)")
for xi, txt in zip(x, pos_text):
    axes[1].text(xi, -0.11, txt, ha="center", va="top", transform=axes[1].get_xaxis_transform(), fontsize=9)

plt.tight_layout()
out_pdf = os.path.join("figs", "zero_shot_auc_mcc_bar.pdf")
out_svg = os.path.join("figs", "zero_shot_auc_mcc_bar.svg")

plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
print("[OK] Saved:", out_pdf, out_svg)

print(f"[OK] Saved figure to: {out_pdf}")

# =========================
# 4) 可选：再输出一张“阈值 τ_src 分布图”（对这一小节非常加分）
# =========================
fig2, ax2 = plt.subplots(figsize=(10.5, 2.8))
for method in METHOD_ORDER:
    sub = res[res["method"] == method].set_index("target").reindex(x_targets)
    ax2.plot(x, sub["tau_src"].to_numpy(), marker="o", linewidth=1.8, label=METHOD_ALIAS.get(method, method))
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.set_ylabel(r"$\tau_{\mathrm{src}}$")
ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax2.legend(ncol=2, frameon=False, loc="upper right")
plt.tight_layout()
out_pdf2 = os.path.join("figs", "zero_shot_tau_src.pdf")
out_svg2 = os.path.join("figs", "zero_shot_tau_src.svg")

plt.savefig(out_pdf2, bbox_inches="tight")
plt.savefig(out_svg2, bbox_inches="tight")
print("[OK] Saved:", out_pdf2, out_svg2)

print(f"[OK] Saved figure to: {out_pdf2}")
