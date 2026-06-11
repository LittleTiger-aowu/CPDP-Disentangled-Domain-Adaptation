# -*- coding: utf-8 -*-
"""
Generate 2 ablation figures (PDF + SVG) from hard-coded results (no file I/O).

Outputs:
  - fig_ablation_metrics_intervals.pdf / .svg
  - fig_scorr_vs_mcc_intervals.pdf / .svg

Run:
  python plot_ablation_from_table.py
"""

from __future__ import annotations

import math
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


def _save_both(fig, base_name: str) -> None:
    """Save figure as PDF (vector) + SVG (XML editable)."""
    fig.savefig(f"{base_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_name}.svg", bbox_inches="tight")


def main() -> None:
    # -----------------------------
    # Hard-coded data (from your pasted tables)
    # -----------------------------
    # Metrics table (Mylyn-3.1)
    metrics: Dict[str, Dict[str, float]] = {
        "full_method": {
            "roc_auc": 0.7850131795353132,
            "default_f1": 0.0,
            "default_mcc": -0.007000130575423803,
            "best_f1": 0.31199068684516884,
            "best_mcc": 0.2828282708025004,
        },
        "w_o_orthogonality": {
            "roc_auc": 0.7850063903348942,
            "default_f1": 0.0,
            "default_mcc": -0.007000130575423803,
            "best_f1": 0.30146231721034866,
            "best_mcc": 0.2770106907984141,
        },
        "w_o_project_supervision": {
            "roc_auc": 0.7839353939688138,
            "default_f1": 0.0,
            "default_mcc": -0.0049490221167685375,
            "best_f1": 0.3051948051948052,
            "best_mcc": 0.25762054228401254,
        },
        "w_o_gcn": {
            "roc_auc": 0.4967844649516015,
            "default_f1": 0.13790074659639878,
            "default_mcc": 0.039929263760686166,
            "best_f1": 0.14232765011119347,
            "best_mcc": 0.07642680178317166,
        },
    }

    # S_corr table (Mylyn-3.1)
    scorr: Dict[str, float] = {
        "full_method": 0.3776610791683197,
        "w_o_orthogonality": 0.46621227264404297,
        "w_o_project_supervision": 0.8638637661933899,
        "w_o_gcn": 0.7998971939086914,
    }

    # Plot order (match your discussion: full -> remove ortho -> remove project sup -> remove gcn)
    order: List[str] = [
        "full_method",
        "w_o_orthogonality",
        "w_o_project_supervision",
        "w_o_gcn",
    ]

    # Short x labels (still editable in SVG)
    xlabels = {
        "full_method": "full_method",
        "w_o_orthogonality": "w_o_orthogonality",
        "w_o_project_supervision": "w_o_project_supervision",
        "w_o_gcn": "w_o_gcn",
    }

    # -----------------------------
    # Figure 1: AUC / MCC / F1 with interval [default@0.5, best]
    # -----------------------------
    x = np.arange(len(order), dtype=float)

    auc = np.array([metrics[v]["roc_auc"] for v in order], dtype=float)

    mcc_best = np.array([metrics[v]["best_mcc"] for v in order], dtype=float)
    mcc_def = np.array([metrics[v]["default_mcc"] for v in order], dtype=float)
    # asymmetric yerr: draw interval from best down to default
    mcc_yerr = np.vstack([mcc_best - mcc_def, np.zeros_like(mcc_best)])

    f1_best = np.array([metrics[v]["best_f1"] for v in order], dtype=float)
    f1_def = np.array([metrics[v]["default_f1"] for v in order], dtype=float)
    f1_yerr = np.vstack([f1_best - f1_def, np.zeros_like(f1_best)])

    fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

    # (a) AUC
    axes[0].bar(x, auc)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Zero-shot ablation metrics on target Mylyn-3.1 (interval = default@0.5 → best)")
    for xi, val in zip(x, auc):
        axes[0].text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # (b) best MCC with interval to default MCC@0.5
    axes[1].bar(x, mcc_best, yerr=mcc_yerr, capsize=6)
    axes[1].set_ylabel("best MCC")
    for xi, lo, hi in zip(x, mcc_def, mcc_best):
        # interval label
        axes[1].text(
            xi,
            hi,
            f"[{lo:.3f}, {hi:.3f}]",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        # also mark default point
        axes[1].scatter([xi], [lo], s=20)

    # (c) best F1 with interval to default F1@0.5
    axes[2].bar(x, f1_best, yerr=f1_yerr, capsize=6)
    axes[2].set_ylabel("best F1")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([xlabels[v] for v in order], rotation=15, ha="right")
    for xi, lo, hi in zip(x, f1_def, f1_best):
        axes[2].text(
            xi,
            hi,
            f"[{lo:.3f}, {hi:.3f}]",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        axes[2].scatter([xi], [lo], s=20)

    fig1.tight_layout()
    _save_both(fig1, "fig_ablation_metrics_intervals")
    plt.close(fig1)

    # -----------------------------
    # Figure 2: S_corr vs MCC (interval = default@0.5 → best)
    # -----------------------------
    xs = np.array([scorr[v] for v in order], dtype=float)
    ys_best = np.array([metrics[v]["best_mcc"] for v in order], dtype=float)
    ys_def = np.array([metrics[v]["default_mcc"] for v in order], dtype=float)

    fig2 = plt.figure(figsize=(9, 6))
    ax = plt.gca()

    # Draw interval segments at fixed x: from default to best
    for v, xv, y0, y1 in zip(order, xs, ys_def, ys_best):
        ax.plot([xv, xv], [y0, y1], linewidth=2)   # interval line
        ax.scatter([xv], [y1], s=50)               # best point
        ax.scatter([xv], [y0], s=25)               # default point
        ax.text(xv, y1, f"  {xlabels[v]}", va="center", fontsize=9)

    ax.set_xlabel(r"$S_{\mathrm{corr}} = \|\mathbf{C}\|_F$")
    ax.set_ylabel("MCC on target Mylyn-3.1")
    ax.set_title(r"$S_{\mathrm{corr}}$ vs MCC (interval = default@0.5 → best)")
    ax.grid(True, linestyle="--", linewidth=0.5)

    # A small legend-like note without creating a formal legend (keeps SVG simpler to edit)
    ax.text(
        0.02,
        0.02,
        "Vertical segment: default@0.5 → best\nSmall dot: default@0.5, Large dot: best",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
    )

    fig2.tight_layout()
    _save_both(fig2, "fig_scorr_vs_mcc_intervals")
    plt.close(fig2)

    print("Saved:")
    print("  - fig_ablation_metrics_intervals.pdf / .svg")
    print("  - fig_scorr_vs_mcc_intervals.pdf / .svg")


if __name__ == "__main__":
    main()
