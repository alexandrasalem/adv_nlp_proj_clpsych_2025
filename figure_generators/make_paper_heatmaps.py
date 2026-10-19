"""Rebuild the 3 paper-quality heatmaps:
1. System-level: 13×13 (6 metrics + 7 human), lower triangle + diagonal
2. Instance-level: 6 metrics × 7 human questions (full rectangle, no masking)
3. Instance-level: 6 metrics × 7 Mistral judge questions (full rectangle)

All: 8 discrete color bands (RdBu, blue=positive), paper-style labels, no titles.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import os

os.makedirs("outputs/final_analysis/figures", exist_ok=True)

# ── Paper labels ──
ALL_LABELS_13 = [
    "CS\n(Gold)", "CS\n(Evidence)", "EA\n(Gold)", "EA\n(Evidence)",
    "CT\n(Gold)", "CT\n(Evidence)",
    "Q1", "Q2", "Q3", "Q3a", "Q3b", "Q3c", "Q3d",
]

METRIC_LABELS_6 = ["CS\n(Gold)", "CS\n(Evidence)", "EA\n(Gold)", "EA\n(Evidence)",
                   "CT\n(Gold)", "CT\n(Evidence)"]
HUMAN_LABELS_7 = ["Q1", "Q2", "Q3", "Q3a", "Q3b", "Q3c", "Q3d"]

# Original column names in order matching the labels
COLS_13 = [
    "post_mean_consistency_gold", "post_mean_consistency_evidence",
    "post_max_entailment_gold", "post_max_entailment_evidence",
    "post_max_contradiction_gold", "post_max_contradiction_evidence",
    "human::q1_overall_quality", "human::q2_no_contradiction",
    "human::q3_coverage", "human::q3a_dominant_self_state",
    "human::q3b_element_recall", "human::q3c_unsupported_elements",
    "human::q3d_polarity_correctness",
]

METRIC_COLS_6 = ["post_mean_consistency_gold", "post_mean_consistency_evidence",
                 "post_max_entailment_gold", "post_max_entailment_evidence",
                 "post_max_contradiction_gold", "post_max_contradiction_evidence"]

HUMAN_COLS_7 = ["human::q1_overall_quality", "human::q2_no_contradiction",
                "human::q3_coverage", "human::q3a_dominant_self_state",
                "human::q3b_element_recall", "human::q3c_unsupported_elements",
                "human::q3d_polarity_correctness"]

# 8 discrete color boundaries
BOUNDS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.001]
cmap = plt.get_cmap("RdBu")
norm = BoundaryNorm(BOUNDS, cmap.N)


def plot_square_lower(rho_matrix, labels, out_path):
    """Square matrix, lower triangle + diagonal, all labels on both axes."""
    n = len(labels)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # hide strict upper

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(np.ma.array(rho_matrix, mask=mask), norm=norm, cmap=cmap, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(i + 1):  # lower triangle + diagonal
            v = rho_matrix[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, ticks=BOUNDS[:-1], shrink=0.7)
    cbar.set_ticklabels([f"{b:.2f}" for b in BOUNDS[:-1]], fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_rectangle(rho_matrix, pval_matrix, row_labels, col_labels, out_path):
    """Full rectangle (all cells visible), no masking. Asterisk on p<0.05."""
    n_rows, n_cols = rho_matrix.shape

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * n_cols), max(4, 0.7 * n_rows)))
    im = ax.imshow(rho_matrix, norm=norm, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(n_rows):
        for j in range(n_cols):
            v = rho_matrix[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.6 else "black"
                sig = "*" if (not np.isnan(pval_matrix[i, j]) and pval_matrix[i, j] < 0.05) else ""
                ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, ticks=BOUNDS[:-1], shrink=0.7)
    cbar.set_ticklabels([f"{b:.2f}" for b in BOUNDS[:-1]], fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── 1. System-level: 13×13 lower triangle (from spearman_rho_matrix.csv) ──
print("Building system-level 13x13 heatmap...")
sys_rho = pd.read_csv("outputs/final_analysis/human_vs_metrics/spearman_rho_matrix.csv", index_col=0)
# Reorder to match ALL_LABELS_13
sys_rho = sys_rho.reindex(index=COLS_13, columns=COLS_13)
plot_square_lower(sys_rho.values, ALL_LABELS_13,
                  "outputs/final_analysis/figures/heatmap_system_level.png")

# ── 2. Instance-level: 6 metrics × 7 human questions (full rectangle) ──
print("Building instance-level (human) heatmap...")
inst = pd.read_csv("outputs/final_analysis/human_vs_metrics/instance_correlations.csv")
pivot = inst.pivot_table(index="metric", columns="human", values="rho")
pivot = pivot.reindex(index=METRIC_COLS_6, columns=HUMAN_COLS_7)
pval = inst.pivot_table(index="metric", columns="human", values="p_value")
pval = pval.reindex(index=METRIC_COLS_6, columns=HUMAN_COLS_7)
plot_rectangle(pivot.values, pval.values, METRIC_LABELS_6, HUMAN_LABELS_7,
               "outputs/final_analysis/figures/heatmap_instance_human.png")

# ── 3. Instance-level: 6 metrics × 7 Mistral judge questions (full rectangle) ──
print("Building instance-level (Mistral judge) heatmap...")
inst_j = pd.read_csv("outputs/final_analysis/mistral_judge/instance_correlations.csv")
pivot_j = inst_j.pivot_table(index="metric", columns="human", values="rho")
pivot_j = pivot_j.reindex(index=METRIC_COLS_6, columns=HUMAN_COLS_7)
pval_j = inst_j.pivot_table(index="metric", columns="human", values="p_value")
pval_j = pval_j.reindex(index=METRIC_COLS_6, columns=HUMAN_COLS_7)
plot_rectangle(pivot_j.values, pval_j.values, METRIC_LABELS_6, HUMAN_LABELS_7,
               "outputs/final_analysis/figures/heatmap_instance_judge.png")

print("Done. All 3 figures in outputs/final_analysis/figures/")
