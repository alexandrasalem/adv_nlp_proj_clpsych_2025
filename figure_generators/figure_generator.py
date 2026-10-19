"""Figure generator for the CLPsych metric-critique paper.
Generates all figures at once; fine-tune one by one.

Figures:
  1. CS-fidelity inversion horizontal bar chart (single-column, 7 systems)
  2. System-level Spearman rho heatmap (lower triangle + diagonal)
  3. Instance-level heatmaps (metrics × human, metrics × judge)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import os as _os
_os.makedirs("outputs/figures", exist_ok=True)

# ── System-level data ──
data = {
    "Gold/St":     [0.871, 0.959, 0.633, 0.789, 0.717, 0.365, 0.582, 0.636, 0.514, 112],
    "Gold/Sem":   [0.867, 0.956, 0.669, 0.749, 0.752, 0.423, 0.589, 0.589, 0.507, 114],
    "Pred/St":     [0.859, 0.950, 0.599, 0.744, 0.742, 0.417, 0.584, 0.618, 0.499, 111],
    "Rand/St":     [0.872, 0.956, 0.636, 0.773, 0.712, 0.390, 0.609, 0.529, 0.469, 114],
    "ST-top2":               [0.894, 0.970, 0.700, 0.757, 0.764, 0.472, 0.615, 0.475, 0.460, 248],
    "ST-Baseline":     [0.896, 0.952, 0.765, 0.924, 0.829, 0.502, 0.640, 0.424, 0.418, 233],
    "ST-top1":                    [0.906, 0.970, 0.269, 0.704, 0.514, 0.135, 0.351, 0.524, 0.286, 70],
    "Gold/Hy":    [0.863, 0.951, 0.662, 0.761, 0.762, 0.433, 0.590, 0.588, 0.491, 112],
    "Pred/Sem":  [0.865, 0.956, 0.639, 0.749, 0.751, 0.396, 0.582, 0.554, 0.471, 114],
    "Pred/Hy":    [0.869, 0.952, 0.644, 0.741, 0.729, 0.417, 0.570, 0.572, 0.468, 112],
    "Constant":                [0.904, np.nan, 0.648, np.nan, 0.274, np.nan, 0.206, 0.447, 0.201, 16],
    "Gold/Rd": [0.868, 0.956, 0.626, 0.775, 0.749, 0.398, 0.578, 0.529, 0.469, 109],
    "Gold": [0.876, np.nan, np.nan, np.nan, np.nan, np.nan, 0.431, 0.551, 0.362, 73],
    "Empty": [0.0, np.nan, np.nan, np.nan, 1.0, np.nan, 0.0, 1.0, 0.0, 0.0],
}
cols = ["CS_gold", "CS_evidence", "entail_gold", "entail_evidence",
        "contra_gold", "contra_evidence", "fid_R", "fid_P", "fid_F1", "len"]
df = pd.DataFrame(data, index=cols).T


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: CS-fidelity inversion horizontal bar chart
# Single-column width, 7 systems top to bottom:
#   Constant, BLUE, Official Baseline, EAIonFlux, Ours (Rand-Structural),
#   Ours (Gold-Random), Ours (Gold-Structural)
# ════════════════════════════════════════════════════════════════════════════

fig1_systems = [
    "Empty",
    "Constant",
    "ST-top1",
    "ST-Baseline",
    "ST-top2",
    "Gold",
    "Rand/St",
    "Gold/Rd",
    "Gold/St",
]
plot_df = df.loc[fig1_systems]

f1_vals = plot_df["fid_F1"].values
cs_vals = plot_df["CS_gold"].values

# Colors: blue for fidelity, red for CS; constant gets lighter + hatched
colors_f1 = ["#2c7bb6"] * len(fig1_systems)
colors_cs = ["#d7191c"] * len(fig1_systems)
const_idx = 1  # Constant is first
colors_f1[const_idx] = "#6baed6"
colors_cs[const_idx] = "#fcae91"

# Wider figure for single-column paper
fig, ax1 = plt.subplots(figsize=(5, 5.5))
y = np.arange(len(plot_df))
bar_h = 0.32

# Fidelity bars (bottom half)
bars1 = ax1.barh(y - bar_h/2, f1_vals, bar_h, color=colors_f1,
                 edgecolor="white", linewidth=0.4)
bars1[const_idx].set_hatch("//")
ax1.set_xlabel("ABCD-Fidelity F1", color="#2c7bb6", fontsize=10, fontweight="bold")
ax1.tick_params(axis="x", colors="#2c7bb6", labelsize=8)
ax1.set_yticks(y)
ax1.set_yticklabels(plot_df.index, fontsize=9)
ax1.set_xlim(0, 0.65)

# CS bars (top half, shared x-axis via twiny)
ax2 = ax1.twiny()
bars2 = ax2.barh(y + bar_h/2, cs_vals, bar_h, color=colors_cs,
                 edgecolor="white", linewidth=0.4)
bars2[const_idx].set_hatch("//")
ax2.set_xlabel("Consistency Score (CS)", color="#d7191c", fontsize=10, fontweight="bold")
ax2.tick_params(axis="x", colors="#d7191c", labelsize=8)
ax2.set_xlim(0.84, 0.93)

# Legend (bottom right, no overlap with bars)
legend_elements = [
    Patch(facecolor="#2c7bb6", label="Fidelity F1"),
    Patch(facecolor="#d7191c", label="CS"),
    Patch(facecolor="#6baed6", hatch="//", label="Degenerate"),
]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=7.5,
           framealpha=0.9, handlelength=1.2, borderpad=0.4)

fig.tight_layout()
fig.savefig("outputs/figures/fig1_cs_fidelity_inversion.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote outputs/figures/fig1_cs_fidelity_inversion.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: System-level Spearman rho heatmap (lower triangle + diagonal)
# Full square matrix: official metrics + fidelity + length
# Two versions:
#   2a (main paper): CS (Gold), CT (Gold), EA (Evidence) + Fid R/P/F1 + Length (9 systems)
#   2b (appendix): all 6 official metrics + Fid R/P/F1 + Length (9 systems)
# 9 systems = our 6 sweep conditions + EAIonFlux + BLUE + Official Baseline
# ════════════════════════════════════════════════════════════════════════════

# Build combined table from the df already loaded (9 real systems, no Constant)
# Exclude Constant and Gold-Random-Retrieval (degenerate / extra)
heatmap_systems = [
    "Gold/St",
    "Gold/Sem",
    "Gold/Hy",
    "Pred/St",
    "Pred/Sem",
    "Pred/Hy",
    "Rand/St",
    "ST-top2",
    "ST-Baseline",
    "ST-top1",
]
combined_df = df.loc[heatmap_systems]

# Column labels (paper style)
ALL_COLS = [
    "CS (Gold)", "CS (Evidence)", "EA (Gold)", "EA (Evidence)",
    "CT (Gold)", "CT (Evidence)",
    "Fid. Recall", "Fid. Precision", "Fid. F1",
    "Length",
]

# Map from df columns to paper labels
COL_MAP = {
    "CS_gold": "CS (Gold)", "CS_evidence": "CS (Evidence)",
    "entail_gold": "EA (Gold)", "entail_evidence": "EA (Evidence)",
    "contra_gold": "CT (Gold)", "contra_evidence": "CT (Evidence)",
    "fid_R": "Fid. Recall", "fid_P": "Fid. Precision", "fid_F1": "Fid. F1",
    "len": "Length",
}
combined_df = combined_df.rename(columns=COL_MAP)

# Main paper version: 3 official metrics + 3 fidelity + length = 7 columns
MAIN_COLS = ["CS (Gold)", "CT (Gold)", "EA (Evidence)",
             "Fid. Recall", "Fid. Precision", "Fid. F1", "Length"]

# Appendix version: all 6 official + 3 fidelity + length = 10 columns
APPENDIX_COLS = ALL_COLS

# Discrete colormap (8 bands)
BOUNDS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.001]
cmap = plt.get_cmap("RdBu")
norm = BoundaryNorm(BOUNDS, cmap.N)


def compute_rho_matrix(df_subset, cols):
    """Compute full Spearman rho matrix for given columns."""
    n = len(cols)
    rho = np.full((n, n), np.nan)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                rho[i, j] = 1.0
            else:
                pair = df_subset[[a, b]].dropna()
                if len(pair) >= 3 and pair[a].nunique() > 1 and pair[b].nunique() > 1:
                    r, _ = spearmanr(pair[a], pair[b])
                    rho[i, j] = r
                    rho[j, i] = r
    return rho


def plot_heatmap(rho_mat, labels, out_path, figsize, font_size=12, precision=1):
    """Plot lower-triangle + diagonal heatmap."""
    nn = len(labels)
    mask = np.triu(np.ones((nn, nn), dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.array(rho_mat, mask=mask), norm=norm, cmap=cmap, aspect="equal")

    ax.set_xticks(range(nn))
    ax.set_xticklabels(labels, fontsize=font_size, ha="center", rotation=35)
    ax.set_yticks(range(nn))
    ax.set_yticklabels(labels, fontsize=font_size, rotation=35)

    for i in range(nn):
        for j in range(i + 1):  # lower triangle + diagonal
            v = rho_mat[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.5 else "black"
                if precision == 1:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=font_size - 2, color=color)
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=font_size - 2, color=color)

    cbar = fig.colorbar(im, ax=ax, ticks=BOUNDS[:-1], shrink=0.7)
    cbar.set_ticklabels([f"{b:.2f}" for b in BOUNDS[:-1]], fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# 2a: Main paper — 3 official + 3 fidelity + length (7x7)
# Main paper: no parentheses on official metric labels
MAIN_COLS = ["CS", "CT", "EA",
             "Fid. Recall", "Fid. Precision", "Fid. F1", "Length"]
main_rho = compute_rho_matrix(combined_df, ["CS (Gold)", "CT (Gold)", "EA (Evidence)",
             "Fid. Recall", "Fid. Precision", "Fid. F1", "Length"])
file_path = Path('outputs') / 'figures' / 'fig2_spearman_main.png'
plot_heatmap(main_rho, MAIN_COLS, file_path, (6, 5.5), precision=2)

# 2b: Appendix — all 6 official + 3 fidelity + length (10x10)
# Appendix: keep parentheses to distinguish Gold vs Evidence
APPENDIX_COLS = ALL_COLS
all_rho = compute_rho_matrix(combined_df, APPENDIX_COLS)
file_path = Path('outputs') / 'figures' / 'fig2_spearman_appendix.png'
plot_heatmap(all_rho, APPENDIX_COLS, file_path, (8, 7.5), precision=2)


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: System-level Spearman rho between official metrics and human questions
# 3 official metrics (CS, CT, EA) × 7 human questions (Q1–Q3d)
# Uses the 4 human-eval systems (EAIonFlux, BLUE, Official Baseline, Ours)
# Full square matrix on both axes: CS, CT, EA, Q1, Q2, Q3, Q3a, Q3b, Q3c, Q3d
# ════════════════════════════════════════════════════════════════════════════

# Build the combined table for the 4 human-eval systems
# (from system_scores.csv which has both metrics and human columns)
sys_scores = pd.read_csv("outputs/final_analysis/human_vs_metrics/system_scores.csv", index_col=0)

# The CSV has system IDs like "aeion-flux_task-b-k5", etc.
# We need CS (Gold), CT (Gold), EA (Evidence) + human Q1-Q3d
fig3_metric_map = {
    "post_mean_consistency_gold": "CS",
    "post_max_contradiction_gold": "CT",
    "post_max_entailment_evidence": "EA",
}
fig3_human_map = {
    "human::q1_overall_quality": "Q1",
    "human::q2_no_contradiction": "Q2",
    "human::q3_coverage": "Q3",
    "human::q3a_dominant_self_state": "Q3a",
    "human::q3b_element_recall": "Q3b",
    "human::q3c_unsupported_elements": "Q3c",
    "human::q3d_polarity_correctness": "Q3d",
}

# Rename and select
fig3_df = sys_scores.rename(columns={**fig3_metric_map, **fig3_human_map})
FIG3_COLS = ["CS", "CT", "EA", "Q1", "Q2", "Q3", "Q3a", "Q3b", "Q3c", "Q3d"]
fig3_df = fig3_df[[c for c in FIG3_COLS if c in fig3_df.columns]]

# Compute rho matrix
fig3_rho = compute_rho_matrix(fig3_df, FIG3_COLS)

# Plot with larger fonts (all short labels, so can be bigger)
fig3_labels = FIG3_COLS
file_path = Path('outputs') / 'figures' / 'fig3_spearman_metrics_human.png'
plot_heatmap(fig3_rho, fig3_labels, file_path, (7, 6.5), font_size=14)


# ════════════════════════════════════════════════════════════════════════════
# Figure 4: Instance-level Spearman rho (n=120) — metrics × human questions
# Figure 5: Instance-level Spearman rho (n=120) — metrics × LLM-judge questions
# Both: 3 official metrics (rows) × 7 questions (columns), full rectangle
# Asterisk on p < 0.05
# ════════════════════════════════════════════════════════════════════════════

def plot_instance_heatmap(rho_vals, p_vals, row_labels, col_labels, out_path, figsize, font_size=12):
    """Plot a full rectangle heatmap with asterisks on significant cells."""
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rho_vals, norm=norm, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=font_size, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=font_size)

    for i in range(n_rows):
        for j in range(n_cols):
            v = rho_vals[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.5 else "black"
                sig = "*" if (not np.isnan(p_vals[i, j]) and p_vals[i, j] < 0.05) else ""
                ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center",
                        fontsize=font_size - 2, color=color)

    cbar = fig.colorbar(im, ax=ax, ticks=BOUNDS[:-1], shrink=0.7)
    cbar.set_ticklabels([f"{b:.2f}" for b in BOUNDS[:-1]], fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# Instance-level metric column names in the CSVs
INST_METRIC_COLS = ["post_mean_consistency_gold", "post_max_contradiction_gold",
                    "post_max_entailment_evidence"]
INST_METRIC_LABELS = ["CS", "CT", "EA"]
INST_HUMAN_COLS = ["human::q1_overall_quality", "human::q2_no_contradiction",
                   "human::q3_coverage", "human::q3a_dominant_self_state",
                   "human::q3b_element_recall", "human::q3c_unsupported_elements",
                   "human::q3d_polarity_correctness"]
INST_HUMAN_LABELS = ["Q1", "Q2", "Q3", "Q3a", "Q3b", "Q3c", "Q3d"]


# Figure 4: Instance-level metrics × human questions (n=120)
inst_human = pd.read_csv("outputs/final_analysis/human_vs_metrics/instance_correlations.csv")
pivot_rho = inst_human.pivot_table(index="metric", columns="human", values="rho")
pivot_rho = pivot_rho.reindex(index=INST_METRIC_COLS, columns=INST_HUMAN_COLS)
pivot_p = inst_human.pivot_table(index="metric", columns="human", values="p_value")
pivot_p = pivot_p.reindex(index=INST_METRIC_COLS, columns=INST_HUMAN_COLS)

file_path = Path('outputs') / 'figures' / 'fig4_instance_human.png'
plot_instance_heatmap(pivot_rho.values, pivot_p.values,
                      INST_METRIC_LABELS, INST_HUMAN_LABELS,
                      file_path, (7, 3.5), font_size=13)


# Figure 5: Instance-level metrics × LLM-judge questions (n=120)
inst_judge = pd.read_csv("outputs/final_analysis/mistral_judge/instance_correlations.csv")
pivot_j_rho = inst_judge.pivot_table(index="metric", columns="human", values="rho")
pivot_j_rho = pivot_j_rho.reindex(index=INST_METRIC_COLS, columns=INST_HUMAN_COLS)
pivot_j_p = inst_judge.pivot_table(index="metric", columns="human", values="p_value")
pivot_j_p = pivot_j_p.reindex(index=INST_METRIC_COLS, columns=INST_HUMAN_COLS)

file_path = Path('outputs') / 'figures' / 'fig5_instance_judge.png'
plot_instance_heatmap(pivot_j_rho.values, pivot_j_p.values,
                      INST_METRIC_LABELS, INST_HUMAN_LABELS,
                      file_path, (7, 3.5), font_size=13)