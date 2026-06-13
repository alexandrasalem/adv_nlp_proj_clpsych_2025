"""
Paired Wilcoxon signed-rank test: does ABCD label source affect the official
Task B metric (post_mean_consistency_gold)?

For each post, the official eval produced a consistency_gold value under each
label source (gold / predicted / random) and each retrieval config. This test
pairs posts BY post_id and asks whether label source shifts the score.

Two views:
  PRIMARY   — per post, average consistency_gold over the 5 one-shot retrieval
              configs, then pair label sources across posts. One test per pair.
  PER-CONFIG— the same pairwise tests within each retrieval config separately,
              to show the result is not an artifact of averaging.

Requires the results CSV produced by the patched run.py (must contain a
`post_id` column).

Usage:
    python wilcoxon_label_test.py --results clpsych2025/evaluation/results/results_dev_<ts>.csv
"""

import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRIC = "post_mean_consistency_gold"
LABEL_SOURCES = ["gold", "predicted", "random"]
ONESHOT_CONFIGS = ["hybrid-a0.25", "hybrid-a0.50", "hybrid-a0.75", "structural", "semantic"]


def paired_test(pivot: pd.DataFrame, a: str, b: str):
    """Wilcoxon signed-rank on two paired columns of `pivot` (label sources a, b)."""
    if a not in pivot.columns or b not in pivot.columns:
        print(f"  {a} vs {b}: missing column — skipped")
        return

    x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
    diff = x - y
    n = len(diff)
    n_a_gt_b = int((diff > 0).sum())
    n_b_gt_a = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())

    if n - n_tie == 0:
        print(f"  {a:>9} vs {b:<9}: all {n} pairs identical — no test")
        return

    # wilcoxon needs at least one non-zero difference; zero-diff pairs dropped.
    stat, p = wilcoxon(x, y)  # default: drops zeros, normal approx for large n
    sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "n.s."
    print(f"  {a:>9} vs {b:<9}: n={n:<4} "
          f"medianΔ={np.median(diff):+.4f}  meanΔ={np.mean(diff):+.4f}  "
          f"W={stat:<10.1f} p={p:.4f}  {sig}")
    print(f"             {'':9}   ({a}>{b}: {n_a_gt_b},  {b}>{a}: {n_b_gt_a},  ties: {n_tie})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Patched run.py results CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if "post_id" not in df.columns:
        raise SystemExit(
            "No 'post_id' column in the CSV. Re-run the patched run.py "
            "(the version that tags Task B rows with post_id)."
        )

    df = df[df["metric"] == METRIC].copy()
    # Guard: a post must have a real post_id (Task B rows do).
    df = df[df["post_id"].astype(str) != ""]

    print(f"Metric: {METRIC}")
    print(f"Rows after filtering: {len(df)}\n")

    # ── PRIMARY: per-post mean over the 5 one-shot retrieval configs ──
    oneshot = df[df["submission_id"].isin(ONESHOT_CONFIGS)]
    pivot = (oneshot.groupby(["timeline_id", "post_id", "team_name"])["value"]
                    .mean()
                    .unstack("team_name"))
    before = len(pivot)
    pivot = pivot.dropna(subset=[c for c in LABEL_SOURCES if c in pivot.columns])
    dropped = before - len(pivot)

    print("=" * 70)
    print("PRIMARY TEST — per-post mean over the 5 one-shot retrieval configs")
    print(f"  paired posts: {len(pivot)}" + (f"  ({dropped} dropped: not in all label sources)" if dropped else ""))
    print("=" * 70)
    for a, b in itertools.combinations(LABEL_SOURCES, 2):
        paired_test(pivot, a, b)

    # ── PER-CONFIG: same pairwise tests within each retrieval config ──
    print("\n" + "=" * 70)
    print("PER-CONFIG TESTS — within each retrieval config separately")
    print("=" * 70)
    for cfg in ONESHOT_CONFIGS:
        sub = df[df["submission_id"] == cfg]
        p = (sub.groupby(["timeline_id", "post_id", "team_name"])["value"]
                .mean()
                .unstack("team_name"))
        p = p.dropna(subset=[c for c in LABEL_SOURCES if c in p.columns])
        print(f"\n[{cfg}]  paired posts: {len(p)}")
        for a, b in itertools.combinations(LABEL_SOURCES, 2):
            paired_test(p, a, b)

    print("\nInterpretation: 'n.s.' for gold-vs-random / gold-vs-predicted means "
          "the official Task B metric does not significantly distinguish correct, "
          "predicted, and random ABCD labels.")


if __name__ == "__main__":
    main()
