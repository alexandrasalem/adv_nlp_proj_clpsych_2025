"""
Paired Wilcoxon signed-rank test on per-post ABCD-fidelity F1: does ABCD label
source (gold / predicted / random) affect fidelity when retrieval mode is held
constant (structural)?

This is the counterpart to wilcoxon_label_test.py, but tests OUR metric
(ABCD-fidelity F1) instead of the official CS. The goal is to show the
monotonic gold > predicted > random increase in fidelity is real and driven by
the structured retrieval, with a matched-pairs effect size.

For each pair (e.g. gold vs random), pairs BY post_id:
  - drops ties (posts where both get identical F1 — common for short summaries
    that assert the same slots)
  - reports:
      W        Wilcoxon signed-rank statistic
      p        two-sided p-value (normal approximation, n > ~25)
      r        matched-pairs rank-biserial correlation = (W+ - W-) / (W+ + W-)
               where W+ = sum of positive ranks, W- = sum of negative ranks.
               r = +1 means every non-tie post favors A; -1 means every favors B.
               Equivalent to the proportion of ranks favoring A minus favoring B.
      wins_A   posts where A's F1 > B's F1
      wins_B   posts where A's F1 < B's F1
      ties     posts where F1 is identical (dropped from the test)
      n_eff    effective n after dropping ties (what the test actually runs on)

Usage:
    python wilcoxon_fidelity_test.py
        --per_summary outputs/abcd_fidelity_wilcoxon/abcd_fidelity_per_summary.csv
        --output_dir ./outputs/wilcoxon_fidelity

The per_summary CSV must contain columns: system, post_id, f1 (from abcd_fidelity.py).
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata


def signed_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-).

    For the non-zero differences d = x - y, rank |d|, assign sign.
    W+ = sum of ranks where d > 0, W- = sum of ranks where d < 0.
    Returns +1 if all non-tie pairs favor x, -1 if all favor y.
    Equivalent to 2*W+/total - 1 where total = sum of all |rank| of non-zero diffs.
    """
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    w_plus = np.sum(ranks[nonzero > 0])
    w_minus = np.sum(ranks[nonzero < 0])
    total = w_plus + w_minus
    return (w_plus - w_minus) / total if total > 0 else 0.0


def paired_test(pivot: pd.DataFrame, a: str, b: str) -> dict:
    """
    Wilcoxon signed-rank on two paired columns, with effect size.
    a, b are system names (column headers in pivot).
    """
    base = {"a": a, "b": b}
    if a not in pivot.columns or b not in pivot.columns:
        return {**base, "note": "missing column"}

    x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
    diff = x - y
    n_total = len(diff)
    wins_a = int((diff > 0).sum())
    wins_b = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    n_eff = n_total - ties

    row = {**base, "n_total": n_total, "wins_a": wins_a, "wins_b": wins_b,
           "ties": ties, "n_eff": n_eff,
           "median_diff": float(np.median(diff)),
           "mean_diff": float(np.mean(diff))}

    if n_eff == 0:
        print(f"  {a:>22} vs {b:<22}: all {n_total} pairs identical — no test")
        return {**row, "W": np.nan, "p_value": np.nan, "r": np.nan, "significant": False,
                "note": "all pairs identical"}

    stat, p = wilcoxon(x, y)
    r = signed_rank_biserial(x, y)
    sig = bool(p < 0.05)
    sig_str = "SIGNIFICANT" if sig else "n.s."

    print(f"  {a:>22} vs {b:<22}: "
          f"n_eff={n_eff}/{n_total} (ties dropped: {ties})  "
          f"wins {a}: {wins_a}  wins {b}: {wins_b}  "
          f"median_diff={np.median(diff):+.4f}  "
          f"W={stat:.1f}  p={p:.4g}  r={r:+.3f}  {sig_str}")
    return {**row, "W": float(stat), "p_value": float(p), "r": r,
            "significant": sig, "note": ""}


def main():
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon test on per-post ABCD-fidelity F1 (label source comparison)."
    )
    parser.add_argument("--per_summary", required=True,
                        help="abcd_fidelity_per_summary.csv with columns system, post_id, f1.")
    parser.add_argument("--output_dir", default="./outputs/wilcoxon_fidelity")
    parser.add_argument("--pairs", default=None,
                        help="Comma-separated pairs to test as a=1,b=2;a=3,b=4. "
                             "Default: all 3 pairs among gold/predicted/random structural.")
    args = parser.parse_args()

    df = pd.read_csv(args.per_summary)
    needed = {"system", "post_id", "f1"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"per_summary CSV missing columns {missing}; got {list(df.columns)}")

    # Pivot: rows = post_id, columns = system, values = f1
    pivot = df.pivot_table(index="post_id", columns="system", values="f1")
    systems = list(pivot.columns)
    print(f"Systems in per-summary CSV: {systems}")
    print(f"Paired posts: {len(pivot)}\n")

    # Default pairs: all combinations of the 3 structural conditions
    if args.pairs:
        pairs = []
        for spec in args.pairs.split(";"):
            a, b = spec.split(",")
            pairs.append((a.strip(), b.strip()))
    else:
        target = ["gold_structural", "predicted_structural", "random_structural"]
        available = [s for s in target if s in systems]
        pairs = list(itertools.combinations(available, 2))

    print("=" * 70)
    print("PAIRED WILCOXON SIGNED-RANK TEST — per-post ABCD-fidelity F1")
    print("=" * 70)

    results = []
    for a, b in pairs:
        results.append(paired_test(pivot, a, b))

    print("\nInterpretation:")
    print("  r = matched-pairs rank-biserial correlation (+1 = all favor A, -1 = all favor B)")
    print("  n_eff = pairs after dropping ties (identical F1 — common for short summaries)")
    print("  A monotonic gold > predicted > random with significant p and positive r")
    print("  shows the structured retrieval captures real ABCD signal, not noise.")

    cols = ["a", "b", "n_total", "n_eff", "wins_a", "wins_b", "ties",
            "median_diff", "mean_diff", "W", "p_value", "r", "significant", "note"]
    out_df = pd.DataFrame(results).reindex(columns=cols)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "wilcoxon_fidelity_label_source.csv")
    out_df.round(6).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
