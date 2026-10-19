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

Writes every pairwise test (both views) to a tidy CSV — one row per
(scope, config, label-source pair) with n, win counts, median/mean delta, the
Wilcoxon W and p, and a significance flag.

Usage:
    python wilcoxon_label_test.py --results clpsych2025/evaluation/results/results_dev_<ts>.csv
    # -> outputs/wilcoxon_<results_stem>.csv  (override dir with --output_dir or path with --output)
"""

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRIC = "post_mean_consistency_gold"
LABEL_SOURCES = ["gold", "predicted", "random"]
ONESHOT_CONFIGS = ["hybrid-a0.25", "hybrid-a0.50", "hybrid-a0.75", "structural", "semantic"]


def paired_test(pivot: pd.DataFrame, a: str, b: str, scope: str, config: str) -> dict:
    """
    Wilcoxon signed-rank on two paired columns of `pivot` (label sources a, b).
    Prints a human-readable line and returns a result row for the output CSV.
    """
    base = {"scope": scope, "config": config, "a": a, "b": b}
    if a not in pivot.columns or b not in pivot.columns:
        print(f"  {a} vs {b}: missing column — skipped")
        return {**base, "n": 0, "note": "missing column"}

    x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
    diff = x - y
    n = len(diff)
    n_a_gt_b = int((diff > 0).sum())
    n_b_gt_a = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())
    row = {**base, "n": n, "n_a_gt_b": n_a_gt_b, "n_b_gt_a": n_b_gt_a, "n_tie": n_tie,
           "median_diff": float(np.median(diff)), "mean_diff": float(np.mean(diff))}

    if n - n_tie == 0:
        print(f"  {a:>9} vs {b:<9}: all {n} pairs identical — no test")
        return {**row, "W": np.nan, "p_value": np.nan, "significant": False,
                "note": "all pairs identical"}

    # wilcoxon needs at least one non-zero difference; zero-diff pairs dropped.
    stat, p = wilcoxon(x, y)  # default: drops zeros, normal approx for large n
    sig = bool(p < 0.05)
    sig_str = "SIGNIFICANT (p<0.05)" if sig else "n.s."
    print(f"  {a:>9} vs {b:<9}: n={n:<4} "
          f"median_diff={np.median(diff):+.4f}  mean_diff={np.mean(diff):+.4f}  "
          f"W={stat:<10.1f} p={p:.4f}  {sig_str}")
    print(f"             {'':9}   ({a}>{b}: {n_a_gt_b},  {b}>{a}: {n_b_gt_a},  ties: {n_tie})")
    return {**row, "W": float(stat), "p_value": float(p), "significant": sig, "note": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Patched run.py results CSV")
    parser.add_argument("--output_dir", default="./outputs",
                        help="Directory for the output CSV (default ./outputs).")
    parser.add_argument("--output", default=None,
                        help="Full output path (overrides --output_dir / default name).")
    args = parser.parse_args()

    results_rows = []

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
        results_rows.append(paired_test(pivot, a, b, scope="primary", config="mean_of_5_oneshot"))

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
            results_rows.append(paired_test(p, a, b, scope="per_config", config=cfg))

    print("\nInterpretation: 'n.s.' for gold-vs-random / gold-vs-predicted means "
          "the official Task B metric does not significantly distinguish correct, "
          "predicted, and random ABCD labels.")

    # ── Write results ──
    cols = ["scope", "config", "a", "b", "n", "n_a_gt_b", "n_b_gt_a", "n_tie",
            "median_diff", "mean_diff", "W", "p_value", "significant", "note"]
    out_df = pd.DataFrame(results_rows).reindex(columns=cols)
    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"wilcoxon_{Path(args.results).stem}.csv")
    out_df.round(6).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
