"""
Pairwise Preference: Borda Count + Bootstrapping
================================================

Consumes the filled-in pairwise-preference sheets produced by
build_pref_sheets.py (plus the answer key) and produces:

  1. Per-rater system ranking (from pairwise wins).
  2. PW-Combined ranking via Borda count across raters.
  3. Bootstrap stability of the PW-Combined ranking (mean Spearman rho vs sd).

Optionally, if a facet-ratings file is supplied, it also bootstraps the
facet-based ranking (resample posts with replacement, re-rank per-system means).

Expected inputs
---------------
--sheets_dir : a directory PER RATER is the cleanest layout, but you can also
  point at one directory and pass --rater_label. Each sheet CSV must have the
  columns written by build_pref_sheets.py:
      "Gold_Summary (reference)",
      "System_A_Summary",
      "System_B_Summary",
      "Rater_Choice (A / B / Neither)"    <- filled in with A / B / Neither
  and the row order must match the answer key (it does, if unedited).

--key : ANSWER_KEY_do_not_share.csv, with columns:
      sheet, row_index, System_A_is, System_B_is

Because A/B is randomized per row, the choice letter must be mapped back
through the key to a real system before tallying.

Usage
-----
  # one rater:
  python borda_bootstrap.py --sheets_dir ./rater1_sheets --key ./ANSWER_KEY_do_not_share.csv --rater_label R1

  # multiple raters (each subfolder = one rater):
  python borda_bootstrap.py --multi_raters ./all_raters --key ./ANSWER_KEY_do_not_share.csv

  # add facet bootstrap:
  python borda_bootstrap.py --multi_raters ./all_raters --key ./key.csv --facets ./facets.csv
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CHOICE_COL = "Rater_Choice (A / B / Neither)"


# ── Load one rater's pairwise votes ──────────────────────────────────────────

def load_rater_votes(sheets_dir: str, key_df: pd.DataFrame) -> list[dict]:
    """
    Read every sheet_*.csv in a rater's directory, join each row's A/B/Neither
    choice back to real systems via the key, and return a list of decisions:
        {"winner": <system or None>, "loser": <system or None>,
         "a": <sysA>, "b": <sysB>, "choice": "A"/"B"/"Neither"}
    A "Neither" yields winner=loser=None (a null vote).
    """
    votes = []
    sheet_files = sorted(Path(sheets_dir).glob("sheet_*.csv"))
    if not sheet_files:
        raise FileNotFoundError(f"No sheet_*.csv files found in {sheets_dir}")

    for sheet_path in sheet_files:
        sheet_name = sheet_path.stem  # e.g. "sheet_3"
        df = pd.read_csv(sheet_path)

        key_rows = key_df[key_df["sheet"] == sheet_name].sort_values("row_index")
        if len(key_rows) != len(df):
            raise ValueError(
                f"{sheet_name}: sheet has {len(df)} rows but key has "
                f"{len(key_rows)}. Row order must match — was the sheet edited?"
            )

        for (row_idx, sheet_row), (_, key_row) in zip(
            df.iterrows(), key_rows.iterrows()
        ):
            a_sys = key_row["System_A_is"]
            b_sys = key_row["System_B_is"]
            raw = str(sheet_row[CHOICE_COL]).strip().lower()

            if raw in ("a", "system a"):
                winner, loser = a_sys, b_sys
                choice = "A"
            elif raw in ("b", "system b"):
                winner, loser = b_sys, a_sys
                choice = "B"
            elif raw in ("neither", "n", "tie", ""):
                winner, loser = None, None
                choice = "Neither"
            else:
                raise ValueError(
                    f"{sheet_name} row {row_idx}: unrecognized choice '{raw}'. "
                    f"Expected A, B, or Neither."
                )

            votes.append({
                "winner": winner, "loser": loser,
                "a": a_sys, "b": b_sys, "choice": choice,
            })

    return votes


# ── Per-rater ranking from pairwise wins ─────────────────────────────────────

def rank_from_votes(votes: list[dict], systems: list[str]) -> dict:
    """
    Tally pairwise wins into per-system points, then rank (competition ranking,
    ties share a rank). Returns {system: rank} where rank 1 = best.
    """
    points = {s: 0 for s in systems}
    for v in votes:
        if v["winner"] is not None:
            points[v["winner"]] += 1

    # Rank by points descending; ties share the same (lower) rank number.
    ordered = sorted(systems, key=lambda s: points[s], reverse=True)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        # find the run of systems tied on points
        while j + 1 < len(ordered) and points[ordered[j + 1]] == points[ordered[i]]:
            j += 1
        shared_rank = i + 1  # 1-based
        for k in range(i, j + 1):
            ranks[ordered[k]] = shared_rank
        i = j + 1
    return ranks, points


# ── Borda combination across raters ──────────────────────────────────────────

def borda_combine(per_rater_ranks: list[dict], systems: list[str]) -> dict:
    """
    Borda: each rater's ranking awards a system (n_systems - rank) points
    (rank-1 of 4 -> 3 points, ..., rank-4 -> 0). Sum across raters.
    Returns {system: borda_points}.
    """
    n = len(systems)
    borda = {s: 0 for s in systems}
    for ranks in per_rater_ranks:
        for s in systems:
            borda[s] += (n - ranks[s])
    return borda


def borda_to_ranking(borda_points: dict, systems: list[str]) -> dict:
    """Turn Borda point totals into a {system: rank} map (1 = best, ties shared)."""
    ordered = sorted(systems, key=lambda s: borda_points[s], reverse=True)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and borda_points[ordered[j + 1]] == borda_points[ordered[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = i + 1
        i = j + 1
    return ranks


# ── Bootstrap the PW-Combined ranking ────────────────────────────────────────

def bootstrap_pairwise(all_rater_votes: list[list[dict]], systems: list[str],
                       n_boot: int = 10000, seed: int = 42):
    """
    Resample each rater's pairwise decisions WITH REPLACEMENT, recompute the
    Borda ranking, and compare to the original via Spearman rho. Repeat n_boot
    times. Returns (mean_rho, sd_rho, original_ranking_vector).
    """
    rng = np.random.default_rng(seed)

    # Original ranking
    orig_per_rater = [rank_from_votes(v, systems)[0] for v in all_rater_votes]
    orig_borda = borda_combine(orig_per_rater, systems)
    orig_rank = borda_to_ranking(orig_borda, systems)
    orig_vec = np.array([orig_rank[s] for s in systems])

    rhos = []
    for _ in range(n_boot):
        boot_per_rater = []
        for votes in all_rater_votes:
            idx = rng.integers(0, len(votes), size=len(votes))  # resample w/ replacement
            resampled = [votes[i] for i in idx]
            boot_per_rater.append(rank_from_votes(resampled, systems)[0])
        boot_borda = borda_combine(boot_per_rater, systems)
        boot_rank = borda_to_ranking(boot_borda, systems)
        boot_vec = np.array([boot_rank[s] for s in systems])

        # Spearman is undefined if a vector is constant (all tied). Skip those.
        if np.all(boot_vec == boot_vec[0]) or np.all(orig_vec == orig_vec[0]):
            continue
        rho, _ = spearmanr(orig_vec, boot_vec)
        if not np.isnan(rho):
            rhos.append(rho)

    rhos = np.array(rhos)
    return float(rhos.mean()), float(rhos.std()), orig_rank, orig_borda


# ── Facet-ranking bootstrap (optional) ───────────────────────────────────────

def bootstrap_facets(facets_df: pd.DataFrame, systems: list[str],
                     score_col: str, system_col: str, post_col: str,
                     n_boot: int = 10000, seed: int = 42):
    """
    Resample POSTS with replacement, recompute each system's mean facet score,
    re-rank, and compare to the original ranking via Spearman rho.

    facets_df must be long-format: one row per (post, system) with a score.
    """
    rng = np.random.default_rng(seed)
    posts = facets_df[post_col].unique()

    def rank_from_means(df):
        means = df.groupby(system_col)[score_col].mean()
        means = means.reindex(systems)
        ordered = means.sort_values(ascending=False).index.tolist()
        ranks = {s: r + 1 for r, s in enumerate(ordered)}
        return ranks

    orig_rank = rank_from_means(facets_df)
    orig_vec = np.array([orig_rank[s] for s in systems])

    rhos = []
    for _ in range(n_boot):
        boot_posts = rng.choice(posts, size=len(posts), replace=True)
        boot_df = pd.concat(
            [facets_df[facets_df[post_col] == p] for p in boot_posts],
            ignore_index=True,
        )
        boot_rank = rank_from_means(boot_df)
        boot_vec = np.array([boot_rank[s] for s in systems])
        rho, _ = spearmanr(orig_vec, boot_vec)
        if not np.isnan(rho):
            rhos.append(rho)

    rhos = np.array(rhos)
    return float(rhos.mean()), float(rhos.std()), orig_rank


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Borda count + bootstrap over pairwise sheets.")
    parser.add_argument("--key", required=True, help="Path to ANSWER_KEY_do_not_share.csv")
    parser.add_argument("--sheets_dir", help="One rater's sheets directory")
    parser.add_argument("--rater_label", default="R1", help="Label for single-rater mode")
    parser.add_argument("--multi_raters", help="Parent dir; each subfolder = one rater's sheets")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./pairwise_results.json")
    # optional facet bootstrap
    parser.add_argument("--facets", help="Long-format facet CSV (post, system, score)")
    parser.add_argument("--facet_score_col", default="Q1_Quality")
    parser.add_argument("--facet_system_col", default="system")
    parser.add_argument("--facet_post_col", default="post_id")

    args = parser.parse_args()

    key_df = pd.read_csv(args.key)
    systems = sorted(set(key_df["System_A_is"]) | set(key_df["System_B_is"]))
    print(f"Systems: {systems}")

    # Gather per-rater vote lists
    rater_dirs = {}
    if args.multi_raters:
        for sub in sorted(Path(args.multi_raters).iterdir()):
            if sub.is_dir():
                rater_dirs[sub.name] = str(sub)
    elif args.sheets_dir:
        rater_dirs[args.rater_label] = args.sheets_dir
    else:
        raise SystemExit("Provide either --sheets_dir or --multi_raters")

    all_rater_votes = []
    per_rater_report = {}
    for label, d in rater_dirs.items():
        votes = load_rater_votes(d, key_df)
        all_rater_votes.append(votes)
        ranks, points = rank_from_votes(votes, systems)
        n_neither = sum(1 for v in votes if v["choice"] == "Neither")
        per_rater_report[label] = {
            "points": points, "ranks": ranks,
            "n_votes": len(votes), "n_neither": n_neither,
        }
        print(f"\nRater {label}: {len(votes)} comparisons, {n_neither} Neither")
        for s in sorted(systems, key=lambda x: ranks[x]):
            print(f"  rank {ranks[s]}  {s:12s}  ({points[s]} wins)")

    # Borda combine
    per_rater_ranks = [rank_from_votes(v, systems)[0] for v in all_rater_votes]
    borda_points = borda_combine(per_rater_ranks, systems)
    pw_combined = borda_to_ranking(borda_points, systems)

    print("\n" + "=" * 50)
    print("PW-COMBINED (Borda across raters)")
    print("=" * 50)
    for s in sorted(systems, key=lambda x: pw_combined[x]):
        print(f"  rank {pw_combined[s]}  {s:12s}  ({borda_points[s]} Borda pts)")

    # Bootstrap the pairwise ranking
    mean_rho, sd_rho, orig_rank, orig_borda = bootstrap_pairwise(
        all_rater_votes, systems, n_boot=args.n_boot, seed=args.seed
    )
    print("\n" + "=" * 50)
    print(f"BOOTSTRAP STABILITY (n={args.n_boot})")
    print("=" * 50)
    print(f"  Mean Spearman rho vs original: {mean_rho:.3f}")
    print(f"  SD:                            {sd_rho:.3f}")

    results = {
        "systems": systems,
        "per_rater": per_rater_report,
        "pw_combined_ranking": pw_combined,
        "pw_combined_borda_points": borda_points,
        "bootstrap_pairwise": {
            "n_boot": args.n_boot,
            "mean_spearman_rho": mean_rho,
            "sd_spearman_rho": sd_rho,
        },
    }

    # Optional facet bootstrap
    if args.facets:
        facets_df = pd.read_csv(args.facets)
        f_mean, f_sd, f_rank = bootstrap_facets(
            facets_df, systems,
            score_col=args.facet_score_col,
            system_col=args.facet_system_col,
            post_col=args.facet_post_col,
            n_boot=args.n_boot, seed=args.seed,
        )
        print("\n" + "=" * 50)
        print(f"FACET BOOTSTRAP ({args.facet_score_col}, n={args.n_boot})")
        print("=" * 50)
        for s in sorted(systems, key=lambda x: f_rank[x]):
            print(f"  rank {f_rank[s]}  {s}")
        print(f"  Mean Spearman rho vs original: {f_mean:.3f}")
        print(f"  SD:                            {f_sd:.3f}")
        results["bootstrap_facets"] = {
            "score_col": args.facet_score_col,
            "original_ranking": f_rank,
            "mean_spearman_rho": f_mean,
            "sd_spearman_rho": f_sd,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()