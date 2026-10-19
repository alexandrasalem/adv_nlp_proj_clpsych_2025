"""
Distill an official CLPsych eval CSV (results_dev_*.csv) into readable tables.

Handles both Task B and Task C results — the task is auto-detected from the
`task` column of the CSV (a CSV may contain one or both).

The raw CSV is long-format: one row per (post-or-timeline, metric).

This script is system-agnostic: it does not assume any particular set of
label sources, retrieval configs, or baseline names. The only structure it
relies on is the official columns:

    team_name, submission_id, task, metric, value

Grouping axes:
  - Task B: the per-submission means table always runs. If BOTH `team_name`
    and `submission_id` vary within Task B rows, an extra "two-axis" table is
    produced treating team_name x submission_id as the two swept factors.
    (For your original sweep these were label source x retrieval config, but
    any two-axis sweep is rendered the same way.)
  - Task C: the per-submission means table always runs. Submission ids listed
    in `--baselines` are tagged as baselines; everything else is a "system".

Usage:
    python analyze_official_results.py --results clpsych2025/evaluation/results/results_dev_<ts>.csv
    python analyze_official_results.py --results results_dev_*.csv --output_dir ./outputs --baselines constant,empty,gold
"""

import argparse
import os
import pandas as pd

# Primary (official "Main metric") per task. Override with --primary_b / --primary_c.
PRIMARY_B = "post_mean_consistency_gold"
PRIMARY_C = "timeline_mean_consistency_gold"

# Default degenerate baselines for Task C. Override with --baselines.
DEFAULT_BASELINES = "constant,empty,gold"


def analyze_task_b(df: pd.DataFrame, output_dir: str, primary: str):
    print("\n" + "#" * 70)
    print("# TASK B")
    print("#" * 70)

    # 1. Generic per-submission means: every submission x every metric.
    full = (df.groupby(["submission_id", "metric"])["value"]
              .mean()
              .unstack("metric"))
    print("\n=== Per-submission means (rows = submission_id) ===")
    print(full.round(4).to_string())
    full.round(6).to_csv(f"{output_dir}/official_summary_full.csv")

    # 2. Two-axis decomposition, only if BOTH axes actually vary. Otherwise the
    #    sweep is one-dimensional and this table would just duplicate (1).
    has_team = df["team_name"].nunique() > 1
    has_sub = df["submission_id"].nunique() > 1
    if has_team and has_sub:
        by_team = (df.groupby(["team_name", "metric"])["value"]
                     .mean()
                     .unstack("metric"))
        print("\n=== Effect of team_name (averaged over submission_id) ===")
        print(by_team.round(4).to_string())

        by_sub = (df.groupby(["submission_id", "metric"])["value"]
                     .mean()
                     .unstack("metric"))
        print("\n=== Effect of submission_id (averaged over team_name) ===")
        print(by_sub.round(4).to_string())
    else:
        print("\n(two-axis decomposition skipped: "
              f"team_name varies={has_team}, submission_id varies={has_sub})")

    # 3. Headline: primary metric, team_name x submission_id (falls back to a
    #    single-axis table when one of them does not vary).
    if primary not in df["metric"].unique():
        print(f"\n(primary metric '{primary}' not present in this file; skipping headline)")
    else:
        primary_df = df[df["metric"] == primary]
        if has_team and has_sub:
            headline = (primary_df.groupby(["team_name", "submission_id"])["value"]
                                  .mean()
                                  .unstack("submission_id"))
            print(f"\n=== HEADLINE: {primary} (rows = team_name, cols = submission_id) ===")
        else:
            headline = (primary_df.groupby("submission_id")["value"]
                                  .mean()
                                  .to_frame("value"))
            print(f"\n=== HEADLINE: {primary} (rows = submission_id) ===")
        print(headline.round(4).to_string())
        headline.round(6).to_csv(f"{output_dir}/official_summary_headline.csv")

    print(f"\nWrote official_summary_full.csv"
          f"{' and official_summary_headline.csv' if primary in df['metric'].unique() else ''}"
          f" to {output_dir}/")


def analyze_task_c(df: pd.DataFrame, output_dir: str, primary: str, baselines: set):
    print("\n" + "#" * 70)
    print("# TASK C")
    print("#" * 70)

    # Unit of comparison is submission_id.
    table = (df.groupby(["submission_id", "metric"])["value"]
               .mean()
               .unstack("metric"))
    if primary in table.columns:
        table = table.sort_values(primary, ascending=False)

    # Mark degenerate baselines vs real systems.
    table.insert(0, "kind", [
        "baseline" if s in baselines else "system" for s in table.index
    ])

    print(f"\n=== Per-submission means"
          f"{f' (sorted by {primary}, desc)' if primary in table.columns else ''} ===")
    print(table.round(4).to_string())
    table.round(6).to_csv(f"{output_dir}/official_summary_c.csv")
    print(f"\nWrote official_summary_c.csv to {output_dir}/")

    # Headline: where does the best baseline land vs the real systems?
    if primary in table.columns:
        prim = table[primary]
        systems = prim[table["kind"] == "system"]
        if len(systems):
            present_baselines = [b for b in baselines if b in prim.index]
            for b in present_baselines:
                v = prim[b]
                n_below = int((systems < v).sum())
                print(f"Headline: baseline_{b} {primary} = {v:.4f} — "
                      f"beats {n_below}/{len(systems)} real Task C systems "
                      f"(systems range {systems.min():.4f}-{systems.max():.4f}).")
            if not present_baselines:
                print(f"(no configured baselines present in this file; "
                      f"systems range {systems.min():.4f}-{systems.max():.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results_dev_*.csv")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES,
                        help=f"Comma-separated Task C baseline submission_ids "
                             f"(default: {DEFAULT_BASELINES})")
    parser.add_argument("--primary_b", default=PRIMARY_B,
                        help=f"Primary Task B metric (default: {PRIMARY_B})")
    parser.add_argument("--primary_c", default=PRIMARY_C,
                        help=f"Primary Task C metric (default: {PRIMARY_C})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    baselines = {b.strip() for b in args.baselines.split(",") if b.strip()}

    df = pd.read_csv(args.results)
    tasks = set(df["task"].astype(str).unique())
    print(f"Loaded {len(df)} rows; tasks present: {sorted(tasks)}")

    if "B" in tasks:
        analyze_task_b(df[df["task"].astype(str) == "B"], args.output_dir, args.primary_b)
    if "C" in tasks:
        analyze_task_c(df[df["task"].astype(str) == "C"], args.output_dir,
                       args.primary_c, baselines)
    if not ({"B", "C"} & tasks):
        raise SystemExit(f"No Task B or C rows found; task column had {sorted(tasks)}.")

    print("\nReading guide: consistency / entailment — higher is better; "
          "contradiction — lower is better.")


if __name__ == "__main__":
    main()
