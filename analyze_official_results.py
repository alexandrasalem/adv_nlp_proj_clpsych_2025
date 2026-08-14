"""
Distill an official CLPsych eval CSV (results_dev_*.csv) into readable tables.

Handles both Task B and Task C results — the task is auto-detected from the
`task` column of the CSV (a CSV may contain one or both).

The raw CSV is long-format: one row per (post-or-timeline, metric).

Task B unit of comparison — the *condition*:
    team_name      = label source     (predicted / gold / random)
    submission_id  = retrieval config (hybrid-a0.25/0.50/0.75, structural,
                                       semantic, zeroshot)

Task C unit of comparison — the *submission*:
    submission_id  = the timeline-summary variant (one row each)
    (team_name is uniform, e.g. "c", so it is not a grouping axis)

Usage:
    python analyze_official_results.py --results clpsych2025/evaluation/results/results_dev_<ts>.csv
"""

import argparse
import pandas as pd

# Primary (official "Main metric") per task.
PRIMARY_B = "post_mean_consistency_gold"
PRIMARY_C = "timeline_mean_consistency_gold"

# Task B sweep structure.
LABEL_SOURCES = ["gold", "predicted", "random"]
ONESHOT_CONFIGS = ["hybrid-a0.25", "hybrid-a0.50", "hybrid-a0.75", "structural", "semantic"]

# Task C degenerate baselines (everything else is a real system).
TASK_C_BASELINES = {"constant", "empty", "gold"}


def analyze_task_b(df: pd.DataFrame, output_dir: str):
    print("\n" + "#" * 70)
    print("# TASK B")
    print("#" * 70)

    # 1. Full table: every condition x every metric (mean over all posts).
    full = (df.groupby(["team_name", "submission_id", "metric"])["value"]
              .mean()
              .unstack("metric"))
    print("\n=== Per-condition means (rows = label_source x retrieval) ===")
    print(full.round(4).to_string())
    full.round(6).to_csv(f"{output_dir}/official_summary_full.csv")

    # 2. Effect of label source (averaged over retrieval configs, zeroshot dropped).
    oneshot = df[df["submission_id"] != "zeroshot"]
    by_label = (oneshot.groupby(["team_name", "metric"])["value"]
                       .mean()
                       .unstack("metric"))
    print("\n=== Effect of LABEL SOURCE (one-shot only, averaged over 5 retrieval configs) ===")
    print(by_label.round(4).to_string())

    # 3. Effect of retrieval config (averaged over label sources).
    by_retrieval = (oneshot.groupby(["submission_id", "metric"])["value"]
                           .mean()
                           .unstack("metric"))
    print("\n=== Effect of RETRIEVAL CONFIG (averaged over label sources) ===")
    print(by_retrieval.round(4).to_string())

    # 4. Headline: primary metric only, label source x retrieval config.
    primary = df[df["metric"] == PRIMARY_B]
    headline = (primary.groupby(["team_name", "submission_id"])["value"]
                       .mean()
                       .unstack("submission_id"))
    print(f"\n=== HEADLINE: {PRIMARY_B} (rows = label source, cols = retrieval) ===")
    print(headline.round(4).to_string())
    headline.round(6).to_csv(f"{output_dir}/official_summary_headline.csv")
    print(f"\nWrote official_summary_full.csv and official_summary_headline.csv to {output_dir}/")


def analyze_task_c(df: pd.DataFrame, output_dir: str):
    print("\n" + "#" * 70)
    print("# TASK C")
    print("#" * 70)

    # Unit of comparison is submission_id (team_name is uniform for Task C).
    table = (df.groupby(["submission_id", "metric"])["value"]
               .mean()
               .unstack("metric"))
    if PRIMARY_C in table.columns:
        table = table.sort_values(PRIMARY_C, ascending=False)

    # Mark degenerate baselines vs real systems.
    table.insert(0, "kind", [
        "baseline" if s in TASK_C_BASELINES else "system" for s in table.index
    ])

    print(f"\n=== Per-submission means (sorted by {PRIMARY_C}, desc) ===")
    print(table.round(4).to_string())
    table.round(6).to_csv(f"{output_dir}/official_summary_c.csv")
    print(f"\nWrote official_summary_c.csv to {output_dir}/")

    # Headline: where does the `constant` baseline land vs the real systems?
    if PRIMARY_C in table.columns:
        prim = table[PRIMARY_C]
        systems = prim[table["kind"] == "system"]
        if "constant" in prim.index and len(systems):
            c = prim["constant"]
            n_below = int((systems < c).sum())
            print(f"\nHeadline: baseline_constant {PRIMARY_C} = {c:.4f} — "
                  f"beats {n_below}/{len(systems)} real Task C systems "
                  f"(systems range {systems.min():.4f}-{systems.max():.4f}).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results_dev_*.csv")
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    tasks = set(df["task"].astype(str).unique())
    print(f"Loaded {len(df)} rows; tasks present: {sorted(tasks)}")

    if "B" in tasks:
        analyze_task_b(df[df["task"].astype(str) == "B"], args.output_dir)
    if "C" in tasks:
        analyze_task_c(df[df["task"].astype(str) == "C"], args.output_dir)
    if not ({"B", "C"} & tasks):
        raise SystemExit(f"No Task B or C rows found; task column had {sorted(tasks)}.")

    print("\nReading guide: consistency / entailment — higher is better; "
          "contradiction — lower is better.")


if __name__ == "__main__":
    main()
