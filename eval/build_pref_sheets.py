"""
Pairwise Preference Sheet Builder
=================================

Takes one CSV with columns [Gold, Ours, BLUE, Baseline, AEIonFlux] (one row
per post) and produces 6 anonymized pairwise-preference sheets, one per
system pair.

Each output sheet has four columns:
    Gold_Summary (reference) | System_A_Summary | System_B_Summary | Rater_Choice

- Gold leads each row as the reference to compare against.
- Systems are anonymized as A / B; the A/B assignment is randomized per row
  and differs per sheet so position and identity carry no signal.
- Rater_Choice is left blank for the reviewer (fill with A, B, or Neither).
- An answer key records every (sheet, row) -> which real system is A/B so you
  can de-anonymize when scoring. Do not share the key with reviewers.

Usage
-----
    python build_pref_sheets.py --input summaries.csv --output_dir ./pref_sheets
"""

import os
import csv
import json
import random
import argparse
import itertools
from datetime import datetime

import pandas as pd

GOLD_COL = "Gold"


def load_input(path: str):
    """Load the CSV; return (dataframe, list_of_system_columns)."""
    df = pd.read_csv(path)
    if GOLD_COL not in df.columns:
        raise ValueError(
            f"Input CSV must contain a '{GOLD_COL}' column. Found: {list(df.columns)}"
        )
    system_cols = [c for c in df.columns if c != GOLD_COL]
    if len(system_cols) < 2:
        raise ValueError(
            f"Need at least 2 system columns to form a pair. Found: {system_cols}"
        )
    print(f"Loaded {len(df)} rows. Systems: {system_cols}")
    return df, system_cols


def build_sheets(df, system_cols, output_dir, seed):
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    pairs = list(itertools.combinations(system_cols, 2))
    print(f"Generating {len(pairs)} pairwise sheets")

    key_rows = []

    for sheet_idx, (sys1, sys2) in enumerate(pairs, start=1):
        sheet_name = f"sheet_{sheet_idx}"
        out_path = os.path.join(output_dir, f"{sheet_name}.csv")

        out_rows = []
        for row_idx, (_, row) in enumerate(df.iterrows()):
            # Randomize per row which real system is A vs B
            if rng.random() < 0.5:
                a_sys, b_sys = sys1, sys2
            else:
                a_sys, b_sys = sys2, sys1

            out_rows.append({
                "Gold_Summary (reference)": row[GOLD_COL],
                "System_A_Summary": row[a_sys],
                "System_B_Summary": row[b_sys],
                "Rater_Choice (A / B / Neither)": "",
            })

            key_rows.append({
                "sheet": sheet_name,
                "row_index": row_idx,
                "System_A_is": a_sys,
                "System_B_is": b_sys,
            })

        pd.DataFrame(out_rows).to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"  {sheet_name}.csv  ({sys1} vs {sys2}, {len(out_rows)} rows)")

    # Answer key — keep away from reviewers
    key_path = os.path.join(output_dir, "ANSWER_KEY_do_not_share.csv")
    pd.DataFrame(key_rows).to_csv(key_path, index=False)

    manifest = {
        "generated": datetime.now().isoformat(),
        "seed": seed,
        "n_rows_per_sheet": len(df),
        "systems": system_cols,
        "pairs": [f"{a} vs {b}" for a, b in pairs],
        "note": "A/B labels randomized per row and per sheet. "
                "Use ANSWER_KEY_do_not_share.csv to de-anonymize.",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nKey file: {key_path}")
    print("Do NOT give reviewers the key file or manifest.")


def main():
    parser = argparse.ArgumentParser(
        description="Build anonymized pairwise-preference sheets from a "
                    "system-comparison CSV."
    )
    parser.add_argument("--input", required=True,
                        help="Path to the input CSV (Gold + 4 system columns).")
    parser.add_argument("--output_dir", default="./pref_sheets",
                        help="Directory to write the sheets and key file.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for A/B assignment.")
    args = parser.parse_args()

    df, system_cols = load_input(args.input)
    build_sheets(df, system_cols, args.output_dir, args.seed)


if __name__ == "__main__":
    main()