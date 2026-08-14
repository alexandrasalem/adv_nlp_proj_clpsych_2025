"""
Generate degenerate-baseline files in pipeline.py's *task_b_results* format
(NOT the official-submission format — see make_degenerate_baselines.py for that).

Each output file is a drop-in task_b_results_*.json: same keys, same shape as a
real pipeline run, with `zero_shot_summary` and `one_shot_summary` overwritten
by a degenerate value. Intended to be fed to downstream tooling (e.g. a
teammate's Task C process) that consumes the pipeline's own output format.

Baselines:
    baseline-constant     — every post's summary = one fixed generic sentence
    baseline-goldsummary  — every post's summary = that post's gold summary
    baseline-empty        — every post's summary = ""

Usage:
    python make_degenerate_task_b_results.py --source outputs/task_b_results_<...>.json
"""

import os
import json
import argparse
import datetime

CONSTANT_SUMMARY = (
    "The person is experiencing emotional distress and is finding it "
    "difficult to cope with their situation."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="An existing task_b_results_*.json file, used as the "
                             "post inventory + source of gold_summary / evidence_texts.")
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.source, "r", encoding="utf-8") as f:
        source_entries = json.load(f)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # name -> function(post_entry) -> degenerate summary string
    baselines = {
        "baseline-constant":    lambda e: CONSTANT_SUMMARY,
        "baseline-goldsummary": lambda e: e.get("gold_summary", ""),
        "baseline-empty":       lambda e: "",
    }

    for name, summary_fn in baselines.items():
        out_entries = []
        for entry in source_entries:
            new = dict(entry)  # preserve every original key/value
            summary = summary_fn(entry)
            # A degenerate baseline has a single summary — set both modes to it.
            new["zero_shot_summary"] = summary
            new["one_shot_summary"] = summary
            out_entries.append(new)

        out_path = os.path.join(
            args.output_dir, f"task_b_results_{name}_{timestamp}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_entries, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}  ({len(out_entries)} posts)")


if __name__ == "__main__":
    main()
