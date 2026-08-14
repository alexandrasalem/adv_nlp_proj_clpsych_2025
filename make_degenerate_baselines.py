"""
Generate degenerate-baseline submissions for the official CLPsych Task B eval.

Purpose (critique instrument): run trivial non-systems through the *same*
official scorer as the real models. If a content-free constant string scores
near the real systems, the official Task B metric has almost no floor — direct
evidence that it does not measure summary quality with usable resolution.

Baselines produced:
    baseline_goldsummary.json  — predicted summary = the gold summary (ceiling;
                                 should score ~1.0)
    baseline_constant.json     — every post gets the SAME generic sentence
                                 (fluent-but-content-free floor)
    baseline_empty.json        — predicted summary = "" (true floor; run.py
                                 scores empty predictions as 0.0)

These need no model inference. They are built from an existing task_b_results
file (for the post inventory + gold summaries) and post_data.csv (for the
timeline/post_index -> post_id mapping the official format keys on).

Usage:
    python make_degenerate_baselines.py ^
        --task_b_results outputs/task_b_results_gold_hybrid-a0.25_<ts>.json ^
        --post_data post_data.csv ^
        --output_dir clpsych2025/evaluation/submissions_dev

Then re-run the official scorer; the 3 baselines get scored alongside the
real submissions in submissions_dev/.
"""

import os
import json
import argparse
import pandas as pd

# A fluent, generic, content-free clinical-sounding sentence. Deliberately says
# nothing post-specific — if this scores well, the metric rewards fluency, not
# faithfulness.
CONSTANT_SUMMARY = (
    "The person is experiencing emotional distress and is finding it "
    "difficult to cope with their situation."
)


def load_post_map(post_data_csv: str) -> dict:
    """timeline_id -> {post_index: post_id} from post_data.csv."""
    df = pd.read_csv(post_data_csv)
    post_map: dict = {}
    for _, row in df.iterrows():
        post_map.setdefault(row["timeline_id"], {})[row["post_index"]] = row["post_id"]
    return post_map


def build_submission(task_b_entries: list, post_map: dict, summary_fn) -> dict:
    """
    Build one official-format submission dict.

    summary_fn(entry) -> str decides each post's predicted summary, where
    `entry` is the task_b_results record for that post (or None if the post
    has no record — those stay blank, matching the official converter).
    """
    by_key = {(e["timeline_id"], e["post_index"]): e for e in task_b_entries}

    out: dict = {}
    for timeline_id, idx_map in post_map.items():
        out[timeline_id] = {"timeline_level": {"summary": ""}, "post_level": {}}
        for post_index, post_id in idx_map.items():
            entry = by_key.get((timeline_id, post_index))
            template = {
                "adaptive_evidence": [],
                "maladaptive_evidence": [],
                "summary": "",
                "wellbeing_score": None,
            }
            if entry is not None:
                template["summary"] = summary_fn(entry)
            out[timeline_id]["post_level"][post_id] = template
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_b_results", required=True,
                        help="Any one task_b_results_*.json file (post inventory "
                             "+ source of gold summaries).")
    parser.add_argument("--post_data", default="post_data.csv",
                        help="post_data.csv with timeline_id, post_index, post_id.")
    parser.add_argument("--output_dir", default="clpsych2025/evaluation/submissions_dev",
                        help="Where to write the baseline submission JSONs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    post_map = load_post_map(args.post_data)
    with open(args.task_b_results, "r", encoding="utf-8") as f:
        task_b_entries = json.load(f)

    # filename stem must be exactly two underscore-separated parts so the
    # official run.py parse_filename() reads (team_name, submission_id).
    baselines = {
        "baseline_goldsummary": lambda e: e.get("gold_summary", ""),
        "baseline_constant":    lambda e: CONSTANT_SUMMARY,
        "baseline_empty":       lambda e: "",
    }

    for name, summary_fn in baselines.items():
        submission = build_submission(task_b_entries, post_map, summary_fn)
        out_path = os.path.join(args.output_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submission, f)
        n_posts = sum(len(t["post_level"]) for t in submission.values())
        print(f"Wrote {out_path}  ({len(submission)} timelines, {n_posts} posts)")

    print("\nDone. Re-run the official scorer to evaluate these alongside the "
          "real submissions.")


if __name__ == "__main__":
    main()
