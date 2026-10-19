"""
Generate Task C timeline-level summaries for EAIonFlux and the 6 new-prompt
sweep systems, using the same Qwen2.5-14B-Instruct model (via ollama) that
generated the Task B summaries.

Task C: for each timeline (30 total), concatenate all posts in chronological
order and generate a timeline-level summary describing the evolution of
self-states over time. The output is an official-format submission JSON with
only timeline_level.summary populated (post_level can be empty — Task C scores
only the timeline summary).

Usage:
    python generate_task_c.py --output_dir ./outputs/task_c
"""
import argparse
import json
import os
import time

import pandas as pd
import ollama

MODEL = "qwen2.5-14b-instruct"

PROMPT_TEMPLATE = """You are a clinical psychology expert analyzing social media posts for mental health dynamics.

Given the following series of Reddit posts from one user over time, generate a timeline-level summary. Begin by determining which self-state is dominant (adaptive/maladaptive) and describe it first. Focus on the interplay between adaptive and maladaptive self-states over time, emphasizing temporal dynamics such as flexibility, rigidity, improvement, and deterioration.

Describe how the dominance of self-states shifts over time, highlighting key emotional, cognitive, and behavioral changes that contribute to these transitions. Use the ABCD framework where relevant: Affect (A), Behavior (B), Cognition (C), Desire/Need (D).

For each self-state present, highlight the central organizing ABCD aspect that drives the state.

You must not make anything up. Keep the description concise (5-10 sentences). Only describe observations fully supported by the text.

Timeline:
{timeline_text}

Timeline Summary:"""


def load_timelines(post_data_csv: str) -> dict[str, str]:
    """Map timeline_id -> concatenated post texts in chronological order."""
    df = pd.read_csv(post_data_csv)
    timelines = {}
    for _, row in df.iterrows():
        tl_id = row["timeline_id"]
        post = str(row.get("post", "") or "")
        if tl_id not in timelines:
            timelines[tl_id] = []
        timelines[tl_id].append((row.get("post_index", 0), post))
    # Sort by post_index, join
    return {tl_id: "\n\n".join(p for _, p in sorted(posts))
            for tl_id, posts in timelines.items()}


def generate_timeline_summary(client, timeline_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(timeline_text=timeline_text)
    r = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"num_predict": 600, "temperature": 0.1, "top_p": 0.9},
    )
    return (r.message.content or "").strip()


def build_submission(timeline_summaries: dict[str, str]) -> dict:
    """Build official Task C submission JSON."""
    out = {}
    for tl_id, summary in timeline_summaries.items():
        out[tl_id] = {
            "timeline_level": {"summary": summary},
            "post_level": {},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate Task C timeline summaries via ollama.")
    parser.add_argument("--post_data", default="post_data.csv")
    parser.add_argument("--output_dir", default="./outputs/task_c")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.post_data)
    print(f"Loaded {len(timelines)} timelines from {args.post_data}")

    client = ollama.Client()
    print(f"Model: {MODEL}")

    # Generate one set of timeline summaries (the model + prompt are the same
    # for all our systems — Task C doesn't use ABCD labels or retrieval).
    # EAIonFlux uses the same model, so one set covers all 7 systems.
    t0 = time.time()
    summaries = {}
    for i, (tl_id, text) in enumerate(timelines.items(), 1):
        summary = generate_timeline_summary(client, text)
        summaries[tl_id] = summary
        print(f"  [{i}/{len(timelines)}] {tl_id}: {len(summary)} chars "
              f"({time.time() - t0:.0f}s elapsed)")

    # Write one submission JSON — all 7 systems share the same Task C summaries
    # since Task C doesn't depend on ABCD labels or retrieval config.
    submission = build_submission(summaries)
    out_path = os.path.join(args.output_dir, "task_c_all_systems.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)
    print(f"\nWrote {out_path} ({len(submission)} timelines)")
    print("This single submission covers all systems (same model, same prompt).")
    print("Copy to submissions_dev as different filenames for scoring.")


if __name__ == "__main__":
    main()
