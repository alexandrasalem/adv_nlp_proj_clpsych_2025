"""
Zero-shot-only Task B generation for the gold and predicted label sources
(new prompt), completing the new-prompt sweep grid.

The August 2026 new-prompt sweep runs for gold/predicted were generated with
zero-shot skipped (their task_b_results files have zero_shot_summary = null),
so no new-prompt zero-shot condition exists for those two label sources. The
random label source already has zero-shot summaries (generated in the same
run as random/structural — see
outputs/sweep_rerun_random/task_b_results_random_structural_20260825-151157.json).

This script regenerates ONLY the zero-shot summaries:
  * gold      — spans carry their gold ABCD keys
  * predicted — spans carry the A.3 predicted keys reused verbatim from the
                August sweep (outputs/sweep_rerun/task_a3_predictions_20260824-000216.json),
                so the 'predicted' condition matches the rest of the sweep.
Zero-shot ignores retrieval, so no retrieval is performed and no one-shot
summaries are generated (one_shot_summary = null).

Same Qwen2.5-14B-Instruct model via ollama, same updated prompt, same
generation settings as the sweep runs (max_tokens=400, temperature=0.1).

Usage:
    python run_zeroshot_label_sources.py --output_dir ./outputs/zeroshot_rerun
"""
import argparse
import json
import os
import datetime

from pipelines.pipeline import (
    load_timelines, extract_posts, build_zero_shot_prompt, generate,
)

MODEL_CTX = 16384
DATA_DIR = r"C:\Users\owner\Documents\PhD\AdvancedNLP\train-clpsych2025-v2"
A3_PREDICTIONS = r"outputs\sweep_rerun\task_a3_predictions_20260824-000216.json"


def load_predicted_labels(a3_path):
    """Map (timeline_id, post_index) -> ordered list of predicted ABCD keys,
    aligned with the evidence-span order produced by extract_posts
    (state_type-major, then ABCD_KEYS order)."""
    with open(a3_path, encoding="utf-8") as f:
        spans = json.load(f)["spans"]
    by_post = {}
    for s in spans:
        by_post.setdefault((s["timeline_id"], s["post_index"]), []).append(s)
    return by_post


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot-only Task B for gold + predicted (new prompt).")
    parser.add_argument("--output_dir", default="./outputs/zeroshot_rerun")
    parser.add_argument("--labels", default="gold,predicted",
                        help="Comma-separated subset of: gold,predicted")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]

    from pipelines.pipeline import _OllamaLlama
    llm = _OllamaLlama.from_pretrained(n_ctx=MODEL_CTX)

    timelines = load_timelines(DATA_DIR)
    posts = extract_posts(timelines, context_window=2)
    annotated = [p for p in posts if p.evidence_spans and p.gold_summary]
    print(f"Annotated posts: {len(annotated)}")

    predicted_by_post = load_predicted_labels(A3_PREDICTIONS) if "predicted" in labels else {}

    for label in labels:
        if label == "gold":
            for p in annotated:
                for span in p.evidence_spans:
                    span.predicted_abcd_key = span.gold_abcd_key
        elif label == "predicted":
            n_checked = 0
            for p in annotated:
                key = (p.timeline_id, p.post_index)
                a3_entries = predicted_by_post[key]
                gold_seq = [s.gold_abcd_key for s in p.evidence_spans]
                a3_gold_seq = [s["gold"] for s in a3_entries]
                if gold_seq != a3_gold_seq:
                    raise SystemExit(
                        f"A.3 alignment mismatch for {key}: "
                        f"spans {gold_seq} vs a3 file {a3_gold_seq}")
                for span, a3 in zip(p.evidence_spans, a3_entries):
                    span.predicted_abcd_key = a3["predicted"]
                n_checked += len(a3_entries)
            print(f"Aligned {n_checked} predicted labels from the August A.3 run")
        else:
            raise SystemExit(f"Unknown label source: {label}")

        print(f"\n=== Label source: {label} — zero-shot only ===")
        results = []
        for i, post in enumerate(annotated):
            print(f"\n--- Post {i+1}/{len(annotated)} "
                  f"(timeline={post.timeline_id}, index={post.post_index}) ---")
            entry = {
                "timeline_id": post.timeline_id,
                "post_index": post.post_index,
                "wellbeing": post.wellbeing,
                "gold_summary": post.gold_summary,
                "n_evidence_spans": len(post.evidence_spans),
                "evidence_texts": [s.text for s in post.evidence_spans],
            }
            zs_prompt = build_zero_shot_prompt(post, use_predicted=True)
            zs_summary = generate(llm, zs_prompt, max_tokens=400, temperature=0.1)
            entry["zero_shot_summary"] = zs_summary
            entry["one_shot_summary"] = None
            print(f"  Zero-shot: {zs_summary[:100]}...")
            results.append(entry)

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.output_dir,
                                f"task_b_results_{label}_zeroshot_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(results)} zero-shot summaries to {out_path}")


if __name__ == "__main__":
    main()
