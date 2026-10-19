"""
Degenerate baseline: gold-structural system but with RANDOM in-context example
retrieval instead of structural similarity retrieval. The gold ABCD labels and
the prompt are unchanged — only the retrieved example is random (not the
structurally nearest post). This isolates whether structural retrieval
contributes to fidelity, or whether any in-context example suffices.

Uses the same Qwen2.5-14B-Instruct model (via ollama), the same updated prompt,
and gold ABCD labels — the only change from the gold_structural condition is
swapping retrieve_nearest() for random.choice().

Usage:
    python run_random_retrieval.py --output_dir ./outputs/random_retrieval
    python run_random_retrieval.py --labels predicted,random --output_dir ./outputs/random_retrieval
"""
import argparse
import json
import os
import random
import datetime

from pipelines.pipeline import (
    load_timelines, extract_posts, build_all_profiles,
    build_zero_shot_prompt, build_one_shot_prompt,
    generate, ABCD_KEYS,
)

MODEL = "qwen2.5-14b-instruct"
DATA_DIR = r"C:\Users\owner\Documents\PhD\AdvancedNLP\train-clpsych2025-v2"
SEED = 42
A3_PREDICTIONS = r"outputs\sweep_rerun\task_a3_predictions_20260824-000216.json"


def load_predicted_by_post(a3_path):
    """Map (timeline_id, post_index) -> ordered list of predicted ABCD keys
    from the August 2026 A.3 run, aligned with extract_posts span order."""
    with open(a3_path, encoding="utf-8") as f:
        spans = json.load(f)["spans"]
    by_post = {}
    for s in spans:
        by_post.setdefault((s["timeline_id"], s["post_index"]), []).append(s)
    return by_post


def main():
    parser = argparse.ArgumentParser(description="Random-retrieval condition per ABCD label source.")
    parser.add_argument("--output_dir", default="./outputs/random_retrieval")
    parser.add_argument("--labels", default="gold",
                        help="Comma-separated subset of: gold,predicted,random")
    parser.add_argument("--skip_zero_shot", action="store_true", default=True,
                        help="Zero-shot is retrieval-independent and already "
                             "generated per label source; skip it by default.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]

    from pipelines.pipeline import _OllamaLlama
    llm = _OllamaLlama.from_pretrained(n_ctx=16384)

    timelines = load_timelines(DATA_DIR)
    posts = extract_posts(timelines, context_window=2)
    annotated = [p for p in posts if p.evidence_spans and p.gold_summary]
    print(f"Annotated posts: {len(annotated)}")

    all_span_refs = [span for p in annotated for span in p.evidence_spans]
    predicted_by_post = (load_predicted_by_post(A3_PREDICTIONS)
                         if "predicted" in labels else {})

    for label in labels:
        if label == "gold":
            for span in all_span_refs:
                span.predicted_abcd_key = span.gold_abcd_key
        elif label == "predicted":
            for p in annotated:
                key = (p.timeline_id, p.post_index)
                a3_entries = predicted_by_post[key]
                gold_seq = [s.gold_abcd_key for s in p.evidence_spans]
                a3_gold_seq = [s["gold"] for s in a3_entries]
                if gold_seq != a3_gold_seq:
                    raise SystemExit(f"A.3 alignment mismatch for {key}: "
                                     f"{gold_seq} vs {a3_gold_seq}")
                for span, a3 in zip(p.evidence_spans, a3_entries):
                    span.predicted_abcd_key = a3["predicted"]
        elif label == "random":
            # Same draw as the pipeline sweep (Random(SEED).choice over the
            # spans in post-traversal order) so the labels match the
            # random_structural condition exactly.
            rng_labels = random.Random(SEED)
            for span in all_span_refs:
                span.predicted_abcd_key = rng_labels.choice(ABCD_KEYS)
        else:
            raise SystemExit(f"Unknown label source: {label}")

        build_all_profiles(annotated, use_predicted=True, alpha=0.5)
        print(f"\n=== Label source: {label} — random retrieval ===")

        # Fresh retrieval RNG per label source -> the retrieved examples are
        # IDENTICAL across label sources (and to the original gold run);
        # only the ABCD labels in the prompt differ.
        rng = random.Random(SEED)

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

            if args.skip_zero_shot:
                entry["zero_shot_summary"] = None
            else:
                zs_prompt = build_zero_shot_prompt(post, use_predicted=True)
                entry["zero_shot_summary"] = generate(
                    llm, zs_prompt, max_tokens=400, temperature=0.1)
                print(f"  Zero-shot: {entry['zero_shot_summary'][:100]}...")

            # One-shot with RANDOM retrieval (not structural)
            candidates = [p for p in annotated
                          if p.timeline_id != post.timeline_id and p.gold_summary]
            example = rng.choice(candidates)
            os_prompt = build_one_shot_prompt(post, example, use_predicted=True)
            os_summary = generate(llm, os_prompt, max_tokens=400, temperature=0.1)
            entry["one_shot_summary"] = os_summary
            entry["retrieved_timeline"] = example.timeline_id
            entry["retrieved_post_index"] = example.post_index
            entry["retrieval_similarity"] = None  # random — no similarity
            entry["retrieval_mode"] = "random"
            print(f"  One-shot (random example from tl={example.timeline_id}): "
                  f"{os_summary[:100]}...")

            results.append(entry)

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.output_dir,
                                f"task_b_results_{label}_random-retrieval_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(results)} summaries to {out_path}")


if __name__ == "__main__":
    main()
