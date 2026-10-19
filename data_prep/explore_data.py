import os
import json
import random
import datetime
import argparse
from collections import Counter
from pipelines.pipeline import (
    load_timelines, extract_posts, load_model, generate,
    ABCD_KEYS, ABCD_DEFINITIONS, STATE_TYPES,
)


def _build_prompt(abcd_name: str, polarity: str, abcd_description: str, span_list: list[str]) -> str:
    """Pass 1: ask the model to distill candidate sub-category labels."""
    span_lines = "\n".join(f'{i+1}. "{text}"' for i, text in enumerate(span_list))
    prompt = f"""You are a clinical psychology expert distilling sub-categories from annotated evidence spans.

Category: {abcd_name} ({polarity})
Definition: {abcd_description}

The following text spans were all annotated as belonging to this category. Read them and produce a list of 5-10 distinct sub-category labels that best characterize the sub-types you observe. Each label should be 2-5 words.

Spans (one per line):
{span_lines}

Return ONLY a comma-separated list of sub-category labels — no preamble, no numbering, no explanation.
"""
    return prompt


def _build_assignment_prompt(
    abcd_name: str, polarity: str, labels: list[str], span_text: str
) -> str:
    """Pass 2: ask the model to assign a single span to one of the candidate labels."""
    label_lines = "\n".join(f"- {label}" for label in labels)
    return f"""You are assigning a text span to a sub-category.

Category: {abcd_name} ({polarity})
Candidate sub-category labels:
{label_lines}

Text span: "{span_text}"

Return ONLY the single label from the list above that best fits this span. If none fit, return the single word: NONE.
"""


def _parse_labels(raw: str) -> list[str]:
    """Parse the first-pass output into a clean list of labels."""
    parts = [p.strip().strip('"').strip("'").rstrip(".") for p in raw.split(",")]
    # Deduplicate while preserving order; case-insensitive.
    seen = set()
    out = []
    for p in parts:
        if not p or p.lower() == "none":
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _match_label(raw: str, labels: list[str]) -> str | None:
    """
    Match the model's free-form pass-2 response back to one of the candidate
    labels. Tries exact match first, then case-insensitive substring either way.
    Returns None if no match (or model said NONE).
    """
    cleaned = raw.strip().strip('"').strip("'").rstrip(".").strip()
    if not cleaned or cleaned.upper() == "NONE":
        return None

    cleaned_lc = cleaned.lower()
    # Exact (case-insensitive) match.
    for label in labels:
        if cleaned_lc == label.lower():
            return label
    # Containment either direction (model may add/drop a word).
    for label in labels:
        if label.lower() in cleaned_lc or cleaned_lc in label.lower():
            return label
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Data Exploration for CLPsych 2025 task"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with timeline JSON files")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local .gguf model file (or downloads from HF if not set)")
    parser.add_argument("--n_ctx", type=int, default=16384,
                        help="Context window size for the model")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="GPU layers to offload (-1 = all)")
    parser.add_argument("--max_spans_per_bucket", type=int, default=0,
                        help="max number of spans sampled from each bucket, 0 is no limit")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="base output directory")
    parser.add_argument("--min_bucket_size", type=int, default=3,
                        help="skip buckets smaller than this size")
    parser.add_argument("--no_two_pass", action="store_true",
                        help="Skip pass 2 (per-span label assignment / hit-count filtering). "
                             "By default the script runs the two-pass refinement.")
    parser.add_argument("--min_label_hits", type=int, default=2,
                        help="In pass 2, keep only labels that picked up at least this many spans. "
                             "If no label clears the threshold, falls back to the top 3 by hit count.")

    args = parser.parse_args()
    two_pass = not args.no_two_pass
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)

    output_dir = args.output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_out_path = os.path.join(output_dir, f"abcd_examples_distilled_{timestamp}.json")
    raw_spans_path = os.path.join(output_dir, f"spans_by_bucket_{timestamp}.json")
    debug_out_path = os.path.join(output_dir, f"bucket_debug_{timestamp}.json")

    timelines = load_timelines(args.data_dir)

    posts = extract_posts(timelines=timelines, context_window=0)

    buckets: dict[tuple[str, str], list[str]] = {}
    for p in posts:
        for s in p.evidence_spans:
            buckets.setdefault((s.gold_abcd_key, s.polarity), []).append(s.text)

    for bucket, spans in buckets.items():
        total_len = sum(len(text) for text in spans)
        print(f"Bucket {bucket}: {len(spans)} spans, {total_len} chars total")

    model = load_model(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    output: dict = {}
    spans_by_bucket: dict = {}
    debug: dict = {}

    for bucket, spans in buckets.items():
        abcd_key, polarity = bucket
        bucket_label = f"{abcd_key}:{polarity}"  # JSON-safe key

        # Skip undersized buckets (LLM-distilled labels won't be reliable).
        if len(spans) < args.min_bucket_size:
            print(f"Skipping {bucket_label}: {len(spans)} spans < min_bucket_size={args.min_bucket_size}")
            continue

        # Optional reproducible downsampling for very large buckets.
        if args.max_spans_per_bucket and len(spans) > args.max_spans_per_bucket:
            sampled = rng.sample(spans, args.max_spans_per_bucket)
            print(f"Sampling {bucket_label}: {len(spans)} -> {len(sampled)} spans "
                  f"(seed={args.seed})")
        else:
            sampled = spans

        # Persist the raw spans actually sent to the LLM (post-sampling)
        # so the distilled labels are reproducible from this file alone.
        spans_by_bucket[bucket_label] = sampled

        defn = ABCD_DEFINITIONS[abcd_key]

        # ── Pass 1: distill candidate sub-category labels ──
        prompt = _build_prompt(defn["name"], polarity, defn["description"], sampled)
        first_pass_raw = generate(model, prompt, max_tokens=200, temperature=0.1)
        candidate_labels = _parse_labels(first_pass_raw)
        print(f"\n[{bucket_label}] pass 1: {len(candidate_labels)} candidate labels")
        print(f"  -> {candidate_labels}")

        if not two_pass or not candidate_labels:
            # No refinement — use the first-pass output directly.
            output.setdefault(abcd_key, {})[polarity] = ", ".join(candidate_labels) if candidate_labels else first_pass_raw
            debug[bucket_label] = {
                "first_pass_raw": first_pass_raw,
                "candidate_labels": candidate_labels,
                "two_pass": False,
            }
            continue

        # ── Pass 2: assign each span to one of the candidate labels ──
        counts: Counter = Counter()
        assignments = []
        for span in sampled:
            ass_prompt = _build_assignment_prompt(
                defn["name"], polarity, candidate_labels, span
            )
            raw = generate(model, ass_prompt, max_tokens=30, temperature=0.1)
            matched = _match_label(raw, candidate_labels)
            assignments.append({"span": span, "raw_response": raw, "assigned": matched})
            if matched is not None:
                counts[matched] += 1

        # Filter labels by hit count; if nothing clears the bar, keep top 3.
        kept = [lbl for lbl in candidate_labels if counts[lbl] >= args.min_label_hits]
        fallback_used = False
        if not kept:
            kept = [lbl for lbl, _ in counts.most_common(3)]
            fallback_used = True

        # Order kept labels by hit count (descending) — most representative first.
        kept.sort(key=lambda lbl: counts[lbl], reverse=True)

        result_str = ", ".join(kept) if kept else first_pass_raw
        print(f"[{bucket_label}] pass 2: kept {len(kept)}/{len(candidate_labels)} labels "
              f"(min_label_hits={args.min_label_hits}{', fallback to top-3' if fallback_used else ''})")
        for lbl in kept:
            print(f"  - {lbl} ({counts[lbl]} spans)")

        output.setdefault(abcd_key, {})[polarity] = result_str
        debug[bucket_label] = {
            "first_pass_raw": first_pass_raw,
            "candidate_labels": candidate_labels,
            "two_pass": True,
            "counts": dict(counts),
            "kept": kept,
            "fallback_used": fallback_used,
            "assignments": assignments,
        }

    with open(data_out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(raw_spans_path, "w", encoding="utf-8") as r:
        json.dump(spans_by_bucket, r, indent=2, ensure_ascii=False)
    with open(debug_out_path, "w", encoding="utf-8") as d:
        json.dump(debug, d, indent=2, ensure_ascii=False)
    print(f"\nWrote distilled labels to {data_out_path}")
    print(f"Wrote raw spans to {raw_spans_path}")
    print(f"Wrote pass-2 debug info to {debug_out_path}")

if __name__ == "__main__":
    main()