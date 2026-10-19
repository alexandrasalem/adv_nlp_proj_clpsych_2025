"""
Run EAIonFlux Task C (timeline-level summaries) using their own prompt
(paper Appendix A.3, verbatim), retrieval system (timeline representation ->
embedding -> top-k similar), system message, and model (llama3.1:8b).

Cascades their existing Task B k=5 one-shot summaries into the Task C prompt as
the "Post-level Summaries (if available)" block (the paper's cascade design,
input_type="task_b_predicted").

Usage:
    python run_eaionflux_task_c.py --output_dir ./outputs/task_c_eaionflux
"""
import argparse
import json
import os

import pandas as pd

from pipelines.eaionflux_model import (
    load_timelines,
    make_generator,
    run_task_c,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_EMBEDDING_MODEL,
)

DATA_DIR = r"C:\Users\owner\Documents\PhD\AdvancedNLP\train-clpsych2025-v2"
TASK_B_OUTPUT = "outputs/eaion_flux/aeion-flux_task-b-k5.json"


def extract_post_summaries(path: str) -> dict:
    """Official-format Task B JSON -> {timeline_id: {post_id: summary}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for tl_id, tl in data.items():
        for post_id, entry in (tl.get("post_level") or {}).items():
            s = (entry or {}).get("summary", "") or ""
            if s.strip():
                out.setdefault(tl_id, {})[post_id] = s
    return out


def main():
    parser = argparse.ArgumentParser(description="EAIonFlux Task C run (paper prompt + retrieval).")
    parser.add_argument("--output_dir", default="./outputs/task_c_eaionflux")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timelines = load_timelines(DATA_DIR)
    generate = make_generator(args.model, None, args.temperature, args.max_tokens)
    print(f"Generation model (Ollama): {args.model} | Embedding: {args.embedding_model}")

    # Cascade: feed their Task B one-shot summaries as post-level input.
    post_summaries = extract_post_summaries(TASK_B_OUTPUT)
    n_sums = sum(len(v) for v in post_summaries.values())
    print(f"Cascaded Task B post summaries: {n_sums} posts across {len(post_summaries)} timelines")

    run_task_c(
        timelines, generate, args.embedding_model, args.k,
        args.output_dir,
        post_summaries=post_summaries,
        input_type="task_b_predicted",
        prompt_style="eaionflux",
        exclude_same_timeline=True,
    )


if __name__ == "__main__":
    main()
