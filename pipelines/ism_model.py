"""ISM (CLPsych 2025) Task B system — reimplementation for the critique paper.

Reference:
    Vu Tran and Tomoko Matsui. "Team ISM at CLPsych 2025: Capturing Mental
    Health Dynamics from Social Media Timelines using A Pretrained Large
    Language Model with In-Context Learning." CLPsych 2025, pp. 287-291.

What this reproduces (paper §3, Fig. 1, Task B)
------------------------------------------------
* LLM: meta-llama/Meta-Llama-3-8B-Instruct (no fine-tuning), temperature 0.1
  (paper-reported).
* Template-based in-context learning with 5 past analyses ("Analysis i:"
  blocks: post + adaptive segments + maladaptive segments + well-being +
  assessment/summary) — the paper's Figure 1 template, which jointly serves
  Tasks A.1/A.2/B.
* Two-step sampling-and-selection: repeat the generation over a number of
  rounds with freshly drawn random examples, then select the most likely
  candidate via Gaussian kernel density estimation (Scott's rule) over the
  candidate embeddings (sentence-transformers/all-MiniLM-L6-v2).

Project-convention adaptations (logged in the run's settings file):
* GOLD evidence spans and GOLD well-being scores are filled into the target
  post's section of the template, so the model only generates the
  "Assessment:" (summary) — the evidence is fixed and only the summary varies
  across candidates. Consequently the KDE is fitted on the SUMMARY embeddings
  alone (embedding the fixed evidence would add nothing).
* 10 candidates per post (the paper specifies neither the number of rounds
  nor how the "sample size 225" maps onto them; our documented choice is 10
  rounds. Each round draws a random pool of 225 eligible training posts —
  the paper's sample size — and then 5 examples at random from that pool).
* Leave-one-timeline-out: no round may draw an example from the target
  post's own timeline.
* Every (post, round) pair has a deterministic seed (prompt + example draw +
  sampling), so resumed runs regenerate identical candidates.
* Decoding beyond temperature: the paper reports only temperature 0.1;
  top_p 1.0 and max_tokens 600 are our documented defaults.

Run on the cluster (vLLM, local weights, tensor parallelism):
    python pipelines/ism_model.py --data_dir <timeline json dir> \
        --model /path/to/Meta-Llama-3-8B-Instruct --backend vllm

Outputs (to --output_dir):
    checkpoint_ism_taskb.jsonl     resumable per-post results (selected summary
                                    + all candidates + KDE densities)
    ism_taskb_submission.json       official submission format (dev.json-like)
    ism_taskb_settings.md           generation-settings log
"""

import argparse
import datetime
import os
import random
import time
import zlib

from system_utils import (
    Post, load_timelines, span_texts, make_backend, Checkpoint,
    write_task_b_submission, write_settings_log, drop_until_fits,
)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEMPERATURE = 0.1        # paper-reported
DEFAULT_N_ROUNDS = 10           # our documented choice (paper: unspecified)
DEFAULT_POOL_SIZE = 225          # paper: "sample size to 225"
DEFAULT_N_EXAMPLES = 5           # paper: 5 past analyses

ISM_TEMPLATE = """You are a mental health expert and analyzing a patient's social media post to determine their well-being, their dominant self-state of either adaptive or maladaptive. The following is your past analysis.
{analyses}
Now analyze the following patient post.
{post}
Adaptive post segments:
{adaptive_segments}
Maladaptive post segments:
{maladaptive_segments}
Well-being: {wellbeing}
Assessment:
"""


# ── Prompt construction ──────────────────────────────────────────────────────

def _segments_block(post: Post, polarity: str) -> str:
    spans = span_texts(post, polarity)
    if not spans:
        return "None identified"
    return "\n".join(f"* {s}" for s in spans)


def format_analysis(post: Post, index: int) -> str:
    wb = str(post.wellbeing) if post.wellbeing is not None else "Unknown"
    return (
        f"Analysis {index}:\n"
        f"{post.text}\n"
        f"Adaptive post segments:\n{_segments_block(post, 'adaptive')}\n"
        f"Maladaptive post segments:\n{_segments_block(post, 'maladaptive')}\n"
        f"Well-being: {wb}\n"
        f"Assessment:\n{post.summary}"
    )


def build_messages(target: Post, examples: list[Post]) -> list[dict]:
    analyses = "\n".join(format_analysis(ex, i + 1)
                         for i, ex in enumerate(examples))
    wb = str(target.wellbeing) if target.wellbeing is not None else "Unknown"
    prompt = ISM_TEMPLATE.format(
        analyses=analyses if analyses else "(No past analyses available.)",
        post=target.text,
        adaptive_segments=_segments_block(target, "adaptive"),
        maladaptive_segments=_segments_block(target, "maladaptive"),
        wellbeing=wb,
    )
    return [{"role": "user", "content": prompt}]


def deterministic_seed(timeline_id: str, post_index: int, round_idx: int) -> int:
    """Stable per-(post, round) seed so resumed runs regenerate the same
    candidates as a full run would have."""
    key = f"{timeline_id}|{post_index}|{round_idx}"
    return zlib.crc32(key.encode("utf-8")) % (2**31 - 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ISM (CLPsych 2025) Task B system reimplementation.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory of shared-task timeline JSON files.")
    parser.add_argument("--output_dir", default="./outputs/ism")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "ollama", "mock"])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model path (local weights) or id. Default: {DEFAULT_MODEL} (paper).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="vLLM tensor parallelism (8B fits on 1 GPU; set 1 to "
                             "shard the run across 4 parallel single-GPU jobs instead).")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM context length (None = model default; Llama-3 = 8192).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Paper-reported: {DEFAULT_TEMPERATURE}.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--n_rounds", type=int, default=DEFAULT_N_ROUNDS,
                        help="Candidates per post (paper: unspecified; our choice: 10).")
    parser.add_argument("--pool_size", type=int, default=DEFAULT_POOL_SIZE,
                        help="Training posts sampled per round (paper: 'sample size 225').")
    parser.add_argument("--n_examples", type=int, default=DEFAULT_N_EXAMPLES,
                        help="Past analyses per prompt (paper: 5).")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING,
                        help="KDE embedding model (paper: all-MiniLM-L6-v2).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Posts per generation batch (each post costs n_rounds generations).")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    all_posts = [p for tl in timelines for p in tl.posts]
    # Example pool: annotated training posts (gold summary + gold evidence).
    pool_posts = [p for p in all_posts
                  if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])]
    targets = [p for p in all_posts if p.summary]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Example pool: {len(pool_posts)} posts | targets: {len(targets)}")

    backend = make_backend(
        args.backend, args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype, quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    from sentence_transformers import SentenceTransformer
    print(f"[KDE embedding] {args.embedding_model} ...")
    embedder = SentenceTransformer(args.embedding_model)

    from sklearn.neighbors import KernelDensity

    # ── Generation loop (batched, resumable) ────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, "checkpoint_ism_taskb.jsonl")
    checkpoint = Checkpoint(ckpt_path)
    subm_path = os.path.join(args.output_dir, "ism_taskb_submission.json")

    pending = [p for p in targets
               if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"Pending targets: {len(pending)} / {len(targets)} (checkpoint resumed)")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()

        # Build n_rounds prompts per post with per-(post, round) random
        # LOTO example draws and deterministic sampling seeds.
        prompts, overrides, metas = [], [], []
        for p in batch:
            loto_pool = [q for q in pool_posts if q.timeline_id != p.timeline_id]
            for r in range(args.n_rounds):
                rng = random.Random(deterministic_seed(p.timeline_id, p.post_index, r))
                round_pool = rng.sample(loto_pool,
                                         min(args.pool_size, len(loto_pool)))
                examples = rng.sample(round_pool,
                                      min(args.n_examples, len(round_pool)))
                msgs, k_used = drop_until_fits(
                    lambda k: build_messages(p, examples[:k]), backend,
                    len(examples), args.max_model_len, args.max_tokens,
                    min_examples=1)
                prompts.append(msgs)
                overrides.append({"seed": deterministic_seed(p.timeline_id, p.post_index, r)})
                metas.append((p, r, k_used))

        summaries = backend.generate(prompts, sampling_overrides=overrides)

        # Regroup candidates per post and select via Gaussian KDE (Scott).
        by_post = {}
        for (p, r, k_used), s in zip(metas, summaries):
            by_post.setdefault(id(p), {"post": p, "candidates": {}})["candidates"][r] = s

        for entry in by_post.values():
            p = entry["post"]
            candidates = [entry["candidates"][r] for r in range(args.n_rounds)]
            nonempty = [c if c else "" for c in candidates]
            vecs = embedder.encode(nonempty, normalize_embeddings=True,
                                   show_progress_bar=False)
            kde = KernelDensity(kernel="gaussian", bandwidth="scott")
            kde.fit(vecs)
            log_density = kde.score_samples(vecs)
            chosen = int(log_density.argmax())
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": candidates[chosen],
                "candidate_summaries": candidates,
                "kde_log_density": [round(float(v), 6) for v in log_density],
                "chosen_round": chosen,
                "gold_summary": p.summary,
            })
            print(f"  [{p.timeline_id}#{p.post_index}] round {chosen} wins: "
                  f"{candidates[chosen][:70]}...")

        write_task_b_submission(timelines, checkpoint, subm_path)
        done = min(start + args.batch_size, len(pending))
        print(f"Batch {done}/{len(pending)} done "
              f"({time.perf_counter() - t0:.1f}s) — checkpoint + submission saved")

    write_task_b_submission(timelines, checkpoint, subm_path)

    write_settings_log(
        os.path.join(args.output_dir, "ism_taskb_settings.md"),
        "ISM Task B — generation settings",
        settings=[
            ("generation model", args.model, "paper (Meta-Llama-3-8B-Instruct)"),
            ("backend", args.backend, "our setup (vLLM, local weights)"),
            ("tensor_parallel_size", args.tensor_parallel_size, "our setup"),
            ("temperature", args.temperature, "paper (0.1)"),
            ("top_p", args.top_p, "our default — paper silent"),
            ("max_tokens", args.max_tokens, "our default — paper silent"),
            ("n_examples", args.n_examples, "paper (5 past analyses)"),
            ("candidates per post (rounds)", args.n_rounds,
             "our choice — paper does not specify the number of rounds"),
            ("pool size per round", args.pool_size,
             "paper ('sample size 225'); per-round pool the 5 examples are drawn from"),
            ("example selection", "random per (post, round), LOTO",
             "paper (random sampling); LOTO is a project convention"),
            ("selection", "Gaussian KDE (Scott's rule) over all-MiniLM-L6-v2 "
                          "embeddings of the candidate summaries",
             "paper (KDE over candidates) + project convention (summary-only "
             "embedding: gold evidence is fixed across candidates)"),
            ("prompt", "paper Figure 1 template ('patent' typo corrected to "
                       "'patient'; target sections pre-filled with gold)",
             "paper + project convention"),
            ("per-(post, round) seeds", "deterministic (crc32 of timeline|post|round)",
             "our choice (reproducible resume)"),
            ("evidence input", "gold adaptive/maladaptive spans + gold well-being",
             "project convention (original generated all fields jointly)"),
        ],
        notes=[
            "The original system generated evidence, well-being, and summary "
            "jointly and selected among full output triplets; with gold "
            "evidence/well-being fixed, only the summary varies across "
            "candidates, so the KDE is fitted on summary embeddings alone.",
            "The paper leaves the number of sampling rounds unspecified; we "
            "generate 10 candidates per post.",
            "Leave-one-timeline-out enforced on every round's example draw.",
        ],
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\nISM Task B run complete at {ts}. Submission: {subm_path}")


if __name__ == "__main__":
    main()
