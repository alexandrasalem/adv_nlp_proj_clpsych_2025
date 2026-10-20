"""Aquarius (CLPsych 2025) Task B system — reimplementation for the critique paper.

Reference:
    Wang, Na, Gao, Ma, Hua, Chen, Wang. "From Posts to Timelines: Modeling
    Mental Health Dynamics from Social Media Timelines with Hybrid LLMs."
    CLPsych 2025, pp. 249-255. (Shared-task team: Aquarius.)

What this reproduces (paper §3.3, Eq. 6-7, Appendix A Fig. 3, Task B)
--------------------------------------------------------------------
* One-shot in-context learning with Qwen2.5-32B (base model, no fine-tuning).
* In-context example selection: embed posts with BGE-Large-EN-v1.5 and pick
  the SINGLE most cosine-similar training post (argmax, not top-k):
      p' = argmax_i cos(Emb(p), Emb(p'_i))
* The example enters the prompt together with its gold adaptive evidence,
  gold maladaptive evidence, and gold well-being score (Appendix A prompt
  includes the well-being scores even though Eq. 7 omits them — we follow the
  appendix template).
* Prompt: the paper's Appendix A, Figure 3 "Post-level Summary" template,
  transcribed verbatim (PDF-extraction spacing artifacts cleaned).

Project-convention adaptations (logged in the run's settings file):
* GOLD evidence spans and GOLD well-being score feed both the example and
  the target blocks (the original used the team's own fine-tuned Qwen2.5-7B
  predicted evidence/scores — its two official variants "w/ Class." and
  "w/ Gen." both collapse onto gold evidence here).
* Leave-one-timeline-out: the retrieved example never comes from the target
  post's own timeline.
* Decoding: the paper reports no temperature/top-p/max-tokens; we use
  temperature 0.1, top_p 1.0, max_tokens 600 (documented defaults).
* BGE embeddings are computed WITHOUT the retrieval instruction prefix
  (symmetric post-to-post similarity; documented choice).

Run on the cluster (vLLM, local weights, tensor parallelism):
    python pipelines/aquarius_model.py --data_dir <timeline json dir> \
        --model /path/to/Qwen2.5-32B-Instruct --backend vllm

Outputs (to --output_dir):
    checkpoint_aquarius_taskb.jsonl   resumable per-post results
    aquarius_taskb_submission.json    official submission format (dev.json-like)
    aquarius_taskb_settings.md        generation-settings log
"""

import argparse
import datetime
import os
import time

from system_utils import (
    Post, Timeline, load_timelines, span_texts,
    make_backend, Checkpoint, write_task_b_submission, write_settings_log,
    drop_until_fits,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_EMBEDDING = "BAAI/bge-large-en-v1.5"

# Paper Appendix A, Figure 3 — "Post-level Summary" prompt (verbatim except
# for PDF-extraction spacing artifacts, e.g. "post , considering" -> "post, considering").
AQUARIUS_PROMPT = """Summarize the following post, considering the evidence of adaptive and maladaptive states and the well-being score.
Post:
{example_post}
Adaptive Evidence:
{example_adaptive}
Maladaptive Evidence:
{example_maladaptive}
Well-being Score:
{example_wellbeing}
Summary:
{example_summary}
Post:
{target_post}
Adaptive Evidence:
{target_adaptive}
Maladaptive Evidence:
{target_maladaptive}
Well-being Score:
{target_wellbeing}
Summary:
"""


# ── Prompt construction ──────────────────────────────────────────────────────

def _evidence_lines(post: Post, polarity: str) -> str:
    spans = span_texts(post, polarity)
    return "\n".join(f'- "{s}"' for s in spans) if spans else "None identified"


def _wellbeing_str(post: Post) -> str:
    return str(post.wellbeing) if post.wellbeing is not None else "Unknown"


def build_messages(target: Post, example: Post | None) -> list[dict]:
    if example is not None:
        example_block = dict(
            example_post=example.text,
            example_adaptive=_evidence_lines(example, "adaptive"),
            example_maladaptive=_evidence_lines(example, "maladaptive"),
            example_wellbeing=_wellbeing_str(example),
            example_summary=example.summary,
        )
    else:
        example_block = dict(
            example_post="(No example available.)",
            example_adaptive="",
            example_maladaptive="",
            example_wellbeing="",
            example_summary="",
        )
    prompt = AQUARIUS_PROMPT.format(
        **example_block,
        target_post=target.text,
        target_adaptive=_evidence_lines(target, "adaptive"),
        target_maladaptive=_evidence_lines(target, "maladaptive"),
        target_wellbeing=_wellbeing_str(target),
    )
    return [{"role": "user", "content": prompt}]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aquarius (CLPsych 2025) Task B system reimplementation.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory of shared-task timeline JSON files.")
    parser.add_argument("--output_dir", default="./outputs/aquarius")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "ollama", "mock"])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model path (local weights) or id. Default: {DEFAULT_MODEL} (paper).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Paper reports no decoding settings; 0.1 is our documented default.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING,
                        help="Post-embedding model (paper: BGE-Large-EN-v1.5).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    all_posts = [p for tl in timelines for p in tl.posts]
    db_posts = [p for p in all_posts
                if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])]
    targets = [p for p in all_posts if p.summary]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Retrieval DB: {len(db_posts)} posts | targets: {len(targets)}")

    # ── Embedding: BGE-Large-EN-v1.5, one-shot argmax retrieval, LOTO ──────
    from sentence_transformers import SentenceTransformer
    import numpy as np
    print(f"[Embedding] {args.embedding_model} ...")
    embedder = SentenceTransformer(args.embedding_model)
    db_vecs = embedder.encode([p.text for p in db_posts],
                              normalize_embeddings=True, show_progress_bar=False)
    db_vecs = np.asarray(db_vecs)
    db_timeline_ids = np.array([p.timeline_id for p in db_posts])

    target_vecs = np.asarray(embedder.encode([p.text for p in targets],
                                             normalize_embeddings=True,
                                             show_progress_bar=False))

    def retrieve_example(target: Post, target_vec) -> tuple[Post | None, float]:
        """Single most similar DB post, never from the target's own timeline."""
        sims = db_vecs @ target_vec
        sims[db_timeline_ids == target.timeline_id] = -np.inf   # LOTO
        best = int(np.argmax(sims))
        if not np.isfinite(sims[best]):
            return None, 0.0
        return db_posts[best], float(sims[best])

    backend = make_backend(
        args.backend, args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype, quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, seed=args.seed,
    )

    # ── Generation loop (batched, resumable) ────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, "checkpoint_aquarius_taskb.jsonl")
    checkpoint = Checkpoint(ckpt_path)
    subm_path = os.path.join(args.output_dir, "aquarius_taskb_submission.json")

    pending = [p for p in targets
               if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"Pending targets: {len(pending)} / {len(targets)} (checkpoint resumed)")

    vec_by_post = {id(p): tv for p, tv in zip(targets, target_vecs)}

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for p in batch:
            example, sim = retrieve_example(p, vec_by_post[id(p)])
            # Context guard: if even the 1-example prompt exceeds the window,
            # drop the example (zero-shot fallback) rather than crash.
            msgs, k_used = drop_until_fits(
                lambda k: build_messages(p, example if k >= 1 else None),
                backend, 1 if example else 0, args.max_model_len,
                args.max_tokens, min_examples=0)
            prompts.append(msgs)
            metas.append((p, example, sim))
        summaries = backend.generate(prompts)
        for (p, example, sim), summary in zip(metas, summaries):
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": summary,
                "retrieved_post_id": example.post_id if example else None,
                "retrieved_timeline": example.timeline_id if example else None,
                "retrieval_similarity": round(sim, 6) if example else None,
                "gold_summary": p.summary,
                "generation_sec": round(time.perf_counter() - t0, 1) / len(batch),
            })
            print(f"  [{p.timeline_id}#{p.post_index}] {summary[:80]}...")
        write_task_b_submission(timelines, checkpoint, subm_path)
        done = min(start + args.batch_size, len(pending))
        print(f"Batch {done}/{len(pending)} done "
              f"({time.perf_counter() - t0:.1f}s) — checkpoint + submission saved")

    write_task_b_submission(timelines, checkpoint, subm_path)

    write_settings_log(
        os.path.join(args.output_dir, "aquarius_taskb_settings.md"),
        "Aquarius Task B — generation settings",
        settings=[
            ("generation model", args.model, "paper (Qwen2.5-32B, ICL, no fine-tuning)"),
            ("backend", args.backend, "our setup (vLLM, local weights)"),
            ("tensor_parallel_size", args.tensor_parallel_size, "our setup"),
            ("temperature", args.temperature,
             "our default — paper silent on decoding settings"),
            ("top_p", args.top_p, "our default — paper silent"),
            ("max_tokens", args.max_tokens, "our default — paper silent"),
            ("n_examples", 1, "paper (single most similar post, Eq. 6 argmax)"),
            ("embedding model", args.embedding_model, "paper (BGE-Large-EN-v1.5)"),
            ("embedding instruction prefix", "none",
             "our choice (symmetric post-to-post similarity)"),
            ("prompt", "paper Appendix A, Figure 3 (verbatim; spacing artifacts cleaned)",
             "paper"),
            ("well-being in prompt", "gold well-being scores (example + target)",
             "paper appendix template (Eq. 7 omits it; we follow Fig. 3)"),
            ("evidence input", "gold adaptive/maladaptive spans",
             "project convention (original used team's fine-tuned Qwen2.5-7B "
             "predicted evidence — variants 'w/ Class.'/'w/ Gen.' collapse onto gold)"),
            ("leave-one-timeline-out", True, "project convention"),
        ],
        notes=[
            "Original conditioned on its own predicted evidence spans and "
            "well-being scores; this reimplementation feeds gold values, so the "
            "paper's two submission variants are equivalent here.",
            "One-shot: the single most cosine-similar training post is retrieved "
            "with BGE-Large-EN-v1.5; leave-one-timeline-out enforced.",
        ],
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\nAquarius Task B run complete at {ts}. Submission: {subm_path}")


if __name__ == "__main__":
    main()
