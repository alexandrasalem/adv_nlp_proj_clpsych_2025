"""Aquarius (CLPsych 2025) Task B + Task C system — reimplementation for the critique paper.

Reference:
    Wang, Na, Gao, Ma, Hua, Chen, Wang. "From Posts to Timelines: Modeling
    Mental Health Dynamics from Social Media Timelines with Hybrid LLMs."
    CLPsych 2025, pp. 249-255. (Shared-task team: Aquarius.)

Task B (paper §3.3, Eq. 6-7, Appendix A Fig. 3)
-----------------------------------------------
* One-shot ICL with Qwen2.5-32B (no fine-tuning).
* Example selection: BGE-Large-EN-v1.5 embeddings; the SINGLE most
  cosine-similar training post (argmax), p' = argmax_i cos(Emb(p), Emb(p'_i)).
* The example enters the prompt with its gold adaptive evidence, maladaptive
  evidence, well-being score and summary (Fig. 3 includes well-being even
  though Eq. 7 omits it; we follow the figure).
* Prompt: Fig. 3 "Post-level Summary" template, verbatim (PDF spacing
  artifacts cleaned).

Task C (paper §2.3 / §3.3 and Fig. 3 caption)
--------------------------------------------
* Same procedure at timeline level: "all posts associated with each user are
  concatenated" and embedded to retrieve the single most similar training
  timeline; "evidence from all posts is incorporated during the generation".
* The paper does NOT publish the timeline prompt — Fig. 3's caption says it
  "closely follows the structure of the one for post-level summary but
  integrates the information of multiple posts". Our reconstruction keeps the
  Fig. 3 field structure (post / adaptive evidence / maladaptive evidence /
  well-being score) per post, then "Summary:". Flagged in the settings log.

Project-convention adaptations (logged in the settings files):
* GOLD evidence spans and GOLD well-being scores (the original used its
  fine-tuned Qwen2.5-7B predictions; the "w/ Class." and "w/ Gen." variants
  both collapse onto gold here).
* Leave-one-timeline-out for both tasks.
* Model checkpoint: the paper says "Qwen2.5-32B" used via ICL without
  fine-tuning and does not say base vs. Instruct; we use the Instruct
  checkpoint with its chat template (our interpretation).
* Decoding: paper silent; temperature 0.1, top_p 1.0, max_tokens 600.
* BGE embeddings without the retrieval instruction prefix (symmetric
  post-to-post / timeline-to-timeline similarity).
* BGE truncates at 512 tokens, so a concatenated timeline is represented by
  the normalized mean of its post embeddings (our choice; documented).

Run on the cluster:
    python aquarius_model.py --data_dir <dir> --model /path/to/Qwen2.5-32B-Instruct --task bc

Outputs (to --output_dir):
    checkpoint_aquarius_task{b,c}.jsonl   resumable results
    aquarius_task{b,c}_submission.json    official submission format
    aquarius_task{b,c}_settings.md        generation-settings log
"""

import argparse
import datetime
import os
import time

import numpy as np

from system_utils import (
    Post, Timeline, load_timelines, span_texts, Checkpoint,
    write_task_b_submission, write_task_c_submission, write_settings_log,
    drop_until_fits, fit_timeline_prompt, truncate_words, task_c_targets,
    add_common_args, backend_from_args, common_settings,
    make_embedder, free_gpu_memory,
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

# Task C — OUR RECONSTRUCTION of the unpublished timeline prompt, following
# the Fig. 3 structure with the per-post fields repeated for every post.
AQUARIUS_TIMELINE_PROMPT = """Summarize the following timeline, considering the evidence of adaptive and maladaptive states and the well-being score of each post.
{example_block}Timeline:
{target_timeline}
Summary:
"""

AQUARIUS_TIMELINE_EXAMPLE = """Timeline:
{example_timeline}
Summary:
{example_summary}
"""


# ── Prompt construction ──────────────────────────────────────────────────────

def _evidence_lines(post: Post, polarity: str) -> str:
    spans = span_texts(post, polarity)
    return "\n".join(f'- "{s}"' for s in spans) if spans else "None identified"


def _wellbeing_str(post: Post) -> str:
    return str(post.wellbeing) if post.wellbeing is not None else "Unknown"


def build_messages_b(target: Post, example: Post | None) -> list[dict]:
    if example is not None:
        example_block = dict(
            example_post=example.text,
            example_adaptive=_evidence_lines(example, "adaptive"),
            example_maladaptive=_evidence_lines(example, "maladaptive"),
            example_wellbeing=_wellbeing_str(example),
            example_summary=example.summary,
        )
    else:
        example_block = dict(example_post="(No example available.)",
                             example_adaptive="", example_maladaptive="",
                             example_wellbeing="", example_summary="")
    prompt = AQUARIUS_PROMPT.format(
        **example_block,
        target_post=target.text,
        target_adaptive=_evidence_lines(target, "adaptive"),
        target_maladaptive=_evidence_lines(target, "maladaptive"),
        target_wellbeing=_wellbeing_str(target),
    )
    return [{"role": "user", "content": prompt}]


def format_timeline(tl: Timeline, cap: int | None) -> str:
    blocks = []
    for i, p in enumerate(tl.posts, 1):
        blocks.append(
            f"Post {i}:\n{truncate_words(p.text, cap)}\n"
            f"Adaptive Evidence:\n{_evidence_lines(p, 'adaptive')}\n"
            f"Maladaptive Evidence:\n{_evidence_lines(p, 'maladaptive')}\n"
            f"Well-being Score:\n{_wellbeing_str(p)}")
    return "\n".join(blocks)


def build_messages_c(target: Timeline, example: Timeline | None,
                     cap: int | None) -> list[dict]:
    example_block = ""
    if example is not None:
        example_block = AQUARIUS_TIMELINE_EXAMPLE.format(
            example_timeline=format_timeline(example, cap),
            example_summary=example.summary)
    prompt = AQUARIUS_TIMELINE_PROMPT.format(
        example_block=example_block,
        target_timeline=format_timeline(target, cap))
    return [{"role": "user", "content": prompt}]


# ── Retrieval (runs BEFORE the LLM is loaded, so GPU memory is free) ─────────

def argmax_loto(query_vecs, query_tl_ids, db_vecs, db_tl_ids):
    """For each query, index of the most similar DB item from another timeline
    (or None) and its cosine similarity."""
    out = []
    for qv, qtl in zip(query_vecs, query_tl_ids):
        sims = db_vecs @ qv
        sims[db_tl_ids == qtl] = -np.inf                       # LOTO
        best = int(np.argmax(sims))
        out.append((best, float(sims[best])) if np.isfinite(sims[best]) else (None, 0.0))
    return out


def timeline_vectors(embedder, tls: list[Timeline]) -> np.ndarray:
    """Normalized mean of a timeline's post embeddings (BGE caps input at
    512 tokens, so the concatenated timeline cannot be embedded whole)."""
    vecs = []
    for tl in tls:
        pv = np.asarray(embedder.encode([p.text for p in tl.posts],
                                        normalize_embeddings=True,
                                        show_progress_bar=False))
        m = pv.mean(axis=0)
        vecs.append(m / (np.linalg.norm(m) or 1.0))
    return np.asarray(vecs)


def prepare(args, timelines):
    """Retrieve every in-context example up front; return per-task plans."""
    embedder = make_embedder(args.embedding_model, mock=args.backend == "mock")
    plans = {}
    if "b" in args.task:
        all_posts = [p for tl in timelines for p in tl.posts]
        db = [p for p in all_posts
              if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])]
        targets = [p for p in all_posts if p.summary][:args.limit] if args.limit \
            else [p for p in all_posts if p.summary]
        db_vecs = np.asarray(embedder.encode([p.text for p in db],
                                             normalize_embeddings=True,
                                             show_progress_bar=False))
        t_vecs = np.asarray(embedder.encode([p.text for p in targets],
                                            normalize_embeddings=True,
                                            show_progress_bar=False))
        hits = argmax_loto(t_vecs, [p.timeline_id for p in targets], db_vecs,
                           np.array([p.timeline_id for p in db]))
        plans["b"] = [(p, db[i] if i is not None else None, s)
                      for p, (i, s) in zip(targets, hits)]
        print(f"[Task B] retrieval DB {len(db)} posts | targets {len(targets)}")
    if "c" in args.task:
        db = [tl for tl in timelines if tl.summary and tl.posts]
        targets = task_c_targets(timelines, args.limit)
        db_vecs = timeline_vectors(embedder, db)
        t_vecs = timeline_vectors(embedder, targets)
        hits = argmax_loto(t_vecs, [tl.timeline_id for tl in targets], db_vecs,
                           np.array([tl.timeline_id for tl in db]))
        plans["c"] = [(tl, db[i] if i is not None else None, s)
                      for tl, (i, s) in zip(targets, hits)]
        print(f"[Task C] retrieval DB {len(db)} timelines | targets {len(targets)}")
    del embedder
    free_gpu_memory()
    return plans


# ── Generation ───────────────────────────────────────────────────────────────

def run_task_b(args, timelines, plan, backend):
    checkpoint = Checkpoint(os.path.join(args.output_dir, "checkpoint_aquarius_taskb.jsonl"))
    subm_path = os.path.join(args.output_dir, "aquarius_taskb_submission.json")
    pending = [x for x in plan if not checkpoint.has(x[0].timeline_id, x[0].post_id)]
    print(f"[Task B] pending {len(pending)} / {len(plan)}")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for p, example, sim in batch:
            msgs, k_used = drop_until_fits(
                lambda k, p=p, example=example: build_messages_b(p, example if k >= 1 else None),
                backend, 1 if example else 0, args.max_model_len,
                args.max_tokens, min_examples=0)
            prompts.append(msgs)
            metas.append((p, example if k_used else None, sim))
        outputs = backend.generate(prompts)
        dt = time.perf_counter() - t0
        for (p, example, sim), summary in zip(metas, outputs):
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": summary,
                "n_examples": 1 if example else 0,
                "retrieved_post_id": example.post_id if example else None,
                "retrieved_timeline": example.timeline_id if example else None,
                "retrieval_similarity": round(sim, 6) if example else None,
                "gold_summary": p.summary,
                "generation_sec": round(dt / len(batch), 2),
            })
        write_task_b_submission(timelines, checkpoint, subm_path)
        print(f"[Task B] {min(start + args.batch_size, len(pending))}/{len(pending)} "
              f"({dt:.1f}s)")
    write_task_b_submission(timelines, checkpoint, subm_path)
    n_zero = sum(1 for r in checkpoint.records.values() if r.get("n_examples") == 0)

    write_settings_log(
        os.path.join(args.output_dir, "aquarius_taskb_settings.md"),
        "Aquarius Task B — generation settings",
        settings=[
            ("generation model", args.model,
             "paper: Qwen2.5-32B via ICL, no fine-tuning; Instruct checkpoint is "
             "our interpretation (paper does not say base vs. Instruct)"),
            *common_settings(args),
            ("temperature", args.temperature, "our default — paper silent"),
            ("n_examples", 1, "paper (single most similar post, Eq. 6 argmax)"),
            ("embedding model", args.embedding_model, "paper (BGE-Large-EN-v1.5)"),
            ("embedding instruction prefix", "none",
             "our choice (symmetric post-to-post similarity)"),
            ("prompt", "paper Appendix A, Figure 3 (verbatim; spacing artifacts cleaned)",
             "paper"),
            ("well-being in prompt", "gold well-being (example + target)",
             "paper Fig. 3 template (Eq. 7 omits it)"),
            ("evidence input", "gold adaptive/maladaptive spans",
             "project convention (original: team's fine-tuned Qwen2.5-7B predictions)"),
            ("leave-one-timeline-out", True, "project convention"),
            ("posts generated without the example (context guard)", n_zero,
             "reported for transparency"),
        ],
        notes=[
            "Original conditioned on its own predicted evidence and well-being; "
            "with gold values the paper's two submission variants are equivalent.",
            "Retrieval is computed before the LLM is loaded and the embedder is "
            "freed, so it does not compete with vLLM for GPU memory.",
        ],
    )


def run_task_c(args, timelines, plan, backend):
    checkpoint = Checkpoint(os.path.join(args.output_dir, "checkpoint_aquarius_taskc.jsonl"))
    subm_path = os.path.join(args.output_dir, "aquarius_taskc_submission.json")
    pending = [x for x in plan if not checkpoint.has(x[0].timeline_id, "__timeline__")]
    print(f"[Task C] pending {len(pending)} / {len(plan)}")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for tl, example, sim in batch:
            msgs, k_used, cap = fit_timeline_prompt(
                lambda k, cap, tl=tl, example=example:
                    build_messages_c(tl, example if k >= 1 else None, cap),
                backend, 1 if example else 0, args.max_model_len,
                args.max_tokens, min_examples=0)
            prompts.append(msgs)
            metas.append((tl, example if k_used else None, sim, cap))
        outputs = backend.generate(prompts)
        dt = time.perf_counter() - t0
        for (tl, example, sim, cap), summary in zip(metas, outputs):
            checkpoint.append(tl.timeline_id, "__timeline__", {
                "summary": summary,
                "n_examples": 1 if example else 0,
                "post_word_cap": cap,
                "retrieved_timeline": example.timeline_id if example else None,
                "retrieval_similarity": round(sim, 6) if example else None,
                "gold_summary": tl.summary,
                "generation_sec": round(dt / len(batch), 2),
            })
        write_task_c_submission(timelines, checkpoint, subm_path)
        print(f"[Task C] {min(start + args.batch_size, len(pending))}/{len(pending)} "
              f"({dt:.1f}s)")
    write_task_c_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())

    write_settings_log(
        os.path.join(args.output_dir, "aquarius_taskc_settings.md"),
        "Aquarius Task C — generation settings",
        settings=[
            ("generation model", args.model,
             "paper: Qwen2.5-32B via ICL; Instruct checkpoint is our interpretation"),
            *common_settings(args),
            ("temperature", args.temperature, "our default — paper silent"),
            ("n_examples", 1, "paper (single most similar training timeline)"),
            ("timeline embedding",
             f"{args.embedding_model}, normalized mean of post embeddings",
             "paper: BGE over concatenated posts; mean-pooling over posts is our "
             "choice because BGE truncates at 512 tokens"),
            ("prompt", "reconstruction: Fig. 3 field structure repeated per post",
             "our reconstruction — paper does not publish the timeline prompt "
             "(Fig. 3 caption: 'closely follows the structure' of the post prompt)"),
            ("inputs", "every post's text + gold evidence + gold well-being",
             "paper ('evidence from all posts is incorporated'); well-being "
             "follows the Fig. 3 structure"),
            ("leave-one-timeline-out", True, "project convention"),
            ("timelines without the example (context guard)",
             sum(1 for r in recs if r.get("n_examples") == 0), "reported"),
            ("timelines with truncated posts (context guard)",
             sum(1 for r in recs if r.get("post_word_cap")), "reported"),
        ],
        notes=[
            "The timeline-level prompt is not published; this reconstruction is "
            "the main deviation for Aquarius Task C and should be stated as such.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Aquarius (CLPsych 2025) Task B/C reimplementation.")
    add_common_args(parser, DEFAULT_MODEL, "./outputs/aquarius")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING,
                        help="Retrieval embedder (paper: BGE-Large-EN-v1.5).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    plans = prepare(args, timelines)
    backend = backend_from_args(args)
    if "b" in plans:
        run_task_b(args, timelines, plans["b"], backend)
    if "c" in plans:
        run_task_c(args, timelines, plans["c"], backend)
    print(f"\nAquarius run complete at {datetime.datetime.now():%Y%m%d-%H%M%S}.")


if __name__ == "__main__":
    main()
