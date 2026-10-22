"""ISM (CLPsych 2025) Task B + Task C system — reimplementation for the critique paper.

Reference:
    Vu Tran and Tomoko Matsui. "Team ISM at CLPsych 2025: Capturing Mental
    Health Dynamics from Social Media Timelines using A Pretrained Large
    Language Model with In-Context Learning." CLPsych 2025, pp. 287-291.

Shared design (paper §3)
------------------------
* LLM: Meta-Llama-3-8B-Instruct, no fine-tuning, temperature 0.1 (paper).
* Sampling and selection: generation is repeated over several rounds, each with
  freshly drawn random training examples, and the most likely candidate is
  chosen by Gaussian kernel density estimation over the candidates'
  all-MiniLM-L6-v2 embeddings. We use sklearn
  KernelDensity(kernel="gaussian", bandwidth="scott"); scikit-learn's "scott"
  bandwidth is n**(-1/(d+4)) applied to the raw (unit-normalized) embeddings,
  without per-dimension data scaling.

Task B (Fig. 1 template)
------------------------
* 5 random past analyses ("Analysis i:" blocks: post, adaptive segments,
  maladaptive segments, well-being, assessment).
* The target's segments and well-being are pre-filled with GOLD values, so
  the model only writes the "Assessment:" (summary). Only the summary varies
  across candidates, so the KDE is fitted on summary embeddings alone.

Task C (Fig. 2 template)
------------------------
* 3 random past timelines ("Past patient i:" blocks: the timeline's posts,
  then "Final Assessment:" with the gold timeline summary). The paper set 3
  because more examples "often" produced junk responses.
* Target input is the timeline posts only (the paper dropped evidence and
  post summaries for Task C).
* The prompt ends with the paper's literal instruction line
  "Final Assessment: <fill your assessment here; ...; must start with Final
  Assessment:>"; the leading "Final Assessment:" is stripped from outputs.

Project-convention adaptations (logged in the settings files):
* Candidates per post / timeline: 10 rounds. The paper does not give the
  number of rounds; it reports an overall "sample size" of 225, which we do
  not reproduce as a parameter (its mapping onto rounds is not stated —
  state this interpretation in the paper).
* Leave-one-timeline-out on every round's example draw.
* Deterministic per-(item, round) seeds (example draw + sampling), so resumed
  runs regenerate identical candidates.
* Empty candidates are excluded before the KDE fit.
* Context guard (Llama-3 has an 8K window): Task B drops examples down to 1;
  Task C first truncates every post (down to 100 words) while keeping all 3
  examples, then drops examples, then truncates further.
  The number of examples actually used (k_used) and any cap are recorded.
* MiniLM runs on CPU so it never competes with vLLM for GPU memory.
* Decoding beyond temperature: top_p 1.0, max_tokens 600 (paper silent).

Run on the cluster:
    python ism_model.py --data_dir <dir> --model /path/to/Meta-Llama-3-8B-Instruct --task bc
"""

import argparse
import datetime
import os
import random
import time
import zlib

import numpy as np

from system_utils import (
    Post, Timeline, load_timelines, span_texts, Checkpoint,
    write_task_b_submission, write_task_c_submission, write_settings_log,
    drop_until_fits, fit_timeline_prompt, truncate_words, task_c_targets,
    add_common_args, backend_from_args, common_settings, make_embedder,
)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEMPERATURE = 0.1        # paper-reported
DEFAULT_N_ROUNDS = 10            # our documented choice (paper: unspecified)
DEFAULT_N_EXAMPLES_B = 5         # paper: 5 past analyses
DEFAULT_N_EXAMPLES_C = 3         # paper: 3 past timelines

ISM_INTRO = ("You are a mental health expert and analyzing a patient's social media "
             "post to determine their well-being, their dominant self-state of either "
             "adaptive or maladaptive. The following is your past analysis.")

ISM_TEMPLATE_B = ISM_INTRO + """
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

# Fig. 2 — the final line is the paper's literal instruction.
ISM_TEMPLATE_C = ISM_INTRO + """
{past_patients}
Now analyze the following patient.
{posts}
Final Assessment: <fill your assessment here; it should be concise, and focus on change of self-state in the beginning, middle, and end of the post timeline; no need to mention detailed post contents; must start with Final Assessment:>"""


# ── Prompt construction ──────────────────────────────────────────────────────

def _segments_block(post: Post, polarity: str) -> str:
    spans = span_texts(post, polarity)
    return "\n".join(f"* {s}" for s in spans) if spans else "None identified"


def _wb(post: Post) -> str:
    return str(post.wellbeing) if post.wellbeing is not None else "Unknown"


def format_analysis(post: Post, index: int) -> str:
    return (f"Analysis {index}:\n{post.text}\n"
            f"Adaptive post segments:\n{_segments_block(post, 'adaptive')}\n"
            f"Maladaptive post segments:\n{_segments_block(post, 'maladaptive')}\n"
            f"Well-being: {_wb(post)}\n"
            f"Assessment:\n{post.summary}")


def build_messages_b(target: Post, examples: list[Post]) -> list[dict]:
    analyses = "\n".join(format_analysis(ex, i + 1) for i, ex in enumerate(examples))
    prompt = ISM_TEMPLATE_B.format(
        analyses=analyses if analyses else "(No past analyses available.)",
        post=target.text,
        adaptive_segments=_segments_block(target, "adaptive"),
        maladaptive_segments=_segments_block(target, "maladaptive"),
        wellbeing=_wb(target))
    return [{"role": "user", "content": prompt}]


def format_posts(tl: Timeline, cap: int | None) -> str:
    return "\n".join(truncate_words(p.text, cap) for p in tl.posts)


def build_messages_c(target: Timeline, examples: list[Timeline],
                     cap: int | None) -> list[dict]:
    past = "\n".join(
        f"Past patient {i}:\n{format_posts(ex, cap)}\nFinal Assessment:\n{ex.summary}"
        for i, ex in enumerate(examples, 1))
    prompt = ISM_TEMPLATE_C.format(
        past_patients=past if past else "(No past analyses available.)",
        posts=format_posts(target, cap))
    return [{"role": "user", "content": prompt}]


def strip_final_assessment(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("final assessment:"):
        t = t[len("final assessment:"):].strip()
    return t


def deterministic_seed(item_id: str, round_idx: int) -> int:
    """Stable per-(item, round) seed so resumed runs regenerate the same
    candidates as a full run would have."""
    return zlib.crc32(f"{item_id}|{round_idx}".encode("utf-8")) % (2**31 - 1)


# ── KDE selection ────────────────────────────────────────────────────────────

def kde_select(candidates: list[str], embedder) -> tuple[int, list]:
    """Index of the highest-density non-empty candidate (Gaussian KDE, sklearn
    "scott" bandwidth) and per-candidate log-densities (None for empty ones)."""
    from sklearn.neighbors import KernelDensity
    idx = [i for i, c in enumerate(candidates) if c.strip()]
    if not idx:
        return 0, [None] * len(candidates)
    if len(idx) == 1:
        dens = [None] * len(candidates)
        dens[idx[0]] = 0.0
        return idx[0], dens
    vecs = np.asarray(embedder.encode([candidates[i] for i in idx],
                                      normalize_embeddings=True,
                                      show_progress_bar=False))
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(vecs)
    logd = kde.score_samples(vecs)
    dens = [None] * len(candidates)
    for i, v in zip(idx, logd):
        dens[i] = round(float(v), 6)
    return idx[int(np.argmax(logd))], dens


# ── Generation ───────────────────────────────────────────────────────────────

def run_task_b(args, timelines, backend, embedder):
    all_posts = [p for tl in timelines for p in tl.posts]
    pool = [p for p in all_posts
            if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])]
    targets = [p for p in all_posts if p.summary]
    targets = targets[:args.limit] if args.limit else targets
    print(f"[Task B] example pool {len(pool)} | targets {len(targets)}")

    checkpoint = Checkpoint(os.path.join(args.output_dir, "checkpoint_ism_taskb.jsonl"))
    subm_path = os.path.join(args.output_dir, "ism_taskb_submission.json")
    pending = [p for p in targets if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"[Task B] pending {len(pending)} / {len(targets)}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, overrides, metas = [], [], []
        for p in batch:
            loto_pool = [q for q in pool if q.timeline_id != p.timeline_id]
            item = f"{p.timeline_id}|{p.post_index}"
            for r in range(args.n_rounds):
                seed = deterministic_seed(item, r)
                examples = random.Random(seed).sample(
                    loto_pool, min(args.n_examples_b, len(loto_pool)))
                msgs, k_used = drop_until_fits(
                    lambda k, p=p, examples=examples: build_messages_b(p, examples[:k]),
                    backend, len(examples), args.max_model_len, args.max_tokens,
                    min_examples=min(1, len(examples)))
                prompts.append(msgs)
                overrides.append({"seed": seed})
                metas.append((p, r, k_used, [e.post_id for e in examples[:k_used]]))
        outputs = backend.generate(prompts, sampling_overrides=overrides)

        by_post = {}
        for (p, r, k_used, ex_ids), s in zip(metas, outputs):
            e = by_post.setdefault(id(p), {"post": p, "cand": {}, "k": {}, "ex": {}})
            e["cand"][r], e["k"][r], e["ex"][r] = s, k_used, ex_ids
        for e in by_post.values():
            p = e["post"]
            cands = [e["cand"][r] for r in range(args.n_rounds)]
            chosen, dens = kde_select(cands, embedder)
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": cands[chosen],
                "chosen_round": chosen,
                "candidate_summaries": cands,
                "kde_log_density": dens,
                "k_used": [e["k"][r] for r in range(args.n_rounds)],
                "example_post_ids": [e["ex"][r] for r in range(args.n_rounds)],
                "gold_summary": p.summary,
            })
        write_task_b_submission(timelines, checkpoint, subm_path)
        print(f"[Task B] {min(start + args.batch_size, len(pending))}/{len(pending)} "
              f"({time.perf_counter() - t0:.1f}s)")
    write_task_b_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())

    write_settings_log(
        os.path.join(args.output_dir, "ism_taskb_settings.md"),
        "ISM Task B — generation settings",
        settings=[
            ("generation model", args.model, "paper (Meta-Llama-3-8B-Instruct)"),
            *common_settings(args),
            ("temperature", args.temperature, "paper (0.1)"),
            ("n_examples", args.n_examples_b, "paper (5 past analyses)"),
            ("candidates per post (rounds)", args.n_rounds,
             "our choice — paper does not give the number of rounds"),
            ("example selection", "random per (post, round), LOTO",
             "paper (random sampling); LOTO is a project convention"),
            ("selection", "Gaussian KDE, sklearn bandwidth='scott' "
                          "(n^(-1/(d+4)), no data scaling), over all-MiniLM-L6-v2 "
                          "embeddings of the non-empty candidate summaries",
             "paper (KDE, Scott's rule) + project convention (summary-only "
             "embedding: gold evidence is fixed across candidates)"),
            ("prompt", "paper Figure 1 template ('patent' -> 'patient'; target "
                       "sections pre-filled with gold)", "paper + project convention"),
            ("per-(post, round) seeds", "crc32(timeline|post|round)", "our choice"),
            ("evidence input", "gold spans + gold well-being",
             "project convention (original generated all fields jointly)"),
            ("rounds with fewer than n_examples (context guard)",
             sum(1 for r in recs for k in r.get("k_used", []) if k < args.n_examples_b),
             "reported"),
        ],
        notes=[
            "The original generated evidence, well-being and summary jointly and "
            "selected among full outputs; with gold evidence fixed only the "
            "summary varies, so the KDE is fitted on summary embeddings.",
            "The paper's 'sample size 225' is not reproduced as a parameter; its "
            "relation to the number of rounds is not stated in the paper.",
        ],
    )


def run_task_c(args, timelines, backend, embedder):
    pool = [tl for tl in timelines if tl.summary and tl.posts]
    targets = task_c_targets(timelines, args.limit)
    print(f"[Task C] example pool {len(pool)} timelines | targets {len(targets)}")

    checkpoint = Checkpoint(os.path.join(args.output_dir, "checkpoint_ism_taskc.jsonl"))
    subm_path = os.path.join(args.output_dir, "ism_taskc_submission.json")
    pending = [tl for tl in targets if not checkpoint.has(tl.timeline_id, "__timeline__")]
    print(f"[Task C] pending {len(pending)} / {len(targets)}")

    batch_tl = max(1, args.batch_size // 2)       # timelines are long
    for start in range(0, len(pending), batch_tl):
        batch = pending[start:start + batch_tl]
        t0 = time.perf_counter()
        prompts, overrides, metas = [], [], []
        for tl in batch:
            loto_pool = [x for x in pool if x.timeline_id != tl.timeline_id]
            for r in range(args.n_rounds):
                seed = deterministic_seed(f"{tl.timeline_id}|C", r)
                examples = random.Random(seed).sample(
                    loto_pool, min(args.n_examples_c, len(loto_pool)))
                msgs, k_used, cap = fit_timeline_prompt(
                    lambda k, cap, tl=tl, examples=examples:
                        build_messages_c(tl, examples[:k], cap),
                    backend, len(examples), args.max_model_len, args.max_tokens,
                    min_examples=0)
                prompts.append(msgs)
                overrides.append({"seed": seed})
                metas.append((tl, r, k_used, cap,
                              [e.timeline_id for e in examples[:k_used]]))
        outputs = backend.generate(prompts, sampling_overrides=overrides)

        by_tl = {}
        for (tl, r, k_used, cap, ex_ids), s in zip(metas, outputs):
            e = by_tl.setdefault(tl.timeline_id, {"tl": tl, "cand": {}, "k": {},
                                                  "cap": {}, "ex": {}})
            e["cand"][r] = strip_final_assessment(s)
            e["k"][r], e["cap"][r], e["ex"][r] = k_used, cap, ex_ids
        for e in by_tl.values():
            tl = e["tl"]
            cands = [e["cand"][r] for r in range(args.n_rounds)]
            chosen, dens = kde_select(cands, embedder)
            checkpoint.append(tl.timeline_id, "__timeline__", {
                "summary": cands[chosen],
                "chosen_round": chosen,
                "candidate_summaries": cands,
                "kde_log_density": dens,
                "k_used": [e["k"][r] for r in range(args.n_rounds)],
                "post_word_cap": [e["cap"][r] for r in range(args.n_rounds)],
                "example_timelines": [e["ex"][r] for r in range(args.n_rounds)],
                "gold_summary": tl.summary,
            })
        write_task_c_submission(timelines, checkpoint, subm_path)
        print(f"[Task C] {min(start + batch_tl, len(pending))}/{len(pending)} "
              f"({time.perf_counter() - t0:.1f}s)")
    write_task_c_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())
    ks = [k for r in recs for k in r.get("k_used", [])]

    write_settings_log(
        os.path.join(args.output_dir, "ism_taskc_settings.md"),
        "ISM Task C — generation settings",
        settings=[
            ("generation model", args.model, "paper (Meta-Llama-3-8B-Instruct)"),
            *common_settings(args),
            ("temperature", args.temperature, "paper (0.1)"),
            ("n_examples", args.n_examples_c, "paper (3 past timelines)"),
            ("candidates per timeline (rounds)", args.n_rounds,
             "our choice — paper does not give the number of rounds"),
            ("target input", "timeline posts only", "paper (evidence and post "
                                                     "summaries dropped for Task C)"),
            ("prompt", "paper Figure 2 template (final instruction line literal; "
                       "leading 'Final Assessment:' stripped from outputs)", "paper"),
            ("selection", "Gaussian KDE, sklearn bandwidth='scott', over "
                          "all-MiniLM-L6-v2 embeddings of non-empty candidates", "paper"),
            ("leave-one-timeline-out", True, "project convention"),
            ("mean examples actually used (8K context guard)",
             round(float(np.mean(ks)), 3) if ks else None, "reported"),
            ("rounds with truncated posts (context guard)",
             sum(1 for r in recs for c in r.get("post_word_cap", []) if c), "reported"),
        ],
        notes=[
            "Llama-3-8B's 8K window rarely fits three full past timelines plus "
            "the target; the guard truncates every post (to >=100 words) before "
            "dropping any example. Report the mean examples used and the "
            "truncation rate — they are part of the system as run.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="ISM (CLPsych 2025) Task B/C reimplementation.")
    add_common_args(parser, DEFAULT_MODEL, "./outputs/ism",
                    default_temperature=DEFAULT_TEMPERATURE, default_batch_size=8)
    parser.add_argument("--n_rounds", type=int, default=DEFAULT_N_ROUNDS,
                        help="Candidates per item (paper: unspecified; ours: 10).")
    parser.add_argument("--n_examples_b", type=int, default=DEFAULT_N_EXAMPLES_B)
    parser.add_argument("--n_examples_c", type=int, default=DEFAULT_N_EXAMPLES_C)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING,
                        help="KDE embedding model (paper: all-MiniLM-L6-v2).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    embedder = make_embedder(args.embedding_model, mock=args.backend == "mock",
                             device="cpu")
    backend = backend_from_args(args)
    if "b" in args.task:
        run_task_b(args, timelines, backend, embedder)
    if "c" in args.task:
        run_task_c(args, timelines, backend, embedder)
    print(f"\nISM run complete at {datetime.datetime.now():%Y%m%d-%H%M%S}.")


if __name__ == "__main__":
    main()
