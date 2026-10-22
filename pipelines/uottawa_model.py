"""uOttawa (CLPsych 2025) Task B + Task C system — reimplementation for the critique paper.

Reference:
    Chan, Khunkhun, Inkpen, Lossio-Ventura. "Prompt Engineering for Capturing
    Dynamic Mental Health Self-States from Social Media Posts." CLPsych 2025,
    pp. 256-267.

Task B (paper §3 + Appendix A.3 Table 7)
----------------------------------------
* Prompt engineering on local Llama-3.3-70B-Instruct; no fine-tuning, no
  retrieval, no evidence input — the only input is the raw post text.
* Variants: S1 contextual zero-shot (text unpublished; our reconstruction),
  S2 structured one-shot (Table 7 + 1 example; their best Task B run),
  S3 structured few-shot (Table 7 + 7 examples).

Task C (Appendix A.3 Table 8)
-----------------------------
* Input: the chronologically ordered post-level summaries of the timeline
  produced by Task B ({INCLUDE_TEXT (Summaries of posts on specific timeline
  from Task B)}). We use this system's OWN Task B summaries of the same
  variant, read from checkpoint_uottawa_{variant}_taskb.jsonl — so Task C
  must run after Task B (--task bc does both in order).
* Example block: a training timeline's gold post summaries (input) and its
  gold timeline summary (output). S2 = 1 example; S3 = 3 examples (the paper
  reports few-shot working better than one-shot for timelines); S1 =
  zero-shot reconstruction (text unpublished).
* Output heading "### Timeline Summary:"; at most 6 sentences.

Project-convention adaptations (logged in the settings files):
* Hand-picked examples (unpublished) are replaced by seeded programmatic
  selection (Task B: stratified by dominant self-state x post length;
  Task C: seeded random training timelines).
* Leave-one-timeline-out is ON by default (--no_loto to disable): a fixed
  example from the target's own timeline would place that timeline's gold
  summary in the prompt. Same-timeline examples are replaced, keeping the count.
* Table 7 text is transcribed as published, including its typos ("consise",
  "Fight or fight behavior").
* Decoding: the paper explicitly did not tune temperature/top_p; we use
  temperature 0.1, top_p 1.0, max_tokens 600.

Run on the cluster:
    python uottawa_model.py --data_dir <dir> --model /path/to/Llama-3.3-70B-Instruct \
        --variant s2 --task bc
"""

import argparse
import datetime
import os
import random
import time

from system_utils import (
    Post, Timeline, load_timelines, Checkpoint,
    write_task_b_submission, write_task_c_submission, write_settings_log,
    drop_until_fits, fit_timeline_prompt, truncate_words, task_c_targets,
    add_common_args, backend_from_args, common_settings,
)

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
EXAMPLE_SEED = 42

# S1 — "contextual zero-shot" reconstruction. The paper: "Prompts were loosely
# structured, with less explicit delineation between sections such as
# objectives, definitions, and output guidelines. Domain-specific terms
# (e.g. adaptive and maladaptive self states) were included into the prompts
# to provide context." Exact text not published.
UOTTAWA_S1_PROMPT = """You are a mental health expert analyzing a person's social media post to understand their self-states. Using your knowledge of adaptive and maladaptive self-states — identifiable units of Affect, Behavior, and Cognition towards the self or others, where adaptive states help fulfill basic desires and needs such as relatedness, autonomy and competence, and maladaptive states hinder the fulfillment of those needs — write a summary of the interplay between the self-states expressed in the post below. Begin by determining which self-state is dominant and describe it first, identifying the central aspect that drives the state and how it influences the rest; then cover the other self-state if present, emphasizing potential causal relationships. If the post contains only one self-state, summarize only that state. Write the answer as a single clear paragraph of no more than 6 sentences.

Input text:
{post}"""

# S1 Task C — same loose style, reconstructed from Table 8's task statement.
UOTTAWA_S1_TIMELINE_PROMPT = """You are a mental health expert analyzing a person's timeline of social media posts to understand how their self-states change over time. Using your knowledge of adaptive and maladaptive self-states — identifiable units of Affect, Behavior, and Cognition towards the self or others, where adaptive states help fulfill basic desires and needs such as relatedness, autonomy and competence, and maladaptive states hinder the fulfillment of those needs — write a summary of the interplay between adaptive and maladaptive self-states along the timeline, given the chronologically ordered summaries of its posts below. Emphasize temporal dynamics such as flexibility, rigidity, improvement, and deterioration, and, if applicable, describe how the dominance of the self-states changes over time and how changes in affect, behavior, cognition, and desire contribute to these transitions. Write the answer as a single clear paragraph of no more than 6 sentences.

Input text:
{summaries}"""

# S2/S3 Task B — the paper's Table 7 template as published (including the
# original's "consise" and "Fight or fight" typos). The example block is
# repeated for the few-shot variant (S3, 7 examples).
UOTTAWA_STRUCTURED_PROMPT = """### Task:
Your task is to summarize self-states for the social media post below. Specifically, generate a summary of the interplay between adaptive and maladaptive states identified in the post. Begin by determining which self-state is dominant (adaptive/maladaptive) and describe it first. For each self-state, identify the central organizing aspect (A, B, C, or D) that drives the state and structure the summary around it. Describe how this central aspect influences the rest, emphasizing potential causal relationships between them. Then, proceed to the second self-state and follow the same approach. If the post contains only one self-state (either adaptive or maladaptive), summarize only that state. Note that the summary does not need to explicitly highlight A, B, C, or D, but should aim to naturally integrate these elements into the description.
—
### Definitions:
Self-states constitute identifiable units characterized by specific combinations of Affect, Behavior, Cognition, and Desire/Need (ABCD dimensions) that tend to be coactivated in a meaningful manner for limited periods of time.
- An adaptive self-state pertains to aspects of Affect, Behaviour, and Cognition towards the self or others, which is conducive to the fulfillment of basic desires/needs (D), such as relatedness, autonomy and competence.
- A maladaptive self-state pertains to aspects of Affect, Behaviour, and Cognition towards the self or others, that hinder the fulfillment of basic desires/needs (D).
### ABCD dimensions:
1. Affect (A): The type of emotion expressed by the person.
- Adaptive Examples: Calm/Laid back, Emotional Pain/Grieving, Content/Happy, Vigor/Energetic, Justifiable, Anger/Assertive Anger, Proud.
- Maladaptive Examples: Anxious/Tense/Fearful, Depressed/Despair/Hopeless, Mania, Apathetic/Don't care/Blunted, Angry (Aggressive, Disgust, Contempt), Ashamed/Guilty.
2. Behavior of the self with the Other (BO) : The person's main behavior(s) toward the other
- Adaptive Examples: Relating behavior, Autonomous behavior
- Maladaptive Examples: Fight or fight behavior, Overcontrolled/controlling behavior
3. Behavior toward the Self (BS): The person's main behavior(s) toward the self
- Adaptive Examples: Self-care behavior
- Maladaptive Examples: Self-harm, Neglect, Avoidance behavior
4. Cognition of the Other (CO): The person's main perceptions of the other
- Adaptive Examples: Perception of the other as related, Perception of the other as facilitating autonomy needs
- Maladaptive Examples: Perception of the other as detached or over attached, Perception of the other as blocking autonomy needs
5. Cognition of the Self (CS): How the person perceives themselves.
- Adaptive Examples: Self-acceptance and self-compassion
- Maladaptive Examples: Self-criticism
6. Desire (D): The person's main desire, need, intention, fear or expectation
- Adaptive Examples: Relatedness, Autonomy and adaptive control, Competence, Self-esteem, Self-care
- Maladaptive Examples: Expectation that relatedness need will not be met, Expectation that autonomy needs will not be met, Expectation that competence needs will not be met
—
### Guidelines for Output:
- Response Section: Provide an answer under the headings '### Summary:'.
- Format the answer as a single paragraph, making it clear and consise.
- The summary should be no more than 6 sentences. - Ensure the summary captures the main points without unnecessary details.
—
{example_block}### Analyze the following input text based on the given criteria.
### Input Text:
{post}
### Output Text:"""

EXAMPLE_SECTION = """### Example:
##### Input text:
{example_post}
##### Output text:
### Summary:
{example_summary}
—
"""

# S2/S3 Task C — Table 8. Its Definitions and ABCD sections are identical to
# Table 7's, so they are sliced from the Task B template rather than retyped.
_DEFS_ABCD = UOTTAWA_STRUCTURED_PROMPT[
    UOTTAWA_STRUCTURED_PROMPT.index("### Definitions:"):
    UOTTAWA_STRUCTURED_PROMPT.index("—\n### Guidelines for Output:")]

UOTTAWA_TIMELINE_PROMPT = ("""### Task:
Your task is to summarize self-states for each timeline, given the summaries for each post on the timeline. Specifically, generate a summary focusing on the Interplay between adaptive and maladaptive self-states along the timeline. Emphasize temporal dynamics focusing on concepts such as flexibility, rigidity, improvement, and deterioration. If applicable, describe the extent to which the dominance of the self-states changes over time and how changes in aspects (Affect, Behavior, Cognition, and Desire) contribute to these transitions.
—
""" + _DEFS_ABCD + """—
### Guidelines for Output:
- Response Section: Provide an answer under the headings '### Timeline Summary:'.
- Format the answer as a single paragraph, making it clear and consise.
- The summary should be no more than 6 sentences.
- Ensure the timeline summary captures the main points without unnecessary details.
—
{example_block}### Analyze the following input text based on the given criteria.
### Input Text:
{summaries}
### Output Text:""")

TIMELINE_EXAMPLE_SECTION = """### Example:
##### Input text:
{example_summaries}
##### Output text:
### Timeline Summary:
{example_timeline_summary}
—
"""


# ── Example selection (documented reconstruction of hand-picked examples) ────

def select_examples(pool: list[Post], n: int, seed: int = EXAMPLE_SEED) -> list[Post]:
    """Stratify by dominant self-state and post length so the selection covers
    "a range of self states, post lengths" (paper §3). Deterministic."""
    rng = random.Random(seed)
    by_len = sorted(pool, key=lambda p: (len(p.text.split()), p.timeline_id, p.post_index))
    n_third = max(1, len(by_len) // 3)
    thirds = {"short": by_len[:n_third], "medium": by_len[n_third:2 * n_third],
              "long": by_len[2 * n_third:]}
    strata = [[p for p in thirds[l] if p.dominant_state() == s]
              for s in ("adaptive", "maladaptive") for l in ("short", "medium", "long")]
    strata.append([p for p in by_len if p.dominant_state() == "mixed"])
    picked: list[Post] = []
    for stratum in strata:
        if len(picked) >= n:
            break
        cands = [p for p in stratum if p not in picked]
        if cands:
            picked.append(rng.choice(cands))
    remaining = [p for p in by_len if p not in picked]
    while len(picked) < n and remaining:
        picked.append(remaining.pop(rng.randrange(len(remaining))))
    return picked[:n]


def loto_examples(fixed: list, pool: list, target_tl: str, pick) -> list:
    """Replace fixed examples from the target's own timeline with picks from
    other timelines, preserving the count and the order of the rest."""
    kept = [e for e in fixed if e.timeline_id != target_tl]
    if len(kept) == len(fixed):
        return fixed
    extra_pool = [q for q in pool if q.timeline_id != target_tl and q not in kept]
    return kept + pick(extra_pool, len(fixed) - len(kept))


def select_timelines(pool: list[Timeline], n: int, seed: int = EXAMPLE_SEED) -> list[Timeline]:
    pool = sorted(pool, key=lambda t: t.timeline_id)
    return random.Random(seed).sample(pool, min(n, len(pool)))


# ── Prompt construction ──────────────────────────────────────────────────────

def build_messages_b(target: Post, variant: str, examples: list[Post]) -> list[dict]:
    if variant == "s1":
        prompt = UOTTAWA_S1_PROMPT.format(post=target.text)
    else:
        example_block = "".join(
            EXAMPLE_SECTION.format(example_post=ex.text, example_summary=ex.summary)
            for ex in examples)
        prompt = UOTTAWA_STRUCTURED_PROMPT.format(example_block=example_block,
                                                  post=target.text)
    return [{"role": "user", "content": prompt}]


def format_summaries(summaries: list[str], cap: int | None) -> str:
    return "\n".join(f"Post {i}: {truncate_words(s, cap)}"
                     for i, s in enumerate(summaries, 1))


def build_messages_c(input_summaries: list[str], variant: str,
                     examples: list[Timeline], cap: int | None) -> list[dict]:
    if variant == "s1":
        prompt = UOTTAWA_S1_TIMELINE_PROMPT.format(
            summaries=format_summaries(input_summaries, cap))
    else:
        example_block = "".join(
            TIMELINE_EXAMPLE_SECTION.format(
                example_summaries=format_summaries(
                    [p.summary for p in ex.posts if p.summary], cap),
                example_timeline_summary=ex.summary)
            for ex in examples)
        prompt = UOTTAWA_TIMELINE_PROMPT.format(
            example_block=example_block,
            summaries=format_summaries(input_summaries, cap))
    return [{"role": "user", "content": prompt}]


def parse_after(response: str, marker: str) -> str:
    """Text after `marker` if the model echoed the heading, else the response."""
    if marker in response:
        response = response.split(marker, 1)[1]
    return response.strip().strip("-").strip()


# ── Generation ───────────────────────────────────────────────────────────────

def run_task_b(args, timelines, backend):
    n_examples = args.n_examples_b if args.n_examples_b is not None else \
        {"s1": 0, "s2": 1, "s3": 7}[args.variant]
    all_posts = [p for tl in timelines for p in tl.posts]
    pool = [p for p in all_posts if p.summary]
    targets = pool[:args.limit] if args.limit else pool
    fixed = select_examples(pool, n_examples) if n_examples else []
    print(f"[Task B] variant {args.variant} | {n_examples} examples | targets {len(targets)}")
    for ex in fixed:
        print(f"  example {ex.timeline_id}#{ex.post_index} ({ex.dominant_state()}, "
              f"{len(ex.text.split())} words)")

    checkpoint = Checkpoint(os.path.join(
        args.output_dir, f"checkpoint_uottawa_{args.variant}_taskb.jsonl"))
    subm_path = os.path.join(args.output_dir, f"uottawa_{args.variant}_taskb_submission.json")
    pending = [p for p in targets if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"[Task B] pending {len(pending)} / {len(targets)}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for p in batch:
            examples = fixed if args.no_loto else \
                loto_examples(fixed, pool, p.timeline_id, select_examples)
            msgs, k_used = drop_until_fits(
                lambda k, p=p, examples=examples: build_messages_b(p, args.variant, examples[:k]),
                backend, len(examples), args.max_model_len, args.max_tokens, min_examples=0)
            prompts.append(msgs)
            metas.append((p, examples[:k_used]))
        responses = backend.generate(prompts)
        dt = time.perf_counter() - t0
        for (p, examples), response in zip(metas, responses):
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": parse_after(response, "### Summary:"),
                "raw_response": response,
                "variant": args.variant,
                "n_examples": len(examples),
                "example_post_ids": [e.post_id for e in examples],
                "example_timelines": [e.timeline_id for e in examples],
                "gold_summary": p.summary,
                "generation_sec": round(dt / len(batch), 2),
            })
        write_task_b_submission(timelines, checkpoint, subm_path)
        print(f"[Task B] {min(start + args.batch_size, len(pending))}/{len(pending)} ({dt:.1f}s)")
    write_task_b_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())

    write_settings_log(
        os.path.join(args.output_dir, f"uottawa_{args.variant}_taskb_settings.md"),
        f"uOttawa Task B (variant {args.variant}) — generation settings",
        settings=[
            ("generation model", args.model, "paper (Llama-3.3-70B-Instruct, local)"),
            *common_settings(args),
            ("temperature", args.temperature,
             "our default — paper explicitly did not tune decoding hyperparameters"),
            ("variant", args.variant, "paper (S1 zero-shot / S2 one-shot / S3 few-shot)"),
            ("n_examples", n_examples, "paper (S1: 0, S2: 1, S3: 7)"),
            ("example selection",
             f"programmatic stratified (dominant state x length), seed {EXAMPLE_SEED}",
             "our reconstruction — paper hand-picked examples (unpublished)"),
            ("prompt (s2/s3)", "paper Table 7 template as published, incl. its "
             "typos 'consise' and 'Fight or fight'", "paper"),
            ("prompt (s1)", "reconstructed loose zero-shot",
             "our reconstruction — paper describes but does not publish S1"),
            ("input", "raw post text only", "paper (Task B placeholder {INCLUDE_TEXT (POST)})"),
            ("leave-one-timeline-out", not args.no_loto,
             "project convention (a same-timeline example would expose that "
             "timeline's gold summary); same-timeline examples are replaced"),
            ("posts with fewer than n_examples (context guard)",
             sum(1 for r in recs if r.get("n_examples", 0) < n_examples), "reported"),
        ],
        notes=[
            "uOttawa Task B receives no evidence spans — faithful to the original, "
            "whose prompt has no evidence slot.",
            "Example posts are selected programmatically because the paper's "
            "hand-picked examples are not published.",
        ],
    )


def run_task_c(args, timelines, backend):
    n_examples = args.n_examples_c if args.n_examples_c is not None else \
        {"s1": 0, "s2": 1, "s3": 3}[args.variant]
    pool = [tl for tl in timelines if tl.summary and any(p.summary for p in tl.posts)]
    targets = task_c_targets(timelines, args.limit)
    fixed = select_timelines(pool, n_examples) if n_examples else []

    b_path = os.path.join(args.output_dir, f"checkpoint_uottawa_{args.variant}_taskb.jsonl")
    if not os.path.exists(b_path):
        raise SystemExit(f"Task C needs this variant's Task B output first: {b_path} "
                         f"(run with --task b or --task bc).")
    b_ckpt = Checkpoint(b_path)
    print(f"[Task C] variant {args.variant} | {n_examples} examples | targets {len(targets)}")

    checkpoint = Checkpoint(os.path.join(
        args.output_dir, f"checkpoint_uottawa_{args.variant}_taskc.jsonl"))
    subm_path = os.path.join(args.output_dir, f"uottawa_{args.variant}_taskc_submission.json")
    pending = [tl for tl in targets if not checkpoint.has(tl.timeline_id, "__timeline__")]
    print(f"[Task C] pending {len(pending)} / {len(targets)}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for tl in batch:
            inputs, missing = [], 0
            for p in tl.posts:
                rec = b_ckpt.get(tl.timeline_id, p.post_id)
                if rec and rec.get("summary"):
                    inputs.append(rec["summary"])
                elif p.summary:          # annotated post without a Task B output
                    missing += 1
            if missing:
                print(f"  WARNING {tl.timeline_id}: {missing} annotated posts have "
                      f"no Task B summary (incomplete Task B run?)")
            examples = fixed if args.no_loto else loto_examples(
                fixed, pool, tl.timeline_id,
                lambda extra, n: select_timelines(extra, n))
            msgs, k_used, cap = fit_timeline_prompt(
                lambda k, cap, inputs=inputs, examples=examples:
                    build_messages_c(inputs, args.variant, examples[:k], cap),
                backend, len(examples), args.max_model_len, args.max_tokens,
                min_examples=0)
            prompts.append(msgs)
            metas.append((tl, examples[:k_used], cap, len(inputs), missing))
        responses = backend.generate(prompts)
        dt = time.perf_counter() - t0
        for (tl, examples, cap, n_in, missing), response in zip(metas, responses):
            checkpoint.append(tl.timeline_id, "__timeline__", {
                "summary": parse_after(response, "### Timeline Summary:"),
                "raw_response": response,
                "variant": args.variant,
                "n_examples": len(examples),
                "example_timelines": [e.timeline_id for e in examples],
                "n_input_summaries": n_in,
                "n_missing_taskb": missing,
                "post_word_cap": cap,
                "gold_summary": tl.summary,
                "generation_sec": round(dt / len(batch), 2),
            })
        write_task_c_submission(timelines, checkpoint, subm_path)
        print(f"[Task C] {min(start + args.batch_size, len(pending))}/{len(pending)} ({dt:.1f}s)")
    write_task_c_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())

    write_settings_log(
        os.path.join(args.output_dir, f"uottawa_{args.variant}_taskc_settings.md"),
        f"uOttawa Task C (variant {args.variant}) — generation settings",
        settings=[
            ("generation model", args.model, "paper (Llama-3.3-70B-Instruct, local)"),
            *common_settings(args),
            ("temperature", args.temperature, "our default — paper did not tune"),
            ("variant", args.variant, "paper"),
            ("n_examples", n_examples,
             "S2: 1 (one-shot); S3: 3 — our choice of count for 'few-shot' "
             "(paper reports few-shot > one-shot for timelines)"),
            ("example selection", f"seeded random training timelines, seed {EXAMPLE_SEED}",
             "our reconstruction — paper's examples unpublished"),
            ("example content", "gold post summaries (input) + gold timeline summary (output)",
             "paper Table 8 example section"),
            ("input", f"this system's own Task B summaries (variant {args.variant}), "
                      "chronological", "paper Table 8 ({INCLUDE_TEXT (Summaries of "
                      "posts on specific timeline from Task B)})"),
            ("prompt (s2/s3)", "paper Table 8 (Definitions/ABCD identical to Table 7)", "paper"),
            ("prompt (s1)", "reconstructed loose zero-shot", "our reconstruction"),
            ("leave-one-timeline-out", not args.no_loto, "project convention"),
            ("timelines with missing Task B summaries",
             sum(1 for r in recs if r.get("n_missing_taskb")), "reported"),
            ("timelines with fewer than n_examples (context guard)",
             sum(1 for r in recs if r.get("n_examples", 0) < n_examples), "reported"),
        ],
        notes=[
            "Task C inherits this variant's Task B outputs, as in the original "
            "two-level pipeline.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="uOttawa (CLPsych 2025) Task B/C reimplementation.")
    add_common_args(parser, DEFAULT_MODEL, "./outputs/uottawa")
    parser.add_argument("--variant", default="s2", choices=["s1", "s2", "s3"])
    parser.add_argument("--n_examples_b", type=int, default=None,
                        help="Override Task B example count (S1=0, S2=1, S3=7).")
    parser.add_argument("--n_examples_c", type=int, default=None,
                        help="Override Task C example count (S1=0, S2=1, S3=3).")
    parser.add_argument("--no_loto", action="store_true",
                        help="Disable leave-one-timeline-out (not recommended: a "
                             "same-timeline example leaks that timeline's gold summary).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    backend = backend_from_args(args)
    if "b" in args.task:
        run_task_b(args, timelines, backend)
    if "c" in args.task:
        run_task_c(args, timelines, backend)
    print(f"\nuOttawa ({args.variant}) run complete at {datetime.datetime.now():%Y%m%d-%H%M%S}.")


if __name__ == "__main__":
    main()
