"""uOttawa (CLPsych 2025) Task B system — reimplementation for the critique paper.

Reference:
    Chan, Khunkhun, Inkpen, Lossio-Ventura. "Prompt Engineering for Capturing
    Dynamic Mental Health Self-States from Social Media Posts." CLPsych 2025,
    pp. 256-267.

What this reproduces (paper §3 + Appendix A.3 Table 7, Task B)
--------------------------------------------------------------
* Pure prompt engineering on a local Llama-3.3-70B-Instruct. No fine-tuning,
  no retrieval, no evidence input — the ONLY input is the raw post text
  (their Task B template placeholder is {INCLUDE_TEXT (POST)}).
* Three official submission variants:
    S1 (uOttawa_1): contextual zero-shot — "loosely structured" prompt with
                    domain-specific terms (exact text not published; our
                    reconstruction follows the paper's description and is
                    flagged as such in the settings log).
    S2 (uOttawa_2): structured contextual one-shot — the paper's Table 7
                    template verbatim + one annotated training example.
                    (Their best Task B run: mean consistency 0.860.)
    S3 (uOttawa_3): structured contextual few-shot — same template + seven
                    examples ("carefully selected to cover a range of self
                    states, post lengths...").
* Output guideline enforced by the prompt: a single paragraph, under
  "### Summary:", no more than 6 sentences.

Project-convention adaptations (logged in the run's settings file):
* The paper hand-picked its S2/S3 examples; we cannot, so examples are
  selected programmatically with a fixed seed to cover self-state variety
  (adaptive/maladaptive/mixed) and post-length spread (documented
  reconstruction of "a range of self states, post lengths").
* The paper used no leave-one-timeline-out for its fixed examples (and the
  project LOTO requirement lists Aquarius/BULUSI/EAIonFlux/ISM only), so LOTO
  is OFF by default here; --loto drops/replaces same-timeline examples per
  target post.
* Decoding: the paper explicitly did not tune temperature/top_p ("could not
  ... explore ... hyperparameter tuning of temperature or top_p"); we use
  temperature 0.1, top_p 1.0, max_tokens 600 (documented defaults).

Run on the cluster (vLLM, local weights, tensor parallelism):
    python pipelines/uottawa_model.py --data_dir <timeline json dir> \
        --model /path/to/Llama-3.3-70B-Instruct --backend vllm --variant s2

Outputs (to --output_dir):
    checkpoint_uottawa_{variant}_taskb.jsonl   resumable per-post results
    uottawa_{variant}_taskb_submission.json    official submission format
    uottawa_{variant}_taskb_settings.md        generation-settings log
"""

import argparse
import datetime
import os
import random
import time

from system_utils import (
    Post, Timeline, load_timelines, make_backend, Checkpoint,
    write_task_b_submission, write_settings_log, drop_until_fits,
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

# S2/S3 — the paper's Table 7 one-shot template, VERBATIM as extracted from
# the paper (including the original's "consise" and "Fight or fight" typos).
# The example block is repeated for the few-shot variant (S3, 7 examples).
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
- Maladaptive Examples: Expectation that relatedness need will not be met, Expectation that autonomy need will not be met, Expectation that competence need will not be met
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


# ── Example selection (documented reconstruction of the hand-picked examples) ─

def select_examples(pool: list[Post], n: int, seed: int = EXAMPLE_SEED) -> list[Post]:
    """Programmatic stand-in for the paper's hand-picked examples.

    Stratifies by dominant self-state and post length so the selection covers
    "a range of self states, post lengths" (paper §3). Deterministic (seeded).
    """
    rng = random.Random(seed)
    by_len = sorted(pool, key=lambda p: len(p.text.split()))
    n_third = max(1, len(by_len) // 3)
    strata = {
        ("adaptive", "short"): [p for p in by_len[:n_third] if p.dominant_state() == "adaptive"],
        ("adaptive", "medium"): [p for p in by_len[n_third:2 * n_third] if p.dominant_state() == "adaptive"],
        ("adaptive", "long"): [p for p in by_len[2 * n_third:] if p.dominant_state() == "adaptive"],
        ("maladaptive", "short"): [p for p in by_len[:n_third] if p.dominant_state() == "maladaptive"],
        ("maladaptive", "medium"): [p for p in by_len[n_third:2 * n_third] if p.dominant_state() == "maladaptive"],
        ("maladaptive", "long"): [p for p in by_len[2 * n_third:] if p.dominant_state() == "maladaptive"],
        ("mixed", "any"): [p for p in by_len if p.dominant_state() == "mixed"],
    }
    order = list(strata.keys())
    picked: list[Post] = []
    for key in order:
        if len(picked) >= n:
            break
        candidates = [p for p in strata[key] if p not in picked]
        if candidates:
            picked.append(rng.choice(candidates))
    # Top up from the remaining pool if any stratum was empty.
    remaining = [p for p in pool if p not in picked]
    while len(picked) < n and remaining:
        picked.append(remaining.pop(rng.randrange(len(remaining))))
    return picked[:n]


# ── Prompt construction ──────────────────────────────────────────────────────

def build_messages(target: Post, variant: str,
                   examples: list[Post]) -> list[dict]:
    if variant == "s1":
        prompt = UOTTAWA_S1_PROMPT.format(post=target.text)
    else:
        example_block = "".join(
            EXAMPLE_SECTION.format(example_post=ex.text, example_summary=ex.summary)
            for ex in examples)
        prompt = UOTTAWA_STRUCTURED_PROMPT.format(
            example_block=example_block, post=target.text)
    return [{"role": "user", "content": prompt}]


def parse_summary(response: str) -> str:
    """Take the text after '### Summary:' if the model echoed the heading;
    otherwise the whole response."""
    marker = "### Summary:"
    if marker in response:
        response = response.split(marker, 1)[1]
    return response.strip().strip("-").strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="uOttawa (CLPsych 2025) Task B system reimplementation.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory of shared-task timeline JSON files.")
    parser.add_argument("--output_dir", default="./outputs/uottawa")
    parser.add_argument("--variant", default="s2", choices=["s1", "s2", "s3"],
                        help="S1=zero-shot, S2=one-shot (their best Task B run), "
                             "S3=few-shot (7 examples).")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "ollama", "mock"])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model path (local weights) or id. Default: {DEFAULT_MODEL} (paper).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None,
                        help="Optional vLLM quantization (e.g. 'fp8', 'awq') for 70B "
                             "on smaller GPUs.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Our documented default — paper explicitly did not tune "
                             "temperature/top_p.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Override example count (defaults: S2=1, S3=7, paper values).")
    parser.add_argument("--loto", action="store_true",
                        help="Drop/replace same-timeline examples per target post "
                             "(OFF by default — the paper used fixed hand-picked "
                             "examples and LOTO was not required for uOttawa).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    n_examples = args.n_examples if args.n_examples is not None else \
        {"s1": 0, "s2": 1, "s3": 7}[args.variant]

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    all_posts = [p for tl in timelines for p in tl.posts]
    example_pool = [p for p in all_posts if p.summary]
    targets = [p for p in all_posts if p.summary]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Example pool: {len(example_pool)} posts | targets: {len(targets)} "
          f"| variant: {args.variant} ({n_examples} examples)")

    # Fixed examples (S2/S3) — selected once, as in the paper.
    fixed_examples = select_examples(example_pool, n_examples) if n_examples else []
    for ex in fixed_examples:
        print(f"  Example: {ex.timeline_id}#{ex.post_index} "
              f"({ex.dominant_state()}, {len(ex.text.split())} words)")

    backend = make_backend(
        args.backend, args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype, quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, seed=args.seed,
    )

    # ── Generation loop (batched, resumable) ────────────────────────────────
    ckpt_path = os.path.join(args.output_dir,
                             f"checkpoint_uottawa_{args.variant}_taskb.jsonl")
    checkpoint = Checkpoint(ckpt_path)
    subm_path = os.path.join(args.output_dir,
                             f"uottawa_{args.variant}_taskb_submission.json")

    pending = [p for p in targets
               if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"Pending targets: {len(pending)} / {len(targets)} (checkpoint resumed)")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for p in batch:
            examples = fixed_examples
            if args.loto:
                examples = [ex for ex in fixed_examples
                            if ex.timeline_id != p.timeline_id]
                if len(examples) < len(fixed_examples):
                    # Replace dropped examples with same-pool picks from
                    # other timelines, preserving the count.
                    extra = select_examples(
                        [q for q in example_pool
                         if q.timeline_id != p.timeline_id and q not in examples],
                        len(fixed_examples) - len(examples))
                    examples = examples + extra
            msgs, k_used = drop_until_fits(
                lambda k: build_messages(p, args.variant, examples[:k]), backend,
                len(examples), args.max_model_len, args.max_tokens, min_examples=0)
            prompts.append(msgs)
            metas.append((p, examples[:k_used]))
        responses = backend.generate(prompts)
        for (p, examples), response in zip(metas, responses):
            summary = parse_summary(response)
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": summary,
                "raw_response": response,
                "variant": args.variant,
                "example_post_ids": [e.post_id for e in examples],
                "example_timelines": [e.timeline_id for e in examples],
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
        os.path.join(args.output_dir, f"uottawa_{args.variant}_taskb_settings.md"),
        f"uOttawa Task B (variant {args.variant}) — generation settings",
        settings=[
            ("generation model", args.model, "paper (Llama-3.3-70B-Instruct, local)"),
            ("backend", args.backend, "our setup (vLLM, local weights)"),
            ("tensor_parallel_size", args.tensor_parallel_size, "our setup"),
            ("variant", args.variant,
             "paper (S1 zero-shot / S2 one-shot / S3 few-shot; S2 was their best "
             "Task B submission, mean consistency 0.860)"),
            ("n_examples", n_examples, "paper (S1: 0, S2: 1, S3: 7)"),
            ("example selection",
             f"programmatic stratified (dominant state x length), seed {EXAMPLE_SEED}",
             "our reconstruction — paper hand-picked examples (text unpublished)"),
            ("prompt (s2/s3)",
             "paper Table 7 template verbatim (incl. original typos 'consise', "
             "'Fight or fight')", "paper"),
            ("prompt (s1)", "reconstructed loose zero-shot",
             "our reconstruction — paper describes but does not publish S1's text"),
            ("temperature", args.temperature,
             "our default — paper explicitly did not tune decoding hyperparameters"),
            ("top_p", args.top_p, "our default — paper silent"),
            ("max_tokens", args.max_tokens, "our default — paper silent"),
            ("input", "raw post text only (no evidence spans, no well-being)",
             "paper (Task B placeholder: {INCLUDE_TEXT (POST)})"),
            ("leave-one-timeline-out", args.loto,
             "OFF by default — paper used fixed examples; LOTO not required for "
             "uOttawa per project decision"),
        ],
        notes=[
            "uOttawa Task B is prompt-only on the raw post text; unlike the "
            "other systems it receives no evidence spans, which is faithful to "
            "the original (gold evidence convention does not apply because "
            "their prompt has no evidence slot).",
            "The S2/S3 prompt keeps two typos that appear in the published "
            "Table 7 ('consise', 'Fight or fight behavior'); flagged here so "
            "the reimplementation section can note they are the paper's own.",
            "Example posts are selected programmatically (seeded stratified "
            "sampling over dominant self-state and post length) because the "
            "paper's hand-picked examples were redacted.",
        ],
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\nuOttawa Task B ({args.variant}) run complete at {ts}. "
          f"Submission: {subm_path}")


if __name__ == "__main__":
    main()
