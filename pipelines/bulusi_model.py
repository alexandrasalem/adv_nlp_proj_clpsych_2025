"""BULUSI (CLPsych 2025) Task B + Task C system — reimplementation for the critique paper.

Reference:
    Ravenda, Kara-Isitt, Swift, Mira, Raballo. "From Evidence Mining to
    Meta-Prediction: a Gradient of Methodologies for Task-Specific Challenges
    in Psychological Assessment." CLPsych 2025, pp. 242-248.
    Team code/prompts: https://github.com/Fede-stack/BULUSI-CLPsych

Task B (paper §2.2 + Appendix A)
--------------------------------
* Few-shot summarizer; the best Task B submission used qwen-2.5-72b (the
  mixtral-8x22b / llama-3.1-405b variants are not reimplemented — project
  decision: 72B for both tasks).
* Examples: contriever top-50 re-ranked by ms-marco-MiniLM-L-6-v2 to the 5
  most similar training posts.
* Prompt: prompts/Summary.txt from the team repo, verbatim (placeholder
  "{adaptive/maladaptive}" renamed "{adaptive_maladaptive}" for str.format).

Task C
------
* Prompt: prompts/"Timeline Summary.txt" from the team repo, verbatim
  (placeholders {content} = example timeline summaries, {Results} = the
  target timeline's data).
* {Results}: every post in chronological order with its text and its
  adaptive/maladaptive states (paper Fig. 1: the timeline summarizer sees the
  posts across the timeline plus the evidence; the prompt's own heading says
  the block holds "the adaptive and maladaptive states identified in the
  user's posts over time").
* {content}: gold summaries of the 5 most similar training timelines, using
  the same contriever top-50 -> cross-encoder top-5 pipeline at timeline
  level (paper describes retrieval only for posts; timeline-level use is our
  extension of it, documented).

Project-convention adaptations (logged in the settings files):
* GOLD evidence spans in every states slot (original: BULUSI's G-RAG predictions).
* Leave-one-timeline-out for both tasks.
* Decoding: paper silent; temperature 0.1, top_p 1.0, max_tokens 600.
* Timeline embedding for contriever = normalized mean of post embeddings
  (contriever truncates at 512 tokens); the cross-encoder sees the first
  512 tokens of each (target, candidate) concatenated-timeline pair.
* Retrieval runs before the LLM is loaded; retrieval models are freed.

Run on the cluster:
    python bulusi_model.py --data_dir <dir> --model /path/to/Qwen2.5-72B-Instruct --task bc

Outputs (to --output_dir):
    checkpoint_bulusi_task{b,c}.jsonl, bulusi_task{b,c}_submission.json,
    bulusi_task{b,c}_settings.md
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
    MockEmbedder, MockCrossEncoder, free_gpu_memory, timeline_text,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_CONTRIEVER = "facebook/contriever"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_N_EXAMPLES = 5          # paper: 5 most similar
DEFAULT_N_CANDIDATES = 50       # paper: contriever top-50 first stage

# BULUSI Task B prompt — VERBATIM from BULUSI-CLPsych/prompts/Summary.txt
# (placeholders {content}, {post}, {adaptive_maladaptive} filled at runtime).
BULUSI_PROMPT = """# Adaptive vs. Maladaptive Self-States Analysis

## Task Description
You will analyze social media posts to summarise  evidence of adaptive and maladaptive self-states. 

## Definitions
- **Self-states** are identifiable units characterized by specific combinations of Affect, Behavior, Cognition, and Desire/Need (ABCD) that tend to be coactivated in a meaningful manner for limited periods of time.

- **Adaptive self-state**: Aspects of Affect, Behavior, and Cognition towards the self or others that are conducive to fulfilling basic desires/needs such as relatedness, autonomy, and competence.

- **Maladaptive self-state**: Aspects of Affect, Behavior, and Cognition towards the self or others that hinder the fulfillment of basic desires/needs.

## ABCD Framework Components
1. **Affect (A)**: The emotions expressed
   - Adaptive affects: Calm/laid back, content/happy, emotional pain/grieving, vigor/energetic, justifiable anger, proud
   - Maladaptive affects: Anxious/tense/fearful, depressed/desperate/hopeless, mania, apathetic, aggressive anger, shame/guilt

2. **Behavior (B)**: Actions toward self (BS) or others (BO)
   - Adaptive behaviors: Self-care, relating to others, autonomous behavior
   - Maladaptive behaviors: Self-harm, neglect, avoidance, fight or flight, overcontrolling

3. **Cognition (C)**: Perceptions of self (CS) or others (CO)
   - Adaptive cognitions: Self-acceptance, seeing others as related, seeing others as supporting autonomy
   - Maladaptive cognitions: Self-criticism, seeing others as detached or over-attached, seeing others as blocking autonomy

4. **Desire/Need (D)**: Main desires, expectations, or fears
   - Adaptive desires: Relatedness, autonomy, competence, self-esteem, self-care
   - Maladaptive expectations: Expectation that needs will not be met (relatedness, autonomy, competence)

## Important Interpretation Guidelines
- Look beyond surface-level negativity to identify underlying adaptive elements
- When analyzing text, consider the underlying motivations, connections, and desires reflected in the person's words:
  - Statements that show connection to others, even during suffering, reflect adaptive relatedness
  - Actions taken to maintain relationships, even with difficulty, are adaptive
  - Expressions of caring for others or being cared for by others are adaptive
  - Efforts to improve one's situation, even amidst distress, show adaptive self-care
  - Recognition of one's needs while experiencing challenges is adaptive
  - The person is motivated to stay alive for others (adaptive behavior)

- Adaptive elements can coexist with expressions of suffering. The presence of negative emotions does not automatically make a statement maladaptive if it also shows connection, hope, or positive intention, e.g. sentence like "All I'm doing with my life is surviving for the sake of a handful of people who don't want me to die" should be classified as adaptive despite expressing suffering, because they show that user is motivated to stay alive for others.

## Critical Content Filtering Guidelines
- Only content that directly expresses the user's mental states, emotions, behaviors, or desires should be classified as adaptive or maladaptive
- Explicitly exclude:
  - News headlines, article shares, or external content 
  - References to celebrities, movies, music, or media that don't express the user's state
  - Informational or factual posts that reveal nothing about the user's mental state
  - General questions or requests for information that don't reveal the user's state
  - Quotes or references to external content without personal commentary

- To be classified as adaptive or maladaptive, text MUST:
  1. Directly reflect the user's personal experience (use first-person pronouns or clearly refer to themselves)
  2. Contain at least one ABCD element that reveals something about the user's mental state
  3. Show how the user relates to themselves or others


## Your Task
Your summary should:
1. Begin by determining which self-state is dominant (adaptive/maladaptive) and describe it first
2. For each self-state, identify the central organizing aspect (A, B, C, or D) that drives the state
3. Structure the summary around this central aspect and describe how it influences the rest
4. Emphasize potential causal relationships between different ABCD elements
5. If both adaptive and maladaptive states are present, describe the interplay between them
6. If only one self-state is present (either adaptive or maladaptive), summarize only that state
7. Naturally integrate ABCD elements into the description without explicitly labeling them

## Examples of Posts and Posts Summaries

{content}

## Post to Analyze
{post}

## Adaptive and Maladaptive states associated with post
{adaptive_maladaptive}

## Output Format
Generate a post-level summary that follows the guidelines above and the examples. Your summary should be a cohesive paragraph that naturally incorporates ABCD elements without explicitly labeling them.
"""
# BULUSI Task C prompt — VERBATIM from BULUSI-CLPsych/prompts/Timeline Summary.txt
# (placeholders {content}, {Results} filled at runtime).
BULUSI_TIMELINE_PROMPT = """# Timeline-Level Mental Health Self-States Summary

## Task Description
Your task is to generate a comprehensive summary of the interplay between adaptive and maladaptive self-states across a user's timeline of posts. You'll analyze how these self-states change over time and identify patterns of flexibility, rigidity, improvement, or deterioration.

## Definitions
- **Self-states** are identifiable units characterized by specific combinations of Affect, Behavior, Cognition, and Desire/Need (ABCD) that tend to be coactivated in a meaningful manner for limited periods of time.

- **Adaptive self-state**: Aspects of Affect, Behavior, and Cognition towards the self or others that are conducive to fulfilling basic desires/needs such as relatedness, autonomy, and competence.

- **Maladaptive self-state**: Aspects of Affect, Behavior, and Cognition towards the self or others that hinder the fulfillment of basic desires/needs.

## ABCD Framework Components
1. **Affect (A)**: The emotions expressed (e.g., happy, depressed, anxious, proud)
2. **Behavior (B)**: Actions toward self (BS) or others (BO) (e.g., self-care, self-harm, relating)
3. **Cognition (C)**: Perceptions of self (CS) or others (CO) (e.g., self-acceptance, criticism)
4. **Desire/Need (D)**: Main desires, expectations, or fears (e.g., relatedness, autonomy)

## Timeline Summary Guidelines
Your summary should:

1. **Emphasize temporal dynamics** across the timeline, focusing on:
   - **Flexibility**: How readily the person moves between adaptive and maladaptive states
   - **Rigidity**: Whether the person gets stuck in particular self-states
   - **Improvement**: Positive progression toward more adaptive states over time
   - **Deterioration**: Negative shifts toward more maladaptive states over time

2. **Describe self-state dominance patterns** by analyzing:
   - Whether adaptive or maladaptive states dominate overall
   - If there are periods where one state is completely absent
   - Key transition points where dominance shifts from one state to another
   - If self-states alternate or appear simultaneously in posts

3. **Identify ABCD element changes** that contribute to transitions:
   - How changes in specific affects (emotions) influence state shifts
   - How behavioral patterns evolve over time
   - How cognitive patterns (perceptions of self/others) transform
   - How desires/expectations change throughout the timeline

4. **Note catalysts for change**, if apparent:
   - External events or relationships that trigger state transitions
   - Internal realizations or shifts in perspective
   - Therapeutic interventions or coping strategies that affect states

## Example Timeline Summaries
{content}

## Timeline Data
The following represents the adaptive and maladaptive states identified in the user's posts over time:

{Results}

## Output Format
Generate a timeline-level summary that follows the guidelines above. Your summary should be a cohesive paragraph that naturally incorporates temporal dynamics and ABCD elements without explicitly labeling them. Focus on the overall trajectory and patterns of change over time.

Timeline Summary:
"""


# ── Retrieval models ─────────────────────────────────────────────────────────

class Contriever:
    """facebook/contriever with mean pooling (as in the contriever README)."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Contriever] Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, texts, batch_size: int = 32, **kwargs) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inputs = self.tokenizer(texts[i:i + batch_size], padding=True,
                                        truncation=True, max_length=512,
                                        return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        vecs = np.concatenate(out, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def load_retrievers(args):
    if args.backend == "mock":
        print("[Mock] Stand-in contriever + cross-encoder")
        return MockEmbedder(dim=768), MockCrossEncoder()
    from sentence_transformers.cross_encoder import CrossEncoder
    print(f"[Reranker] Loading {args.reranker_model}")
    return Contriever(args.contriever_model), CrossEncoder(args.reranker_model, max_length=512)


def two_stage(query_text, query_vec, query_tl, db_texts, db_vecs, db_tl_ids,
              reranker, n_candidates, n_examples):
    """contriever top-n_candidates (LOTO-masked) -> cross-encoder top-n_examples.
    Returns DB indices, best first."""
    sims = db_vecs @ query_vec
    sims[db_tl_ids == query_tl] = -np.inf                       # LOTO
    order = [i for i in np.argsort(sims)[::-1][:n_candidates] if np.isfinite(sims[i])]
    if not order:
        return []
    scores = reranker.predict([(query_text, db_texts[i]) for i in order])
    return [order[j] for j in np.argsort(scores)[::-1][:n_examples]]


def timeline_vectors(encoder, tls: list[Timeline]) -> np.ndarray:
    vecs = []
    for tl in tls:
        m = np.asarray(encoder.encode([p.text for p in tl.posts])).mean(axis=0)
        vecs.append(m / (np.linalg.norm(m) or 1.0))
    return np.asarray(vecs)


def prepare(args, timelines):
    contriever, reranker = load_retrievers(args)
    plans = {}
    if "b" in args.task:
        all_posts = [p for tl in timelines for p in tl.posts]
        db = [p for p in all_posts
              if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])]
        targets = [p for p in all_posts if p.summary]
        targets = targets[:args.limit] if args.limit else targets
        db_texts = [p.text for p in db]
        db_vecs = np.asarray(contriever.encode(db_texts))
        t_vecs = np.asarray(contriever.encode([p.text for p in targets]))
        db_tl = np.array([p.timeline_id for p in db])
        plans["b"] = [
            (p, [db[i] for i in two_stage(p.text, v, p.timeline_id, db_texts, db_vecs,
                                          db_tl, reranker, args.n_candidates,
                                          args.n_examples)])
            for p, v in zip(targets, t_vecs)]
        print(f"[Task B] retrieval DB {len(db)} posts | targets {len(targets)}")
    if "c" in args.task:
        db = [tl for tl in timelines if tl.summary and tl.posts]
        targets = task_c_targets(timelines, args.limit)
        db_texts = [timeline_text(tl) for tl in db]
        db_vecs = timeline_vectors(contriever, db)
        t_vecs = timeline_vectors(contriever, targets)
        db_tl = np.array([tl.timeline_id for tl in db])
        plans["c"] = [
            (tl, [db[i] for i in two_stage(timeline_text(tl), v, tl.timeline_id,
                                           db_texts, db_vecs, db_tl, reranker,
                                           args.n_candidates, args.n_examples)])
            for tl, v in zip(targets, t_vecs)]
        print(f"[Task C] retrieval DB {len(db)} timelines | targets {len(targets)}")
    del contriever, reranker
    free_gpu_memory()
    return plans


# ── Prompt construction ──────────────────────────────────────────────────────

def format_states_block(post: Post) -> str:
    """The '{adaptive/maladaptive}' slot: gold evidence spans of both polarities."""
    lines = []
    adaptive, maladaptive = span_texts(post, "adaptive"), span_texts(post, "maladaptive")
    if adaptive:
        lines.append("Adaptive:")
        lines += [f'- "{s}"' for s in adaptive]
    if maladaptive:
        lines.append("Maladaptive:")
        lines += [f'- "{s}"' for s in maladaptive]
    if not lines:
        lines.append("None identified")
    return "\n".join(lines)


def format_example(post: Post) -> str:
    """One retrieved example: post + gold states + gold summary."""
    return (
        f'Post:\n"{post.text}"\n\n'
        f"Adaptive and Maladaptive states associated with post:\n"
        f"{format_states_block(post)}\n\n"
        f'Summary:\n"{post.summary}"'
    )


def build_messages_b(target: Post, examples: list[Post]) -> list[dict]:
    content_block = "\n\n---\n\n".join(format_example(ex) for ex in examples)
    prompt = BULUSI_PROMPT.format(
        content=content_block if content_block else "(No examples available.)",
        post=target.text,
        adaptive_maladaptive=format_states_block(target),
    )
    return [{"role": "user", "content": prompt}]


def format_results(tl: Timeline, cap: int | None) -> str:
    blocks = []
    for i, p in enumerate(tl.posts, 1):
        head = f"Post {i}" + (f" ({p.date})" if p.date else "")
        blocks.append(f'{head}:\n"{truncate_words(p.text, cap)}"\n'
                      f"States:\n{format_states_block(p)}")
    return "\n\n".join(blocks)


def build_messages_c(target: Timeline, examples: list[Timeline],
                     cap: int | None) -> list[dict]:
    content = "\n\n".join(f"Example {i}:\n{ex.summary}"
                          for i, ex in enumerate(examples, 1))
    prompt = BULUSI_TIMELINE_PROMPT.format(
        content=content if content else "(No examples available.)",
        Results=format_results(target, cap))
    return [{"role": "user", "content": prompt}]


# ── Generation ───────────────────────────────────────────────────────────────

def run_task_b(args, timelines, plan, backend):
    checkpoint = Checkpoint(os.path.join(args.output_dir, "checkpoint_bulusi_taskb.jsonl"))
    subm_path = os.path.join(args.output_dir, "bulusi_taskb_submission.json")
    pending = [x for x in plan if not checkpoint.has(x[0].timeline_id, x[0].post_id)]
    print(f"[Task B] pending {len(pending)} / {len(plan)}")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for p, examples in batch:
            msgs, k_used = drop_until_fits(
                lambda k, p=p, examples=examples: build_messages_b(p, examples[:k]),
                backend, len(examples), args.max_model_len, args.max_tokens,
                min_examples=min(1, len(examples)))
            prompts.append(msgs)
            metas.append((p, examples[:k_used]))
        outputs = backend.generate(prompts)
        dt = time.perf_counter() - t0
        for (p, examples), summary in zip(metas, outputs):
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": summary,
                "n_examples": len(examples),
                "retrieved_post_ids": [e.post_id for e in examples],
                "retrieved_timelines": [e.timeline_id for e in examples],
                "gold_summary": p.summary,
                "generation_sec": round(dt / len(batch), 2),
            })
        write_task_b_submission(timelines, checkpoint, subm_path)
        print(f"[Task B] {min(start + args.batch_size, len(pending))}/{len(pending)} ({dt:.1f}s)")
    write_task_b_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())

    write_settings_log(
        os.path.join(args.output_dir, "bulusi_taskb_settings.md"),
        "BULUSI Task B — generation settings",
        settings=[
            ("generation model", args.model, "paper (best Task B submission, qwen-2.5-72b)"),
            *common_settings(args),
            ("temperature", args.temperature, "our default — paper silent"),
            ("n_examples", args.n_examples, "paper (5 most similar posts)"),
            ("retrieval stage 1", f"{args.contriever_model} top-{args.n_candidates}",
             "paper (contriever, top-50)"),
            ("retrieval stage 2", f"{args.reranker_model} -> top-{args.n_examples}",
             "paper (ms-marco-MiniLM-L-6-v2 re-rank)"),
            ("prompt", "BULUSI-CLPsych/prompts/Summary.txt (verbatim)", "team repo"),
            ("evidence input", "gold adaptive/maladaptive spans",
             "project convention (original: BULUSI's G-RAG predictions)"),
            ("example block", "post + gold states + gold summary",
             "our reconstruction (paper Fig. 1 'Examples w/ states'; prompt "
             "section 'Examples of Posts and Posts Summaries')"),
            ("leave-one-timeline-out", True, "project convention"),
            ("posts with fewer than n_examples (context guard)",
             sum(1 for r in recs if r.get("n_examples", 0) < args.n_examples), "reported"),
        ],
        notes=[
            "Gold evidence replaces BULUSI's predicted G-RAG evidence, as for "
            "every system in this study.",
            "The mixtral-8x22b and llama-3.1-405b variants are not reimplemented.",
        ],
    )


def run_task_c(args, timelines, plan, backend):
    checkpoint = Checkpoint(os.path.join(args.output_dir, "checkpoint_bulusi_taskc.jsonl"))
    subm_path = os.path.join(args.output_dir, "bulusi_taskc_submission.json")
    pending = [x for x in plan if not checkpoint.has(x[0].timeline_id, "__timeline__")]
    print(f"[Task C] pending {len(pending)} / {len(plan)}")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for tl, examples in batch:
            msgs, k_used, cap = fit_timeline_prompt(
                lambda k, cap, tl=tl, examples=examples:
                    build_messages_c(tl, examples[:k], cap),
                backend, len(examples), args.max_model_len, args.max_tokens,
                min_examples=0)
            prompts.append(msgs)
            metas.append((tl, examples[:k_used], cap))
        outputs = backend.generate(prompts)
        dt = time.perf_counter() - t0
        for (tl, examples, cap), summary in zip(metas, outputs):
            # The prompt ends with "Timeline Summary:"; strip it if echoed.
            if summary.lower().startswith("timeline summary:"):
                summary = summary[len("timeline summary:"):].strip()
            checkpoint.append(tl.timeline_id, "__timeline__", {
                "summary": summary,
                "n_examples": len(examples),
                "post_word_cap": cap,
                "retrieved_timelines": [e.timeline_id for e in examples],
                "gold_summary": tl.summary,
                "generation_sec": round(dt / len(batch), 2),
            })
        write_task_c_submission(timelines, checkpoint, subm_path)
        print(f"[Task C] {min(start + args.batch_size, len(pending))}/{len(pending)} ({dt:.1f}s)")
    write_task_c_submission(timelines, checkpoint, subm_path)
    recs = list(checkpoint.records.values())

    write_settings_log(
        os.path.join(args.output_dir, "bulusi_taskc_settings.md"),
        "BULUSI Task C — generation settings",
        settings=[
            ("generation model", args.model,
             "project decision (72B for both tasks; paper reports llama-3.1-405b "
             "doing well on Task C)"),
            *common_settings(args),
            ("temperature", args.temperature, "our default — paper silent"),
            ("prompt", "BULUSI-CLPsych/prompts/Timeline Summary.txt (verbatim)", "team repo"),
            ("{Results}", "every post (text + gold states), chronological",
             "paper Fig. 1 (posts across the timeline + evidence)"),
            ("{content}", f"gold summaries of {args.n_examples} retrieved training timelines",
             "prompt heading 'Example Timeline Summaries'; count follows the "
             "paper's 5 examples"),
            ("timeline retrieval",
             f"{args.contriever_model} (mean of post embeddings) top-{args.n_candidates} "
             f"-> {args.reranker_model} top-{args.n_examples}",
             "our extension of the paper's post-level retrieval to timelines"),
            ("evidence input", "gold adaptive/maladaptive spans", "project convention"),
            ("leave-one-timeline-out", True, "project convention"),
            ("timelines with fewer than n_examples (context guard)",
             sum(1 for r in recs if r.get("n_examples", 0) < args.n_examples), "reported"),
            ("timelines with truncated posts (context guard)",
             sum(1 for r in recs if r.get("post_word_cap")), "reported"),
        ],
        notes=[
            "The paper describes retrieval only for posts; retrieving whole "
            "timelines with the same two-stage pipeline is our extension.",
            "The cross-encoder sees only the first 512 tokens of each "
            "(target, candidate) concatenated-timeline pair.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="BULUSI (CLPsych 2025) Task B/C reimplementation.")
    add_common_args(parser, DEFAULT_MODEL, "./outputs/bulusi")
    parser.add_argument("--n_examples", type=int, default=DEFAULT_N_EXAMPLES)
    parser.add_argument("--n_candidates", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--contriever_model", default=DEFAULT_CONTRIEVER)
    parser.add_argument("--reranker_model", default=DEFAULT_RERANKER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    plans = prepare(args, timelines)
    backend = backend_from_args(args)
    if "b" in plans:
        run_task_b(args, timelines, plans["b"], backend)
    if "c" in plans:
        run_task_c(args, timelines, plans["c"], backend)
    print(f"\nBULUSI run complete at {datetime.datetime.now():%Y%m%d-%H%M%S}.")


if __name__ == "__main__":
    main()
