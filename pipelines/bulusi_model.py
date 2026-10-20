"""BULUSI (CLPsych 2025) Task B system — reimplementation for the critique paper.

Reference:
    Ravenda, Kara-Isitt, Swift, Mira, Raballo. "From Evidence Mining to
    Meta-Prediction: a Gradient of Methodologies for Task-Specific Challenges
    in Psychological Assessment." CLPsych 2025, pp. 242-248.
    Team code: https://github.com/Fede-stack/BULUSI-CLPsych (cloned locally at
    ./BULUSI-CLPsych, untracked).

What this reproduces (paper §2.2 + Appendix A, Task B)
------------------------------------------------------
* Few-shot summarizer LLM. The paper's best Task B submission used
  `qwen-2.5-72b`; the other variants (mixtral-8x22b, llama-3.1-405b) are
  not reimplemented (per project decision, 72B only).
* In-context examples: "we identified the 5 most similar posts from the
  training set" — retrieval via `contriever` (top-50) re-ranked by
  `ms-marco-MiniLM-L-6-v2` (cross-encoder) down to the top 5.
* Prompt: the team's actual Task B prompt, verbatim from
  BULUSI-CLPsych/prompts/Summary.txt (placeholders {content}, {post},
  {adaptive/maladaptive}).

Project-convention adaptations (logged in the run's settings file):
* GOLD evidence spans feed the "{adaptive/maladaptive}" slot (the original
  used the team's own G-RAG predicted evidence).
* Leave-one-timeline-out: the 5 retrieved examples never come from the
  target post's own timeline.
* Decoding: the paper reports no temperature/top-p/max-tokens; we use
  temperature 0.1, top_p 1.0, max_tokens 600 (documented defaults).
* The example block includes each example post's gold evidence states and
  gold summary (paper Fig. 1: "Examples w/ states"; the prompt section is
  headed "Examples of Posts and Posts Summaries").

Run on the cluster (vLLM, local weights, tensor parallelism):
    python pipelines/bulusi_model.py --data_dir <timeline json dir> \
        --model /path/to/Qwen2.5-72B-Instruct --backend vllm

Outputs (to --output_dir):
    checkpoint_bulusi_taskb.jsonl   resumable per-post results (skip on restart)
    bulusi_taskb_submission.json    official submission format (dev.json-like)
    bulusi_taskb_settings.md        generation-settings log
"""

import argparse
import datetime
import os
import time

import numpy as np

from system_utils import (
    Post, Timeline, load_timelines, span_texts,
    make_backend, Checkpoint, write_task_b_submission, write_settings_log,
    drop_until_fits,
)

# Paper defaults (Appendix A): contriever retrieval, ms-marco-MiniLM-L-6-v2
# re-ranking, 5 in-context examples (from a top-50 first stage).
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_CONTRIEVER = "facebook/contriever"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_N_EXAMPLES = 5          # paper: 5 most similar posts
DEFAULT_N_CANDIDATES = 50       # paper: contriever top-50 first stage

# BULUSI Task B prompt — VERBATIM from BULUSI-CLPsych/prompts/Summary.txt
# (placeholders {content}, {post}, {adaptive/maladaptive} are filled at runtime).
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


# ── Contriever embedding (facebook/contriever, mean pooling) ─────────────────

def embed_contriever(model_name: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[Contriever] Embedding {len(texts)} texts with {model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state          # (B, T, H)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling
            out.append(mean.cpu().numpy())
    vecs = np.concatenate(out, axis=0)
    # L2-normalize so cosine similarity == dot product.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


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


def build_messages(target: Post, examples: list[Post]) -> list[dict]:
    content_block = "\n\n---\n\n".join(format_example(ex) for ex in examples)
    prompt = BULUSI_PROMPT.format(
        content=content_block if content_block else "(No examples available.)",
        post=target.text,
        adaptive_maladaptive=format_states_block(target),
    )
    return [{"role": "user", "content": prompt}]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BULUSI (CLPsych 2025) Task B system reimplementation.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory of shared-task timeline JSON files.")
    parser.add_argument("--output_dir", default="./outputs/bulusi",
                        help="Where to write checkpoint / submission / settings log.")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "ollama", "mock"],
                        help="Generation backend (cluster: vllm; local tests: ollama/mock).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model path (local weights) or id. Default: {DEFAULT_MODEL} "
                             "(paper's best Task B submission: qwen-2.5-72b).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None,
                         help="Optional vLLM quantization (e.g. 'fp8', 'awq') if the "
                              "cluster GPUs require a quantized 72B.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM context length (None = model default).")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Paper reports no decoding settings; 0.1 is our documented default.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_examples", type=int, default=DEFAULT_N_EXAMPLES,
                        help="In-context examples (paper: 5).")
    parser.add_argument("--n_candidates", type=int, default=DEFAULT_N_CANDIDATES,
                        help="Contriever first-stage candidates (paper: 50).")
    parser.add_argument("--contriever_model", default=DEFAULT_CONTRIEVER)
    parser.add_argument("--reranker_model", default=DEFAULT_RERANKER)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Posts per generation batch (checkpoint saved after each).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N target posts (smoke testing).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timelines = load_timelines(args.data_dir)
    all_posts = [p for tl in timelines for p in tl.posts]
    # Vector DB: training posts that carry both gold evidence and a gold summary.
    db_posts = [p for p in all_posts
                if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])]
    targets = [p for p in all_posts if p.summary]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Retrieval DB: {len(db_posts)} posts | targets: {len(targets)}")

    backend = make_backend(
        args.backend, args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype, quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, seed=args.seed,
    )

    # ── Retrieval: contriever top-50 → cross-encoder rerank → top-5, LOTO ──
    # Embed every DB post and every target post's LOTO candidate pool once.
    # (Pools differ per target only by the LOTO exclusion; embedding all DB
    # posts once and masking at query time is equivalent and much faster.)
    db_vecs = embed_contriever(args.contriever_model, [p.text for p in db_posts])
    target_vecs = embed_contriever(args.contriever_model, [p.text for p in targets])
    db_timeline_ids = np.array([p.timeline_id for p in db_posts])

    from sentence_transformers.cross_encoder import CrossEncoder
    print(f"[Reranker] Loading {args.reranker_model} ...")
    reranker = CrossEncoder(args.reranker_model, max_length=512)

    def retrieve_examples(target: Post, target_vec) -> list[Post]:
        same_tl = db_timeline_ids == target.timeline_id
        sims = db_vecs @ target_vec
        sims[same_tl] = -np.inf                       # LOTO
        cand_idx = np.argsort(sims)[::-1][: args.n_candidates]
        candidates = [db_posts[i] for i in cand_idx]
        pairs = [(target.text, c.text) for c in candidates]
        ce_scores = reranker.predict(pairs)
        top = np.argsort(ce_scores)[::-1][: args.n_examples]
        return [candidates[i] for i in top]

    # ── Generation loop (batched, resumable) ────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, "checkpoint_bulusi_taskb.jsonl")
    checkpoint = Checkpoint(ckpt_path)
    subm_path = os.path.join(args.output_dir, "bulusi_taskb_submission.json")

    pending = [p for p in targets
               if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"Pending targets: {len(pending)} / {len(targets)} (checkpoint resumed)")

    vec_by_post = {id(p): tv for p, tv in zip(targets, target_vecs)}

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        t0 = time.perf_counter()
        prompts, metas = [], []
        for p in batch:
            examples = retrieve_examples(p, vec_by_post[id(p)])
            msgs, k_used = drop_until_fits(
                lambda k: build_messages(p, examples[:k]), backend,
                len(examples), args.max_model_len, args.max_tokens, min_examples=1)
            prompts.append(msgs)
            metas.append((p, examples[:k_used], k_used))
        summaries = backend.generate(prompts)
        for (p, examples, k_used), summary in zip(metas, summaries):
            checkpoint.append(p.timeline_id, p.post_id, {
                "summary": summary,
                "n_examples": k_used,
                "retrieved_post_ids": [e.post_id for e in examples],
                "retrieved_timelines": [e.timeline_id for e in examples],
                "gold_summary": p.summary,
                "generation_sec": round(time.perf_counter() - t0, 1) / len(batch),
            })
            print(f"  [{p.timeline_id}#{p.post_index}] {summary[:80]}...")
        # Rewrite the submission after every batch so crashed runs stay usable.
        write_task_b_submission(timelines, checkpoint, subm_path)
        done = min(start + args.batch_size, len(pending))
        print(f"Batch {done}/{len(pending)} done "
              f"({time.perf_counter() - t0:.1f}s) — checkpoint + submission saved")

    write_task_b_submission(timelines, checkpoint, subm_path)

    write_settings_log(
        os.path.join(args.output_dir, "bulusi_taskb_settings.md"),
        "BULUSI Task B — generation settings",
        settings=[
            ("generation model", args.model,
             "paper (best Task B submission, qwen-2.5-72b)"),
            ("backend", args.backend, "our setup (vLLM, local weights)"),
            ("tensor_parallel_size", args.tensor_parallel_size, "our setup"),
            ("temperature", args.temperature,
             "our default — paper silent on decoding settings"),
            ("top_p", args.top_p, "our default — paper silent"),
            ("max_tokens", args.max_tokens, "our default — paper silent"),
            ("n_examples", args.n_examples, "paper (5 most similar posts)"),
            ("retrieval stage 1", f"{args.contriever_model} top-{args.n_candidates}",
             "paper (contriever, top-50)"),
            ("retrieval stage 2", f"{args.reranker_model} -> top-{args.n_examples}",
             "paper (ms-marco-MiniLM-L-6-v2 re-rank)"),
            ("prompt", "BULUSI-CLPsych/prompts/Summary.txt (verbatim)",
             "team repo"),
            ("evidence input", "gold adaptive/maladaptive spans",
             "project convention (original used BULUSI's predicted G-RAG evidence)"),
            ("example block", "post + gold states + gold summary",
             "our reconstruction (paper Fig. 1: 'Examples w/ states'; prompt section: "
             "'Examples of Posts and Posts Summaries')"),
            ("leave-one-timeline-out", True, "project convention"),
        ],
        notes=[
            "Original system used its own G-RAG predicted evidence spans; this "
            "reimplementation feeds the gold evidence spans instead, consistent "
            "with the other systems in this repo.",
            "Original examples: 5 most similar training posts (contriever top-50 "
            "re-ranked by ms-marco-MiniLM-L-6-v2); leave-one-timeline-out enforced.",
            "The paper's other Task B submission variants (mixtral-8x22b, "
            "llama-3.1-405b) are not reimplemented (project decision).",
        ],
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\nBULUSI Task B run complete at {ts}. Submission: {subm_path}")


if __name__ == "__main__":
    main()
