"""
EAIonFlux (CLPsych 2025, 2nd place) — In-Context Learning reproduction for Tasks B and C.

Reference:
    Anson Antony and Annika M. Schoene. "Retrieval-Enhanced Mental Health
    Assessment: Capturing Self-State Dynamics from Social Media Using In-Context
    Learning." CLPsych 2025. https://aclanthology.org/2025.clpsych-1.23.pdf

What this reproduces
--------------------
The EAIonFlux system is a retrieval-augmented In-Context Learning (ICL) pipeline:

    * A POST-LEVEL vector database (one entry per annotated post) and a
      TIMELINE-LEVEL vector database (one entry per timeline), both embedded with
      a sentence embedding model and queried by cosine similarity.
    * For each target, the top-k (k=5 or k=10 in the paper) most similar examples
      are retrieved and formatted as demonstrations, prioritizing examples that
      have gold summaries.
    * Task B (post-level) and Task C (timeline-level) prompts follow the paper's
      Appendix A.2 / A.3 templates verbatim, augmented with the retrieved
      demonstrations, and are sent to a locally-served LLM via Ollama.

Faithful-but-configurable choices (see the project README):
    * Generation model defaults to the paper's `llama3.3:70b`. With
      --backend ollama (local single-GPU smoke runs) any smaller Ollama model
      can be substituted via --model (e.g. llama3.1:8b); on the cluster use
      --backend vllm --model /path/to/local/Llama-3.3-70B-Instruct, which
      loads local weights with tensor parallelism across GPUs.
    * The embedding model is any sentence-transformers id chosen with
      --embedding_model. The paper used `Linq-AI-Research/Linq-Embed-Mistral`;
      the default here is the lighter `sentence-transformers/all-mpnet-base-v2`
      already used elsewhere in this repo. Pass the Linq id to match the paper.
    * Retrieval is exact top-k cosine similarity. The paper used ChromaDB/HNSW
      approximate nearest-neighbour search; at this dataset's scale exact cosine
      returns the same neighbours.

Resumability: per-post (Task B) / per-timeline (Task C) results are appended to
a JSONL checkpoint after every generation and skipped on restart. Official
submission-format JSONs (gold evidence + gold well-being filled in, dev.json
structure) are rewritten after every checkpoint write, so partial runs stay
scorable and reruns cost only the missing generations.

Like the rest of this repository, Tasks A.1/A.2 are NOT performed: the gold
evidence spans and gold well-being scores are used as the "Task A outputs" that
feed Tasks B and C. This keeps the EAIonFlux reproduction directly comparable to
the other Task B / Task C systems scored here.

IMPORTANT: Run locally. Ollama and sentence-transformers both run on-device. Do
NOT point --model or --embedding_model at a hosted API: the data sharing
agreement prohibits sending the data to third-party LLM providers.

Outputs (written to --output_dir) use the project's existing task_b_results /
task_c schema (see EXAMPLE_DATA.json), so they are drop-in for the existing
tooling (edit_output_for_official_eval.py, make_degenerate_baselines.py,
specificity_analysis.py):
    Task B: task_b_eaionflux_k{K}_{ts}.json — a JSON list of post entries with
            `one_shot_summary` (the k-shot ICL summary) and `zero_shot_summary`
            (no-retrieval ablation, or null with --skip_zero_shot). Convert with
            edit_output_for_official_eval.py --task B --model_name one_shot_summary.
    Task C: task_c_eaionflux_k{K}_{input_type}_{ts}.json — a JSON list of
            {timeline_id, n_posts, input_type, prompt_style, gold_summary,
            predicted_summary, generation_time_sec} entries. Convert with --task C.

Usage:
    python eaionflux_model.py --data_dir <dir of timeline json> --task both --k 5
    python eaionflux_model.py --data_dir <dir> --task C --k 10 --model llama3.3:70b \
        --embedding_model Linq-AI-Research/Linq-Embed-Mistral
"""

import argparse
import datetime
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from system_utils import (
    Checkpoint, write_task_b_submission, write_task_c_submission,
    write_settings_log, VLLMBackend, MockBackend,
)

# ── Configuration ────────────────────────────────────────────────────────────

ABCD_KEYS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
STATE_TYPES = ["adaptive-state", "maladaptive-state"]

# Full ABCD aspect names, used when formatting evidence for the prompts.
ABCD_NAMES = {
    "A": "Affect",
    "B-O": "Behavior-Other",
    "B-S": "Behavior-Self",
    "C-O": "Cognition-Other",
    "C-S": "Cognition-Self",
    "D": "Desire/Need",
}

# Paper defaults: generation = llama3.3:70b, embedding = Linq-Embed-Mistral.
DEFAULT_OLLAMA_MODEL = "llama3.3:70b"
DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_MODEL = "Linq-AI-Research/Linq-Embed-Mistral"

SYSTEM_MESSAGE = "You are an expert in clinical psychology analyzing social media posts."


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Post:
    timeline_id: str
    post_index: int
    post_id: str
    date: str
    text: str
    wellbeing: float | None
    summary: str                       # gold post-level summary ("" if none)
    # evidence: {"adaptive": [(abcd_key, category, span_text), ...],
    #            "maladaptive": [...]}
    evidence: dict = field(default_factory=lambda: {"adaptive": [], "maladaptive": []})
    embedding: np.ndarray = None

    @property
    def is_annotated(self) -> bool:
        return bool(self.summary) or self.wellbeing is not None

    def dominant_state(self) -> str:
        """Which self-state dominates this post, by evidence-span count."""
        n_adapt = len(self.evidence["adaptive"])
        n_malad = len(self.evidence["maladaptive"])
        if n_adapt == 0 and n_malad == 0:
            return "neutral"
        if n_adapt > n_malad:
            return "adaptive"
        if n_malad > n_adapt:
            return "maladaptive"
        return "mixed"


@dataclass
class Timeline:
    timeline_id: str
    summary: str                       # gold timeline-level summary ("" if none)
    posts: list = field(default_factory=list)
    representation: str = ""           # structured text used for embedding
    characteristics: dict = field(default_factory=dict)
    embedding: np.ndarray = None


# ── Data loading (reads the shared-task source JSON files) ────────────────────

def _parse_evidence(evidence_obj) -> dict:
    """Pull (abcd_key, category, span_text) tuples out of a post's gold evidence."""
    out = {"adaptive": [], "maladaptive": []}
    if not isinstance(evidence_obj, dict):
        return out
    for state_type in STATE_TYPES:
        polarity = "adaptive" if state_type == "adaptive-state" else "maladaptive"
        state_data = evidence_obj.get(state_type) or {}
        if not isinstance(state_data, dict):
            continue
        for abcd_key in ABCD_KEYS:
            span_info = state_data.get(abcd_key)
            if not isinstance(span_info, dict):
                continue
            span_text = (span_info.get("highlighted_evidence") or "").strip()
            category = (span_info.get("Category") or "").strip()
            if span_text:
                out[polarity].append((abcd_key, category, span_text))
    return out


def load_timelines(data_dir: str) -> list[Timeline]:
    """
    Load timeline JSON files. Each file is one timeline with the shared-task
    schema: {"timeline_id", "timeline_summary", "posts": [...]}; each post has
    "post_index", "post_id", "date", "post", "Post Summary", "Well-being",
    and "evidence" (adaptive-state / maladaptive-state -> {ABCD: {...}}).
    """
    data_path = Path(data_dir)
    if data_path.is_file() and data_path.suffix == ".json":
        files = [data_path]
    else:
        files = sorted(data_path.glob("*.json"))

    timelines = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tl in (data if isinstance(data, list) else [data]):
            tl_id = tl.get("timeline_id", fpath.stem)
            posts = []
            for pd in tl.get("posts", []):
                wb = pd.get("Well-being", None)
                try:
                    wb_val = float(str(wb).strip())
                except (ValueError, TypeError):
                    wb_val = None
                posts.append(Post(
                    timeline_id=tl_id,
                    post_index=pd.get("post_index", -1),
                    post_id=pd.get("post_id", ""),
                    date=pd.get("date", "") or "",
                    text=pd.get("post", "") or "",
                    wellbeing=wb_val,
                    summary=(pd.get("Post Summary") or "").strip(),
                    evidence=_parse_evidence(pd.get("evidence")),
                ))
            posts.sort(key=lambda p: p.post_index)
            timelines.append(Timeline(
                timeline_id=tl_id,
                summary=(tl.get("timeline_summary") or "").strip(),
                posts=posts,
            ))

    n_posts = sum(len(t.posts) for t in timelines)
    print(f"Loaded {len(timelines)} timelines ({n_posts} posts) from {len(files)} file(s)")
    return timelines


# ── Timeline structured representation (paper §4, Task C Implementation) ───────

def _parse_date(s: str):
    """Best-effort multi-format date parsing; returns a datetime or None."""
    s = (s or "").strip()
    if not s:
        return None
    # Try ISO / common formats first, then fall back to dateutil if present.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
                "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M"):
        try:
            return datetime.datetime.strptime(s[:len(fmt) + 2].strip(), fmt)
        except ValueError:
            continue
    try:
        from dateutil import parser as _dup  # optional dependency
        return _dup.parse(s)
    except Exception:
        return None


def _wellbeing_trend(wellbeings: list[float]) -> str:
    """Classify a well-being series as improving / declining / fluctuating / stable."""
    vals = [w for w in wellbeings if w is not None]
    if len(vals) < 2:
        return "stable"
    xs = np.arange(len(vals), dtype=float)
    slope = np.polyfit(xs, vals, 1)[0]
    spread = max(vals) - min(vals)
    # High variability relative to the range -> fluctuating.
    if spread >= 3 and np.std(vals) >= 1.5 and abs(slope) < 0.3:
        return "fluctuating"
    if slope > 0.3:
        return "improving"
    if slope < -0.3:
        return "declining"
    return "stable"


def _self_state_pattern(posts: list[Post]) -> str:
    """Classify a timeline as Predominantly Adaptive/Maladaptive, Mixed, or Balanced."""
    counts = {"adaptive": 0, "maladaptive": 0, "mixed": 0, "neutral": 0}
    for p in posts:
        counts[p.dominant_state()] += 1
    n_adapt, n_malad, n_mixed = counts["adaptive"], counts["maladaptive"], counts["mixed"]
    total = max(1, n_adapt + n_malad + n_mixed)
    if n_mixed >= 0.5 * total:
        return "Mixed"
    if n_adapt > n_malad and n_adapt >= 0.6 * total:
        return "Predominantly Adaptive"
    if n_malad > n_adapt and n_malad >= 0.6 * total:
        return "Predominantly Maladaptive"
    return "Balanced"


def build_timeline_representation(tl: Timeline) -> tuple[str, dict]:
    """
    Build the structured timeline representation described in the paper:
    timeline metrics (duration, posting frequency), well-being statistics
    (average, range, trend), self-state pattern, and a chronological mapping of
    posts to dominant self-states. Returns (representation_text, characteristics).
    """
    posts = tl.posts
    dates = [d for d in (_parse_date(p.date) for p in posts) if d is not None]
    if len(dates) >= 2:
        span_days = (max(dates) - min(dates)).days
        duration_text = f"{span_days} days"
        weeks = max(span_days / 7.0, 1e-9)
        freq_text = f"{len(posts) / weeks:.2f} posts/week"
        date_lo = min(dates).strftime("%Y-%m-%d")
        date_hi = max(dates).strftime("%Y-%m-%d")
    else:
        span_days = None
        duration_text = "unknown"
        freq_text = "unknown"
        date_lo = posts[0].date if posts else "unknown"
        date_hi = posts[-1].date if posts else "unknown"

    wbs = [p.wellbeing for p in posts if p.wellbeing is not None]
    if wbs:
        avg_wb = f"{np.mean(wbs):.1f}"
        min_wb, max_wb = f"{min(wbs):.0f}", f"{max(wbs):.0f}"
    else:
        avg_wb, min_wb, max_wb = "unknown", "unknown", "unknown"
    trend = _wellbeing_trend([p.wellbeing for p in posts])
    pattern = _self_state_pattern(posts)

    chrono_lines = []
    for p in posts:
        chrono_lines.append(f"- [{p.date or 'no date'}] self-state: {p.dominant_state()}, "
                            f"well-being: {p.wellbeing if p.wellbeing is not None else 'NA'}")
    chrono = "\n".join(chrono_lines)

    characteristics = {
        "duration_text": duration_text,
        "post_count": len(posts),
        "avg_well_being": avg_wb,
        "min_well_being": min_wb,
        "max_well_being": max_wb,
        "trend": trend,
        "state_dynamics": pattern,
        "date_range": f"{date_lo} to {date_hi}",
    }

    representation = (
        f"Timeline ID: {tl.timeline_id}\n"
        f"Time span: {duration_text} ({date_lo} to {date_hi})\n"
        f"Post count: {len(posts)}\n"
        f"Posting frequency: {freq_text}\n"
        f"Average well-being: {avg_wb}\n"
        f"Well-being range: {min_wb} to {max_wb}\n"
        f"Well-being trend: {trend}\n"
        f"Self-state pattern: {pattern}\n"
        f"Chronological self-states:\n{chrono}"
    )
    return representation, characteristics


# ── Embedding + retrieval ─────────────────────────────────────────────────────

def embed_texts(model_name: str, texts: list[str]) -> np.ndarray:
    """Embed a list of texts with a sentence-transformers model (L2-normalized)."""
    from sentence_transformers import SentenceTransformer
    print(f"Embedding {len(texts)} texts with {model_name} ...")
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def top_k_similar(query_vec, candidates: list, k: int) -> list[tuple]:
    """Return [(candidate, similarity), ...] for the k most cosine-similar candidates."""
    if not candidates:
        return []
    M = np.array([c.embedding for c in candidates])
    sims = cosine_similarity(query_vec.reshape(1, -1), M)[0]
    order = np.argsort(sims)[::-1][:k]
    return [(candidates[i], float(sims[i])) for i in order]


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_evidence_block(post: Post) -> tuple[str, str]:
    """Return (adaptive_text, maladaptive_text) formatted for a Task B prompt."""
    def fmt(spans):
        if not spans:
            return "None identified"
        return "\n".join(
            f'{key} ({ABCD_NAMES.get(key, key)}): "{text}" ({category})'
            if category else f'{key} ({ABCD_NAMES.get(key, key)}): "{text}"'
            for key, category, text in spans
        )
    return fmt(post.evidence["adaptive"]), fmt(post.evidence["maladaptive"])


def format_task_b_example(post: Post) -> str:
    adaptive, maladaptive = format_evidence_block(post)
    return (
        f'Post: "{post.text}"\n'
        f"Evidence:\n"
        f"- Adaptive:\n{adaptive}\n"
        f"- Maladaptive:\n{maladaptive}\n"
        f"Summary:\n"
        f'"{post.summary}"'
    )


def build_task_b_prompt(target: Post, examples: list[Post]) -> str:
    """Task B prompt following the paper's Appendix A.2 template."""
    example_block = "\n\n".join(
        f"Example {i + 1}:\n{format_task_b_example(ex)}"
        for i, ex in enumerate(examples)
    ) or "(No similar annotated examples were available.)"

    adaptive, maladaptive = format_evidence_block(target)

    return f"""You are analyzing a social media post for the CLPsych 2025 shared task, focusing on Task B - Post-level summary of self-state's inner dynamics.

Your task is to generate a summary of the interplay between adaptive and maladaptive self-states identified in the post. You need to:
1. Determine which self-state is dominant (adaptive/maladaptive) and describe it first.
2. For each self-state, identify the central organizing aspect (A, B, C, or D) that drives the state.
3. Structure the summary around this central aspect, describing how it influences the rest.
4. Emphasize potential causal relationships between the aspects.
5. Then, proceed to the second self-state and follow the same approach.
6. If the post contains only one self-state, summarize only that state.

## Self-State Definitions
Self-states constitute identifiable units characterized by specific combinations of Affect, Behavior, Cognition, and Desire/Need (ABCD) that tend to be coactivated in a meaningful manner for limited periods of time.
- An adaptive self-state pertains to aspects of Affect, Behavior, and Cognition towards the self or others, which is conducive to the fulfillment of basic desires/needs (D).
- A maladaptive self-state pertains to aspects of Affect, Behavior, and Cognition towards the self or others, that hinder the fulfillment of basic desires/needs (D).

## Similar Examples for Reference
Here are some examples of similar posts with their evidence and summaries:
{example_block}

Please analyze the target post following a similar approach to these examples, but make your own assessment based on the specific content.

## Post to Analyze
Here is the post: "{target.text}"

## Evidence Identified in the Post
Adaptive evidence:
{adaptive}
Maladaptive evidence:
{maladaptive}

## Response Instructions
Write a cohesive paragraph summary (200-300 words) that:
1. First describes the dominant self-state (whichever has more significant evidence).
2. Identifies which ABCD aspect (Affect, Behavior-Self, Behavior-Other, Cognition-Self, Cognition-Other, or Desire/Expectation) is central to each self-state.
3. Explains how this central aspect influences other aspects, focusing on causal relationships.
4. Naturally integrates ABCD elements into the description without explicitly highlighting them.
5. Uses clinical language appropriate for psychological assessment.

Do not use bulleted lists or headers in your summary. Write in a fluid, paragraph style."""


def format_task_c_example(tl: Timeline) -> str:
    c = tl.characteristics
    return (
        f"Timeline Characteristics:\n"
        f"ID: {tl.timeline_id}\n"
        f"Time span: {c.get('duration_text', 'unknown')}\n"
        f"Post count: {c.get('post_count', 'unknown')}\n"
        f"Average well-being: {c.get('avg_well_being', 'unknown')}\n"
        f"Well-being range: {c.get('min_well_being', 'unknown')} to {c.get('max_well_being', 'unknown')}\n"
        f"Well-being trend: {c.get('trend', 'unknown')}\n"
        f"Self-state pattern: {c.get('state_dynamics', 'unknown')}\n"
        f"Timeline Summary:\n"
        f'"{tl.summary}"'
    )


def build_task_c_prompt(
    target: Timeline,
    examples: list[Timeline],
    post_summaries: dict | None = None,
) -> str:
    """Task C prompt following the paper's Appendix A.3 template (verbatim text)."""
    # The paper's template lists only posts WITH evidence spans.
    posts_with_evidence = [p for p in target.posts
                           if p.evidence["adaptive"] or p.evidence["maladaptive"]]
    listing_posts = posts_with_evidence if posts_with_evidence else target.posts
    n_posts = len(listing_posts)
    first_date = listing_posts[0].date if listing_posts else "unknown"
    last_date = listing_posts[-1].date if listing_posts else "unknown"

    chrono_lines = []
    for p in listing_posts:
        adaptive, maladaptive = format_evidence_block(p)
        wb = p.wellbeing if p.wellbeing is not None else "NA"
        chrono_lines.append(
            f'[{p.date or "no date"}] (Well-being: {wb})\n'
            f'Post: "{p.text}"\n'
            f"Adaptive evidence: {adaptive}\n"
            f"Maladaptive evidence: {maladaptive}"
        )
    chronological_post_listing = "\n\n".join(chrono_lines)

    # Post-level summaries (cascaded Task B outputs, or gold, depending on caller).
    ps_map = post_summaries or {}
    ps_lines = []
    for p in listing_posts:
        s = ps_map.get(p.post_id) or ps_map.get(str(p.post_index))
        if s:
            ps_lines.append(f'[{p.date or "no date"}] "{s}"')
    post_level_summaries = "\n".join(ps_lines) if ps_lines else "(No post-level summaries available.)"

    example_block = "\n\n".join(
        f"Example {i + 1} (Timeline ID: {ex.timeline_id}):\n{format_task_c_example(ex)}"
        for i, ex in enumerate(examples)
    ) or "(No similar annotated timelines were available.)"

    return f"""You are analyzing a social media timeline for the CLPsych 2025 shared task, focusing on Task C Timeline-level summary of self-state's dynamics.

Your task is to generate a summary focusing on the interplay between adaptive and maladaptive self-states along the timeline. You need to:

1. Emphasize temporal dynamics focusing on concepts such as flexibility, rigidity, improvement, and deterioration.
2. Describe the extent to which the dominance of the self-states changes over time.
3. Explain how changes in aspects (Affect, Behavior, Cognition, and Desire) contribute to these transitions.

## Timeline to Analyze

This timeline contains {n_posts} posts spanning from {first_date} to {last_date}.

Here are the posts with their Well-being scores and evidence:

{chronological_post_listing}

## Post-level Summaries (if available)

{post_level_summaries}

## Similar Timelines for Reference

Here are some examples of similar timelines with their summaries:

{example_block}

Please analyze the target timeline following a similar approach to these examples, but make your own assessment based on the specific content.

## Response Instructions

Write a cohesive paragraph summary (200-400 words) that:
1. Describes the overall pattern of self-states across the timeline (e.g., predominantly adaptive, predominantly maladaptive, fluctuating).
2. Identifies any shifts or transitions between dominance of adaptive and maladaptive states.
3. Analyzes temporal dynamics using concepts like flexibility, rigidity, improvement, and deterioration.
4. Explains how changes in ABCD aspects (Affect, Behavior, Cognition, Desire) contribute to these transitions.
5. Uses clinical language appropriate for psychological assessment.

Emphasize how the individual's self-states evolve over time. Do not use bulleted lists or headers in your summary. Write in a fluid, paragraph style."""


# ── Generation (local backends: Ollama or vLLM on local weights) ─────────────

def make_generator(backend: str, model: str, host: str | None, temperature: float,
                   max_tokens: int, tensor_parallel_size: int = 4,
                   dtype: str = "auto", quantization: str | None = None,
                   gpu_memory_utilization: float = 0.9,
                   max_model_len: int | None = None, seed: int | None = None):
    """Return a function prompt -> summary string, backed by a local model.

    backend "ollama": local ollama daemon (single-GPU smoke runs).
    backend "vllm":   local weights via vLLM with tensor parallelism (cluster).
    backend "mock":   deterministic pseudo-summaries (functional tests).
    """
    if backend == "ollama":
        import ollama
        client = ollama.Client(host=host) if host else ollama
        options = {"temperature": temperature, "num_predict": max_tokens}

        def generate(prompt: str) -> str:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                options=options,
            )
            return resp["message"]["content"].strip()

        return generate

    if backend == "vllm":
        vllm_backend = VLLMBackend(
            model, tensor_parallel_size=tensor_parallel_size, dtype=dtype,
            quantization=quantization, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, seed=seed,
            temperature=temperature, max_tokens=max_tokens)

        def generate(prompt: str) -> str:
            msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}]
            return vllm_backend.generate([msgs])[0]

        return generate

    if backend == "mock":
        mock = MockBackend(max_tokens=max_tokens)

        def generate(prompt: str) -> str:
            return mock.generate([[{"role": "user", "content": prompt}]])[0]

        return generate

    raise ValueError(f"Unknown backend: {backend}")


# ── Task B / Task C drivers ───────────────────────────────────────────────────

def run_task_b(
    timelines: list[Timeline],
    generate,
    embedding_model: str,
    k: int,
    output_dir: str,
    exclude_same_timeline: bool = True,
    skip_zero_shot: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Generate EAIonFlux Task B summaries for every annotated post with a gold summary.

    Output matches the project's task_b_results schema (see EXAMPLE_DATA.json):
    `one_shot_summary` is the k-shot ICL summary (EAIonFlux proper) and
    `zero_shot_summary` is the same prompt with no retrieved demonstrations
    (an ablation; set to null when --skip_zero_shot).
    """
    print("\n" + "=" * 60)
    print(f"EAIonFlux TASK B — Post-level summaries (k={k})")
    print("=" * 60)

    all_posts = [p for tl in timelines for p in tl.posts]
    # The post-level vector DB: posts that carry both evidence and a gold summary
    # (the paper filters Task B demonstrations to examples that have both).
    db_posts = [
        p for p in all_posts
        if p.summary and (p.evidence["adaptive"] or p.evidence["maladaptive"])
    ]
    # Targets: every post with a gold summary (so it can be scored against gold).
    targets = [p for p in all_posts if p.summary]
    if limit:
        targets = targets[:limit]
    print(f"Post-level DB size: {len(db_posts)} | targets: {len(targets)}")

    # Embed DB posts and targets in one pass (post text is the query/key).
    texts = [p.text for p in db_posts] + [p.text for p in targets]
    vecs = embed_texts(embedding_model, texts)
    for p, v in zip(db_posts, vecs[:len(db_posts)]):
        p.embedding = v
    for p, v in zip(targets, vecs[len(db_posts):]):
        p.embedding = v

    results = []
    # Resumable checkpoint: per-post records appended after every generation,
    # skipped on restart. Official-format submission rewritten after each post.
    checkpoint = Checkpoint(os.path.join(output_dir, f"checkpoint_eaionflux_k{k}_taskb.jsonl"))
    submission_path = os.path.join(output_dir, f"eaionflux_k{k}_taskb_submission.json")

    pending = [p for p in targets if not checkpoint.has(p.timeline_id, p.post_id)]
    print(f"Pending targets: {len(pending)} / {len(targets)} (checkpoint resumed)")

    def _submission_records():
        return {(tl, pid): {"summary": rec["one_shot_summary"]}
                for (tl, pid), rec in checkpoint.records.items()}

    for i, post in enumerate(pending):
        candidates = [
            c for c in db_posts
            if not (c.timeline_id == post.timeline_id and c.post_index == post.post_index)
            and not (exclude_same_timeline and c.timeline_id == post.timeline_id)
        ]
        retrieved = top_k_similar(post.embedding, candidates, k)
        examples = [c for c, _ in retrieved]

        # one_shot = the k-shot ICL summary (EAIonFlux proper).
        one_shot = generate(build_task_b_prompt(post, examples))
        # zero_shot = same prompt with no retrieved demonstrations (ablation).
        zero_shot = None if skip_zero_shot else generate(build_task_b_prompt(post, []))

        top = retrieved[0] if retrieved else None
        checkpoint.append(post.timeline_id, post.post_id, {
            "post_index": post.post_index,
            "wellbeing": post.wellbeing,
            "gold_summary": post.summary,
            "n_evidence_spans": len(post.evidence["adaptive"]) + len(post.evidence["maladaptive"]),
            "evidence_texts": [t for _, _, t in post.evidence["adaptive"]]
                              + [t for _, _, t in post.evidence["maladaptive"]],
            "zero_shot_summary": zero_shot,
            "one_shot_summary": one_shot,
            "retrieved_timeline": top[0].timeline_id if top else None,
            "retrieved_post_index": top[0].post_index if top else None,
            "retrieval_similarity": round(top[1], 6) if top else None,
            # EAIonFlux provenance (additive to the canonical pipeline schema):
            "k": k,
        })
        write_task_b_submission(timelines, _submission_records(), submission_path)
        if (i + 1) % 10 == 0 or i == len(pending) - 1:
            print(f"  [{i + 1}/{len(pending)}] {post.timeline_id}#{post.post_index} "
                  f"-> {one_shot[:80]}...")

    # Final outputs: the canonical task_b_results-style JSON (all posts, in the
    # original target order) + the official-format submission.
    order = {(t.timeline_id, t.post_id): i for i, t in enumerate(targets)}
    results = sorted(
        ({"timeline_id": tl, "post_id": pid,
          **{k2: v for k2, v in rec.items() if k2 != "_key"}}
         for (tl, pid), rec in checkpoint.records.items()),
        key=lambda r: order.get((r["timeline_id"], r["post_id"]), 1 << 30))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(output_dir, f"task_b_eaionflux_k{k}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} Task B summaries to {out_path}")
    write_task_b_submission(timelines, _submission_records(), submission_path)
    return results


def run_task_c(
    timelines: list[Timeline],
    generate,
    embedding_model: str,
    k: int,
    output_dir: str,
    post_summaries: dict | None = None,
    input_type: str = "none",
    prompt_style: str = "eaionflux",
    exclude_same_timeline: bool = True,
    limit: int | None = None,
) -> list[dict]:
    """
    Generate EAIonFlux Task C summaries for every timeline with a gold summary.

    Output matches the project's task_c schema (see EXAMPLE_DATA.json):
    timeline_id, n_posts, input_type, prompt_style, gold_summary,
    predicted_summary, generation_time_sec.

    post_summaries: optional {timeline_id: {post_id: summary}} fed into the prompt
    as cascaded Task B outputs (the paper's cascade). None -> omitted.
    input_type: provenance of the post-level input ("task_b_predicted" for the
    cascade, "task_b_gold", or "none").
    """
    print("\n" + "=" * 60)
    print(f"EAIonFlux TASK C — Timeline-level summaries (k={k})")
    print("=" * 60)

    # Build structured representations for every timeline.
    for tl in timelines:
        tl.representation, tl.characteristics = build_timeline_representation(tl)

    db_timelines = [tl for tl in timelines if tl.summary]
    targets = [tl for tl in timelines if tl.summary]
    if limit:
        targets = targets[:limit]
    print(f"Timeline-level DB size: {len(db_timelines)} | targets: {len(targets)}")

    vecs = embed_texts(embedding_model, [tl.representation for tl in db_timelines])
    for tl, v in zip(db_timelines, vecs):
        tl.embedding = v

    # Resumable checkpoint: per-timeline records appended after every
    # generation, skipped on restart.
    checkpoint = Checkpoint(os.path.join(
        output_dir, f"checkpoint_eaionflux_k{k}_{input_type}_taskc.jsonl"))
    submission_path = os.path.join(
        output_dir, f"eaionflux_k{k}_{input_type}_taskc_submission.json")

    pending = [t for t in targets
               if not checkpoint.has(t.timeline_id, "__timeline__")]
    print(f"Pending timelines: {len(pending)} / {len(targets)} (checkpoint resumed)")

    def _submission_records():
        return {(tl, "__timeline__"): {"summary": rec["predicted_summary"]}
                for (tl, _), rec in checkpoint.records.items()}

    for i, tl in enumerate(pending):
        candidates = [c for c in db_timelines if c.timeline_id != tl.timeline_id]
        retrieved = top_k_similar(tl.embedding, candidates, k)
        examples = [c for c, _ in retrieved]

        ps = (post_summaries or {}).get(tl.timeline_id)
        prompt = build_task_c_prompt(tl, examples, post_summaries=ps)
        t0 = time.perf_counter()
        summary = generate(prompt)
        gen_time = round(time.perf_counter() - t0, 1)

        checkpoint.append(tl.timeline_id, "__timeline__", {
            "n_posts": len(tl.posts),
            "input_type": input_type,
            "prompt_style": prompt_style,
            "gold_summary": tl.summary,
            "predicted_summary": summary,
            "generation_time_sec": gen_time,
            # EAIonFlux provenance (additive to the canonical schema):
            "k": k,
            "retrieved": [
                {"timeline_id": c.timeline_id, "similarity": round(s, 4)}
                for c, s in retrieved
            ],
        })
        write_task_c_submission(timelines, _submission_records(), submission_path)
        print(f"  [{i + 1}/{len(pending)}] {tl.timeline_id} -> {summary[:80]}...")

    order = {t.timeline_id: i for i, t in enumerate(targets)}
    results = sorted(
        ({"timeline_id": tl,
          **{k2: v for k2, v in rec.items() if k2 != "_key"}}
         for (tl, _), rec in checkpoint.records.items()),
        key=lambda r: order.get(r["timeline_id"], 1 << 30))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(output_dir, f"task_c_eaionflux_k{k}_{input_type}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} Task C summaries to {out_path}")
    write_task_c_submission(timelines, _submission_records(), submission_path)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EAIonFlux (CLPsych 2025) ICL reproduction for Tasks B and C."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory of timeline JSON files (or a single .json).")
    parser.add_argument("--output_dir", default="./outputs",
                        help="Where to write task_b_eaionflux / task_c_eaionflux JSON.")
    parser.add_argument("--task", default="both", choices=["B", "C", "both"],
                        help="Which task(s) to run.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of in-context examples to retrieve (paper used 5 and 10).")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "vllm", "mock"],
                        help="Generation backend: ollama (local daemon, default), "
                             "vllm (cluster, local weights + tensor parallelism), "
                             "mock (functional tests).")
    parser.add_argument("--model", default=None,
                        help="Generation model. Defaults: ollama backend -> "
                             f"{DEFAULT_OLLAMA_MODEL} (paper), vllm backend -> "
                             f"{DEFAULT_VLLM_MODEL} (paper). For vllm pass the "
                             "path to the local weights directory.")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
                        help="sentence-transformers embedding model id (paper: "
                             "Linq-AI-Research/Linq-Embed-Mistral). "
                             f"Default: {DEFAULT_EMBEDDING_MODEL}.")
    parser.add_argument("--ollama_host", default=None,
                        help="Ollama host URL (default: local daemon).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="vLLM tensor parallelism across GPUs (vllm backend only).")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None,
                        help="Optional vLLM quantization (e.g. 'fp8', 'awq') for 70B "
                             "on smaller GPUs.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM context length (None = model default).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--no_exclude_same_timeline", action="store_true",
                        help="Allow retrieving demonstrations from the target's own timeline.")
    parser.add_argument("--skip_zero_shot", action="store_true",
                        help="Task B: only generate the k-shot summary (one_shot_summary); "
                             "set zero_shot_summary to null. Halves generation cost.")
    parser.add_argument("--prompt_style", default="eaionflux",
                        help="Label stored in the Task C output's prompt_style field.")
    parser.add_argument("--use_gold_post_summaries", action="store_true",
                        help="Task C only: feed gold post summaries into the prompt when "
                             "Task B was not run in the same invocation. Off by default to "
                             "avoid leaking gold post summaries into the timeline summary.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N targets (smoke testing).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exclude_same = not args.no_exclude_same_timeline

    if args.model is None:
        args.model = DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_VLLM_MODEL

    timelines = load_timelines(args.data_dir)
    generate = make_generator(
        args.backend, args.model, args.ollama_host, args.temperature,
        args.max_tokens, tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype, quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len)
    print(f"Generation backend: {args.backend} | model: {args.model} | "
          f"Embedding model: {args.embedding_model}")

    write_settings_log(
        os.path.join(args.output_dir, "eaionflux_settings.md"),
        f"EAIonFlux Tasks B+C (k={args.k}) — generation settings",
        settings=[
            ("generation model", args.model, "paper (llama3.3:70b)"),
            ("backend", args.backend, "our setup (vLLM local weights on cluster / ollama locally)"),
            ("tensor_parallel_size", args.tensor_parallel_size, "our setup"),
            ("temperature", args.temperature,
             "our default — paper does not report decoding settings"),
            ("top_p", 1.0, "our default — paper silent"),
            ("max_tokens", args.max_tokens, "our default — paper silent"),
            ("k (in-context examples)", args.k, "paper (5 and 10)"),
            ("embedding model", args.embedding_model,
             "paper (Linq-Embed-Mistral); lighter default available"),
            ("retrieval", "exact top-k cosine (paper: ChromaDB ANN)",
             "paper + our equivalence note"),
            ("prompts", "paper Appendix A.2 / A.3 templates", "paper"),
            ("evidence input", "gold adaptive/maladaptive spans + gold well-being",
             "project convention"),
            ("leave-one-timeline-out", exclude_same, "project convention"),
        ],
        notes=[
            "Resumable: per-post/per-timeline JSONL checkpoints; completed "
            "targets are skipped on restart.",
            "Official-format submission JSONs are rewritten after every "
            "generation (eaionflux_k{K}_taskb_submission.json / "
            "eaionflux_k{K}_{input_type}_taskc_submission.json).",
        ],
    )

    # post_summaries map fed into Task C: cascaded Task B outputs if both run,
    # else optionally gold, else None. input_type records that provenance.
    post_summaries = None
    input_type = "none"

    if args.task in ("B", "both"):
        b_results = run_task_b(
            timelines, generate, args.embedding_model, args.k, args.output_dir,
            exclude_same_timeline=exclude_same, skip_zero_shot=args.skip_zero_shot,
            limit=args.limit,
        )
        if args.task == "both":
            post_summaries = {}
            for r in b_results:
                if r.get("one_shot_summary"):
                    post_summaries.setdefault(r["timeline_id"], {})[r["post_id"]] = r["one_shot_summary"]
            input_type = "task_b_predicted"

    if args.task in ("C", "both"):
        if post_summaries is None and args.use_gold_post_summaries:
            post_summaries = {}
            for tl in timelines:
                post_summaries[tl.timeline_id] = {
                    p.post_id: p.summary for p in tl.posts if p.summary
                }
            input_type = "task_b_gold"
        run_task_c(
            timelines, generate, args.embedding_model, args.k, args.output_dir,
            post_summaries=post_summaries, input_type=input_type,
            prompt_style=args.prompt_style, exclude_same_timeline=exclude_same,
            limit=args.limit,
        )

    print("\nEAIonFlux run complete.")


if __name__ == "__main__":
    main()
