"""
CLPsych 2025 Pipeline: Structural Retrieval-Augmented Summarization

Pipeline:
    1. Task A.3 — ABCD classification of gold evidence spans
    2. Structural profile construction (12-dim + wellbeing + ratio)
    3. Cosine similarity retrieval for in-context example selection
    4. Task B — Post-level summarization (zero-shot and one-shot)

IMPORTANT: Run locally. Do not use cloud APIs or cloud-based AI assistants
with this script or the data. The data sharing agreement prohibits sending
data to third-party LLM providers.

NOTE: Teammate has a 4080 super to run locally, testing can be done by @danwein8
"""

import json
import os
import argparse
import time
import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from llama_cpp import Llama

# ── Configuration ────────────────────────────────────────────────────────────

_LLAMA_REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
_LLAMA_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

ABCD_KEYS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
STATE_TYPES = ["adaptive-state", "maladaptive-state"]

# 12-dimensional profile: 6 ABCD elements x 2 polarities (adaptive, maladaptive)
PROFILE_DIM_LABELS = [
    f"{elem}_{pol}" for pol in ["adaptive", "maladaptive"] for elem in ABCD_KEYS
]

# ── MIND Framework Definitions (from Table 10 of the shared task paper) ──────

ABCD_DEFINITIONS = {
    "A": {
        "name": "Affect",
        "description": "The type of emotion expressed by the person.",
        "adaptive_examples": "Calm/Laid back, Emotional Pain/Grieving, Content/Happy, Vigor/Energetic, Justifiable Anger/Assertive Anger, Proud",
        "maladaptive_examples": "Anxious/Tense/Fearful, Depressed/Desperate/Hopeless, Mania, Apathetic/Don't care/Blunted, Angry (Aggressive, Disgust, Contempt), Ashamed/Guilty",
    },
    "B-O": {
        "name": "Behavior toward the Other",
        "description": "The person's main behavior(s) toward the other.",
        "adaptive_examples": "Relating behavior, Autonomous behavior",
        "maladaptive_examples": "Fight or flight behavior, Overcontrolled/controlling behavior",
    },
    "B-S": {
        "name": "Behavior toward the Self",
        "description": "The person's main behavior(s) toward the self.",
        "adaptive_examples": "Self-care behavior",
        "maladaptive_examples": "Self-harm/Neglect/Avoidance behavior",
    },
    "C-O": {
        "name": "Cognition of the Other",
        "description": "The person's main perceptions of the other.",
        "adaptive_examples": "Perception of the other as related, Perception of the other as facilitating autonomy/competence needs",
        "maladaptive_examples": "Perception of the other as detached or over attached, Perception of the other as blocking autonomy needs",
    },
    "C-S": {
        "name": "Cognition of the Self",
        "description": "The person's main self-perceptions.",
        "adaptive_examples": "Self-acceptance and self-compassion",
        "maladaptive_examples": "Self-criticism",
    },
    "D": {
        "name": "Desire",
        "description": "The person's main desire, need, intention, fear or expectation.",
        "adaptive_examples": "Relatedness, Autonomy and adaptive control, Competence, Self-esteem, Self-care",
        "maladaptive_examples": "Expectation that relatedness need will not be met, Expectation that autonomy needs will not be met, Expectation that competence needs will not be met",
    },
}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class EvidenceSpan:
    """A single evidence span with its gold and predicted labels."""
    text: str
    gold_abcd_key: str          # e.g., "A", "B-O", "C-S"
    gold_category: str          # e.g., "Self-criticism"
    polarity: str               # "adaptive" or "maladaptive"
    predicted_abcd_key: str = None
    timeline_id: str = ""
    post_index: int = -1


@dataclass
class PostProfile:
    """A post with its structural profile for retrieval."""
    timeline_id: str
    post_index: int
    post_text: str
    wellbeing: float
    gold_summary: str
    evidence_spans: list        # list of EvidenceSpan
    is_annotated: bool = True   # False for surrounding-context-only posts
    # The 12-dim binary profile + wellbeing + ratio
    structural_vector: np.ndarray = None
    # Classified ABCD labels (from Task A.3)
    classified_abcd: dict = field(default_factory=dict)
    # Surrounding posts in the same timeline, used as prompt context.
    # List of dicts: {"post_index": int, "post_text": str, "position": "before"|"after"}
    context_posts: list = field(default_factory=list)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_timelines(data_dir: str) -> list[dict]:
    """Load all timeline JSON files from a directory."""
    data_path = Path(data_dir)
    if data_path.is_file() and data_path.suffix == ".json":
        json_files = [data_path]
    else:
        json_files = sorted(data_path.glob("*.json"))

    timelines = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            timelines.extend(data)
        else:
            timelines.append(data)

    print(f"Loaded {len(timelines)} timelines from {len(json_files)} file(s)")
    return timelines


def extract_posts(
    timelines: list[dict],
    context_window: int = 1,
) -> list[PostProfile]:
    """
    Extract every post in every timeline into PostProfile objects.

    Annotated posts (those with `Post Summary` or `Well-being` set) get
    `is_annotated=True` and their evidence spans parsed. Unannotated posts
    are kept too — they are used as surrounding-context in the summarization
    prompts but never summarized themselves.

    After extraction, each annotated post is given a `context_posts` list
    containing the `context_window` posts immediately before and after it in
    the same timeline (ordered by `post_index`). Context posts can be either
    annotated or unannotated.
    """
    posts = []
    for tl in timelines:
        tl_id = tl.get("timeline_id", "unknown")
        for post_data in tl.get("posts", []):
            wb = post_data.get("Well-being", None)
            summary = post_data.get("Post Summary", None)
            is_annotated = not (wb is None and summary is None)

            try:
                wb_float = float(str(wb).strip())
            except (ValueError, TypeError):
                wb_float = None

            spans = []
            if is_annotated:
                evidence = post_data.get("evidence", {})
                for state_type in STATE_TYPES:
                    polarity = "adaptive" if state_type == "adaptive-state" else "maladaptive"
                    state_data = evidence.get(state_type, {})
                    for abcd_key in ABCD_KEYS:
                        if abcd_key in state_data:
                            span_info = state_data[abcd_key]
                            span_text = span_info.get("highlighted_evidence", "")
                            category = span_info.get("Category", "")
                            if span_text:
                                spans.append(EvidenceSpan(
                                    text=span_text,
                                    gold_abcd_key=abcd_key,
                                    gold_category=category,
                                    polarity=polarity,
                                    timeline_id=tl_id,
                                    post_index=post_data.get("post_index", -1),
                                ))

            posts.append(PostProfile(
                timeline_id=tl_id,
                post_index=post_data.get("post_index", -1),
                post_text=post_data.get("post", ""),
                wellbeing=wb_float,
                gold_summary=summary or "",
                evidence_spans=spans,
                is_annotated=is_annotated,
            ))

    # Attach surrounding-context posts within each timeline.
    by_timeline: dict[str, list[PostProfile]] = {}
    for p in posts:
        by_timeline.setdefault(p.timeline_id, []).append(p)
    for tl_posts in by_timeline.values():
        tl_posts.sort(key=lambda p: p.post_index)
        for i, p in enumerate(tl_posts):
            if not p.is_annotated:
                continue
            before = tl_posts[max(0, i - context_window): i]
            after = tl_posts[i + 1: i + 1 + context_window]
            p.context_posts = (
                [{"post_index": q.post_index, "post_text": q.post_text, "position": "before"}
                 for q in before]
                + [{"post_index": q.post_index, "post_text": q.post_text, "position": "after"}
                   for q in after]
            )

    annotated = [p for p in posts if p.is_annotated]
    with_evidence = [p for p in annotated if p.evidence_spans]
    print(f"Extracted {len(posts)} posts total, "
          f"{len(annotated)} annotated, {len(with_evidence)} with evidence spans")
    return posts


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str = None, n_ctx: int = 16384, n_gpu_layers: int = -1) -> Llama:
    """
    Load the Llama model from a local GGUF file or download from HuggingFace.

    Args:
        model_path: Path to a local .gguf file. If None, downloads from HF.
        n_ctx: Context window size.
        n_gpu_layers: Number of layers to offload to GPU (-1 = all).
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading model from local path: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    else:
        print(f"Downloading model from HuggingFace: {_LLAMA_REPO_ID}")
        llm = Llama.from_pretrained(
            repo_id=_LLAMA_REPO_ID,
            filename=_LLAMA_FILENAME,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    print("Model loaded successfully")
    return llm


def generate(llm: Llama, prompt: str, max_tokens: int = 256, temperature: float = 0.01) -> str:
    """Generate a completion from the model using chat format."""
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )
    return response["choices"][0]["message"]["content"].strip()


# ── Task A.3: ABCD Classification ────────────────────────────────────────────

def build_abcd_classification_prompt(span_text: str, polarity: str) -> str:
    """
    Build a prompt to classify an evidence span into one of the 6 ABCD categories.

    The span is already known to be adaptive or maladaptive. The model just
    needs to determine which psychological dimension it reflects.
    """
    polarity_label = polarity.capitalize()

    # Build the category definitions section
    category_defs = []
    example_key = "adaptive_examples" if polarity == "adaptive" else "maladaptive_examples"
    for key in ABCD_KEYS:
        defn = ABCD_DEFINITIONS[key]
        examples = defn[example_key]
        category_defs.append(
            f"- {key} ({defn['name']}): {defn['description']} "
            f"Examples of {polarity} sub-categories: {examples}"
        )
    categories_text = "\n".join(category_defs)

    prompt = f"""You are a clinical psychology expert classifying mental health evidence spans.

Given a text span from a social media post that has been identified as reflecting a {polarity_label} self-state, classify it into exactly ONE of the following ABCD categories:

{categories_text}

Text span: "{span_text}"
This span reflects a {polarity_label} self-state.

Respond with ONLY the category key (one of: A, B-O, B-S, C-O, C-S, D) and nothing else."""

    return prompt


def parse_abcd_prediction(raw_output: str) -> str:
    """
    Parse the model's output to extract the predicted ABCD key.
    Handles cases where the model outputs extra text.
    """
    raw = raw_output.strip().upper()

    # Direct match
    for key in ABCD_KEYS:
        if raw == key.upper():
            return key

    # Check if any key appears in the output
    # Check longer keys first to avoid "B-O" matching just "B"
    for key in sorted(ABCD_KEYS, key=len, reverse=True):
        if key.upper() in raw:
            return key

    # Fallback: check for full names
    name_map = {
        "AFFECT": "A",
        "BEHAVIOR TOWARD THE OTHER": "B-O",
        "BEHAVIOR OTHER": "B-O",
        "BEHAVIOR TOWARD THE SELF": "B-S",
        "BEHAVIOR SELF": "B-S",
        "COGNITION OF THE OTHER": "C-O",
        "COGNITION OTHER": "C-O",
        "COGNITION OF THE SELF": "C-S",
        "COGNITION SELF": "C-S",
        "DESIRE": "D",
    }
    for name, key in name_map.items():
        if name in raw:
            return key

    return "UNKNOWN"


def run_abcd_classification(
    llm: Llama,
    posts: list[PostProfile],
    verbose: bool = True
) -> list[EvidenceSpan]:
    """
    Run Task A.3: classify each gold evidence span into ABCD categories.

    This function only produces predictions and stores them on each span.
    Metric computation (accuracy, macro-F1, classification report) lives in
    `evaluation.py` — call `Evaluator.evaluate_task_a3(...)` on the
    `task_a3_predictions.json` file written by main().

    Returns the flat list of all evidence spans with `predicted_abcd_key` set.
    """
    print("\n" + "=" * 60)
    print("TASK A.3: ABCD Classification of Evidence Spans")
    print("=" * 60)

    all_spans = [span for post in posts for span in post.evidence_spans]

    if not all_spans:
        print("No evidence spans to classify.")
        return []

    print(f"Classifying {len(all_spans)} evidence spans...")

    n_correct = 0
    for i, span in enumerate(all_spans):
        prompt = build_abcd_classification_prompt(span.text, span.polarity)
        # MAX_TOKENS=20 bc all we want is a category, 1-3 tokens for category plus a safety net
        # parse_abcd_prediction() gets just the category if LLM is chatty
        raw_output = generate(llm, prompt, max_tokens=20)
        span.predicted_abcd_key = parse_abcd_prediction(raw_output)
        n_correct += int(span.predicted_abcd_key == span.gold_abcd_key)

        if verbose and (i + 1) % 25 == 0:
            running_acc = n_correct / (i + 1)
            print(f"  [{i+1}/{len(all_spans)}] Running accuracy: {running_acc:.3f}")

    print(f"\nClassified {len(all_spans)} spans. "
          f"Run evaluation.py --task a3 to compute final metrics.")
    return all_spans


# ── Structural Profile Construction ──────────────────────────────────────────

def build_structural_profile(post: PostProfile, use_predicted: bool = False) -> np.ndarray:
    """
    Construct a structural profile vector for a post.

    The vector has 14 dimensions:
      - 12 binary indicators: 6 ABCD keys x 2 polarities (adaptive, maladaptive)
        Whether that ABCD element is present in that polarity's evidence.
      - 1 continuous: normalized well-being score (0-1)
      - 1 continuous: ratio of adaptive evidence to total evidence

    Args:
        post: PostProfile with evidence spans
        use_predicted: If True, use predicted ABCD keys from Task A.3.
                       If False, use gold ABCD keys.
    """
    # 12-dim binary: presence of each ABCD element in each polarity
    profile = np.zeros(12, dtype=np.float32)

    for span in post.evidence_spans:
        abcd_key = span.predicted_abcd_key if use_predicted else span.gold_abcd_key
        if abcd_key not in ABCD_KEYS:
            continue
        # show which span keys are present in binary vector
        key_idx = ABCD_KEYS.index(abcd_key)
        if span.polarity == "adaptive":
            profile[key_idx] = 1.0           # indices 0-5: adaptive
        else:
            profile[6 + key_idx] = 1.0       # indices 6-11: maladaptive

    # Normalized well-being score (1-10 -> 0-1)
    wb_norm = (post.wellbeing - 1) / 9.0 if post.wellbeing is not None else 0.5

    # Adaptive ratio
    n_adaptive = sum(1 for s in post.evidence_spans if s.polarity == "adaptive")
    n_total = len(post.evidence_spans)
    adaptive_ratio = n_adaptive / n_total if n_total > 0 else 0.5

    full_vector = np.append(profile, [wb_norm, adaptive_ratio])
    return full_vector


def build_all_profiles(posts: list[PostProfile], use_predicted: bool = False):
    """Build structural profile vectors for all posts."""
    for post in posts:
        post.structural_vector = build_structural_profile(post, use_predicted)

    print(f"Built structural profiles for {len(posts)} posts "
          f"(vector dim: {posts[0].structural_vector.shape[0] if posts else 0})")


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_nearest(
    target: PostProfile,
    pool: list[PostProfile],
    k: int = 1,
    exclude_same_timeline: bool = True,
) -> list[tuple[PostProfile, float]]:
    """
    Retrieve the k nearest posts from the pool based on structural profile
    cosine similarity.

    Args:
        target: The post we want to find a match for.
        pool: The set of candidate posts to retrieve from.
        k: Number of nearest neighbors to return.
        exclude_same_timeline: If True, don't retrieve from the same timeline.

    Returns:
        List of (PostProfile, similarity_score) tuples, sorted by similarity.
    """
    if target.structural_vector is None:
        raise ValueError("Target post has no structural vector. Run build_all_profiles first.")

    # Filter pool
    candidates = []
    for p in pool:
        # must have structural vector profile
        if p.structural_vector is None:
            continue
        # must be in different timeline if exclude_same_timeline=True
        if exclude_same_timeline and p.timeline_id == target.timeline_id:
            continue
        # Don't retrieve the exact same post
        if p.timeline_id == target.timeline_id and p.post_index == target.post_index:
            continue
        # Only retrieve posts that have gold summaries (for use as demonstrations)
        if not p.gold_summary:
            continue
        candidates.append(p)

    if not candidates:
        return []

    # Compute cosine similarities
    target_vec = target.structural_vector.reshape(1, -1)
    pool_vecs = np.array([p.structural_vector for p in candidates])
    similarities = cosine_similarity(target_vec, pool_vecs)[0]

    # Sort by similarity (descending) and take top k
    top_indices = np.argsort(similarities)[::-1][:k]
    results = [(candidates[i], float(similarities[i])) for i in top_indices]

    return results


# ── Task B: Post-Level Summarization ─────────────────────────────────────────

def format_evidence_for_prompt(post: PostProfile, use_predicted: bool = False) -> str:
    """Format a post's evidence spans for inclusion in a prompt."""
    adaptive_spans = []
    maladaptive_spans = []

    for span in post.evidence_spans:
        abcd_key = span.predicted_abcd_key if use_predicted else span.gold_abcd_key
        abcd_name = ABCD_DEFINITIONS.get(abcd_key, {}).get("name", abcd_key)
        entry = f'  - {abcd_name} ({abcd_key}): "{span.text}"'

        if span.polarity == "adaptive":
            adaptive_spans.append(entry)
        else:
            maladaptive_spans.append(entry)

    parts = []
    if adaptive_spans:
        parts.append("Adaptive self-state evidence:\n" + "\n".join(adaptive_spans))
    else:
        parts.append("Adaptive self-state evidence: None identified")

    if maladaptive_spans:
        parts.append("Maladaptive self-state evidence:\n" + "\n".join(maladaptive_spans))
    else:
        parts.append("Maladaptive self-state evidence: None identified")

    return "\n".join(parts)


def format_context_for_prompt(post: PostProfile) -> str:
    """
    Format a post's surrounding-context posts for inclusion in a prompt.

    Returns an empty string if there is no context. Otherwise returns a
    block listing posts that came BEFORE and AFTER the target post in
    the same timeline (oldest first within each section).
    """
    if not post.context_posts:
        return ""

    before = [c for c in post.context_posts if c["position"] == "before"]
    after = [c for c in post.context_posts if c["position"] == "after"]

    parts = []
    if before:
        parts.append("Earlier posts in this timeline:")
        for c in before:
            parts.append(f'  [post {c["post_index"]}] "{c["post_text"]}"')
    if after:
        parts.append("Later posts in this timeline:")
        for c in after:
            parts.append(f'  [post {c["post_index"]}] "{c["post_text"]}"')
    return "\n".join(parts)


def build_zero_shot_prompt(post: PostProfile, use_predicted: bool = False) -> str:
    """
    Build a zero-shot prompt for Task B post-level summarization.

    Includes any surrounding-timeline context, the post text, extracted
    evidence with ABCD classifications, and well-being score.
    """
    evidence_text = format_evidence_for_prompt(post, use_predicted)
    wb_str = f"{post.wellbeing:.0f}" if post.wellbeing is not None else "Unknown"
    context_text = format_context_for_prompt(post)
    context_block = (
        f"For situational context only (do NOT summarize these), here are the "
        f"surrounding posts in the same timeline:\n{context_text}\n\n"
        if context_text else ""
    )

    prompt = f"""You are a clinical psychology expert analyzing social media posts for mental health dynamics.

Analyze the following social media post and generate a clinical summary of the person's self-states. Follow these guidelines:

1. Identify the dominant self-state (adaptive or maladaptive) and describe it first.
2. For each self-state present, highlight the central organizing ABCD aspect — Affect (A), Behavior (B), Cognition (C), or Desire (D) — that drives the state.
3. Describe how this central aspect influences the other aspects, focusing on causal relationships between them.
4. If both adaptive and maladaptive states are present, describe each in turn, noting their interplay.
5. Keep the summary concise (3-6 sentences). Only describe observations fully supported by the text.

{context_block}Post to summarize:
"{post.post_text}"

Extracted evidence with ABCD classifications:
{evidence_text}

Well-being score: {wb_str}/10

Summary:"""

    return prompt


def build_one_shot_prompt(
    post: PostProfile,
    example: PostProfile,
    use_predicted: bool = False,
) -> str:
    """
    Build a one-shot prompt for Task B using a retrieved example.

    The example serves as a demonstration of how the model should structure
    its summary, grounded in a structurally similar post.
    """
    # Format the example (always use gold ABCD for the demonstration)
    ex_evidence = format_evidence_for_prompt(example, use_predicted=False)
    ex_wb = f"{example.wellbeing:.0f}" if example.wellbeing is not None else "Unknown"

    # Format the target
    tgt_evidence = format_evidence_for_prompt(post, use_predicted)
    tgt_wb = f"{post.wellbeing:.0f}" if post.wellbeing is not None else "Unknown"
    tgt_context = format_context_for_prompt(post)
    tgt_context_block = (
        f"For situational context only (do NOT summarize these), here are the "
        f"surrounding posts in the same timeline:\n{tgt_context}\n\n"
        if tgt_context else ""
    )

    prompt = f"""You are a clinical psychology expert analyzing social media posts for mental health dynamics.

For each post, generate a clinical summary capturing the interplay between adaptive and maladaptive self-states. Follow these guidelines:

1. Identify the dominant self-state (adaptive or maladaptive) and describe it first.
2. For each self-state, highlight the central organizing ABCD aspect — Affect (A), Behavior (B), Cognition (C), or Desire (D) — that drives the state.
3. Describe how this central aspect influences the other aspects, focusing on causal relationships.
4. If both states are present, describe each in turn, noting their interplay.
5. Keep the summary concise (3-6 sentences). Only describe observations fully supported by the text.

Here is an example:

Post content:
"{example.post_text}"

Extracted evidence with ABCD classifications:
{ex_evidence}

Well-being score: {ex_wb}/10

Summary:
{example.gold_summary}

---

Now summarize the following post in the same style:

{tgt_context_block}Post to summarize:
"{post.post_text}"

Extracted evidence with ABCD classifications:
{tgt_evidence}

Well-being score: {tgt_wb}/10

Summary:"""

    return prompt


def run_task_b(
    llm: Llama,
    target_posts: list[PostProfile],
    retrieval_pool: list[PostProfile],
    use_predicted: bool = False,
    output_dir: str = "./outputs",
):
    """
    Run Task B summarization in both zero-shot and one-shot modes.

    Args:
        llm: The loaded language model.
        target_posts: Posts to generate summaries for.
        retrieval_pool: Posts available for retrieval (typically training set).
        use_predicted: Whether to use predicted ABCD labels from Task A.3.
        output_dir: Directory to save outputs.
    """
    print("\n" + "=" * 60)
    print("TASK B: Post-Level Summarization")
    print("=" * 60)

    results = []

    for i, post in enumerate(target_posts):
        if not post.gold_summary:
            continue

        print(f"\n--- Post {i+1}/{len(target_posts)} "
              f"(timeline={post.timeline_id}, index={post.post_index}) ---")

        entry = {
            "timeline_id": post.timeline_id,
            "post_index": post.post_index,
            "wellbeing": post.wellbeing,
            "gold_summary": post.gold_summary,
            "n_evidence_spans": len(post.evidence_spans),
            "evidence_texts": [s.text for s in post.evidence_spans],
        }

        # ── Zero-shot ──
        zs_prompt = build_zero_shot_prompt(post, use_predicted)
        zs_summary = generate(llm, zs_prompt, max_tokens=400, temperature=0.1)
        entry["zero_shot_summary"] = zs_summary
        print(f"  Zero-shot summary: {zs_summary[:120]}...")

        # ── One-shot with structural retrieval ──
        retrieved = retrieve_nearest(
            post, retrieval_pool, k=1, exclude_same_timeline=True
        )
        if retrieved:
            example, sim_score = retrieved[0]
            os_prompt = build_one_shot_prompt(post, example, use_predicted)
            os_summary = generate(llm, os_prompt, max_tokens=400, temperature=0.1)
            entry["one_shot_summary"] = os_summary
            entry["retrieved_timeline"] = example.timeline_id
            entry["retrieved_post_index"] = example.post_index
            entry["retrieval_similarity"] = sim_score
            print(f"  One-shot summary (sim={sim_score:.3f}): {os_summary[:120]}...")
        else:
            entry["one_shot_summary"] = None
            print("  One-shot: No suitable example found in retrieval pool.")

        results.append(entry)

    # Save results
    output_path = os.path.join(output_dir, f"task_b_results_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} summaries to {output_path}")

    return results


# ── Evaluation Helpers ───────────────────────────────────────────────────────

def print_retrieval_analysis(posts: list[PostProfile]):
    """Analyze the structural profiles and retrieval characteristics."""
    print("\n" + "=" * 60)
    print("RETRIEVAL ANALYSIS")
    print("=" * 60)

    vectors = np.array([p.structural_vector for p in posts if p.structural_vector is not None])
    if len(vectors) == 0:
        print("No profiles to analyze.")
        return

    # Profile statistics
    print(f"\nProfile statistics (n={len(vectors)}):")
    print(f"  Vector dimensionality: {vectors.shape[1]}")

    # Unique profiles
    unique = len(set(tuple(v[:12].astype(int)) for v in vectors))
    print(f"  Unique binary profiles (12-dim): {unique} / {len(vectors)}")

    # Average active dimensions
    avg_active = vectors[:, :12].sum(axis=1).mean()
    print(f"  Avg active ABCD dimensions per post: {avg_active:.2f}")

    # Pairwise similarity distribution
    if len(vectors) > 1:
        # NxN matrix where (i,j) is the cosine similarity between post i's structural vector and post j's structural vector
        sims = cosine_similarity(vectors)
        # Get upper triangle (exclude diagonal) bc matrix is symmetric and diagonals are 1.0
        upper = sims[np.triu_indices_from(sims, k=1)]
        print(f"  Pairwise cosine similarity:")
        print(f"    Mean: {upper.mean():.3f}")
        print(f"    Std:  {upper.std():.3f}")
        print(f"    Min:  {upper.min():.3f}")
        print(f"    Max:  {upper.max():.3f}")

    # Well-being distribution
    wbs = [p.wellbeing for p in posts if p.wellbeing is not None]
    if wbs:
        print(f"\n  Well-being scores: mean={np.mean(wbs):.2f}, "
              f"std={np.std(wbs):.2f}, range=[{min(wbs):.0f}, {max(wbs):.0f}]")

    # Adaptive ratio distribution
    ratios = vectors[:, -1]
    print(f"  Adaptive ratio: mean={ratios.mean():.3f}, std={ratios.std():.3f}")


# ── Leave-One-Timeline-Out Cross-Validation ──────────────────────────────────

def run_cross_validation(
    llm: Llama,
    posts: list[PostProfile],
    use_predicted: bool = False,
    output_dir: str = "./outputs",
):
    """
    Run leave-one-timeline-out cross-validation for Task B.

    For each timeline, use all other timelines as the retrieval pool
    and generate summaries for posts in the held-out timeline.
    """
    print("\n" + "=" * 60)
    print("LEAVE-ONE-TIMELINE-OUT CROSS-VALIDATION")
    print("=" * 60)

    timeline_ids = sorted(set(p.timeline_id for p in posts))
    print(f"Running over {len(timeline_ids)} timelines")

    all_results = []

    for tl_id in timeline_ids:
        target_posts = [p for p in posts if p.timeline_id == tl_id and p.gold_summary]
        pool_posts = [p for p in posts if p.timeline_id != tl_id and p.gold_summary]

        if not target_posts or not pool_posts:
            continue

        print(f"\n  Timeline {tl_id}: {len(target_posts)} target posts, "
              f"{len(pool_posts)} pool posts")

        tl_results = run_task_b(
            llm, target_posts, pool_posts,
            use_predicted=use_predicted,
            output_dir=output_dir,
        )
        all_results.extend(tl_results)

    # Save consolidated results
    cv_path = os.path.join(output_dir, f"task_b_cv_results_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved cross-validation results ({len(all_results)} summaries) to {cv_path}")

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLPsych 2025: Structural Retrieval-Augmented Summarization Pipeline"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with timeline JSON files")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local .gguf model file (downloads from HF if not set)")
    parser.add_argument("--n_ctx", type=int, default=16384,
                        help="Context window size for the model")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="GPU layers to offload (-1 = all)")
    parser.add_argument("--skip_a3", action="store_true",
                        help="Skip Task A.3 and use gold ABCD labels directly")
    parser.add_argument("--cross_validate", action="store_true",
                        help="Run leave-one-timeline-out cross-validation")
    parser.add_argument("--context_window", type=int, default=2,
                        help="Number of surrounding posts (before and after) to "
                             "include from the same timeline as prompt context. 0 disables.")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    timelines = load_timelines(args.data_dir)
    posts = extract_posts(timelines, context_window=args.context_window)

    annotated_posts = [p for p in posts if p.evidence_spans and p.gold_summary]
    print(f"Posts with both evidence and summaries: {len(annotated_posts)}")

    # ── Load model ──
    llm = load_model(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    # ── Task A.3: ABCD Classification ──
    # Predictions only — metrics are computed by evaluation.py.
    use_predicted = False
    if not args.skip_a3:
        all_spans = run_abcd_classification(
            llm, annotated_posts, verbose=args.verbose
        )
        use_predicted = True

        a3_output = {
            "spans": [
                {
                    "text": s.text[:100],  # truncate for privacy
                    "gold": s.gold_abcd_key,
                    "predicted": s.predicted_abcd_key,
                    "polarity": s.polarity,
                    "timeline_id": s.timeline_id,
                    "post_index": s.post_index,
                }
                for s in all_spans
            ],
        }
        a3_path = os.path.join(args.output_dir, f"task_a3_predictions_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json")
        with open(a3_path, "w") as f:
            json.dump(a3_output, f, indent=2)
        print(f"Saved A.3 predictions to {a3_path}")

    # ── Build structural profiles ──
    build_all_profiles(annotated_posts, use_predicted=use_predicted)
    print_retrieval_analysis(annotated_posts)

    # ── Task B: Summarization ──
    if args.cross_validate:
        run_cross_validation(
            llm, annotated_posts,
            use_predicted=use_predicted,
            output_dir=args.output_dir,
        )
    else:
        # Simple mode: use all posts as both targets and retrieval pool
        # (retrieval excludes same-timeline posts)
        run_task_b(
            llm, annotated_posts, annotated_posts,
            use_predicted=use_predicted,
            output_dir=args.output_dir,
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()