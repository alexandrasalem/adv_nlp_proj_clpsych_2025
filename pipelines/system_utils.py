"""Shared utilities for the CLPsych 2025 Task B system (re)implementations.

Used by pipelines/{bulusi,aquarius,ism,uottawa}_model.py and (for the new
vLLM/checkpoint/submission features) eaionflux_model.py.

Provides:
    * data loading for the shared-task timeline JSONs (same schema and
      dataclasses as eaionflux_model.py);
    * leave-one-timeline-out (LOTO) example filtering — no system may draw an
      in-context example from the target post's own timeline;
    * generation backends:
        - vllm  : local weights, tensor parallelism across GPUs (cluster);
        - ollama: local ollama daemon (single-GPU smoke runs);
        - mock  : deterministic pseudo-summaries (functional tests, no models);
    * resumable per-post checkpoints (JSONL; append after every batch, skipped
      on restart);
    * official submission-format writer ({timeline_id: {"timeline_level",
      "post_level": {post_id: {...}}}} with gold evidence + gold well-being),
      the same JSON structure dev.json / the gold data uses, so outputs plug
      straight into eval/task31_eval.py and clpsych2025/evaluation/run.py;
    * a Markdown generation-settings log recording every decoding setting and
      whether it was reported by the paper or chosen as our documented default
      (this log is the source for the paper's reimplementation section).

Everything runs on local weights / a local inference server. Do NOT point any
backend at a hosted API — the data sharing agreement prohibits sending the
data to third-party LLM providers.
"""

import datetime
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Schema constants (shared with eaionflux_model.py) ────────────────────────

ABCD_KEYS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
STATE_TYPES = ["adaptive-state", "maladaptive-state"]
ABCD_NAMES = {
    "A": "Affect",
    "B-O": "Behavior-Other",
    "B-S": "Behavior-Self",
    "C-O": "Cognition-Other",
    "C-S": "Cognition-Self",
    "D": "Desire/Need",
}


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
    representation: str = ""
    characteristics: dict = field(default_factory=dict)
    embedding: np.ndarray = None


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
    """Load the shared-task timeline JSON files (same loader as eaionflux_model.py).

    Each file is one timeline with the shared-task schema: {"timeline_id",
    "timeline_summary", "posts": [...]}; each post has "post_index", "post_id",
    "date", "post", "Post Summary", "Well-being", and "evidence".
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


# ── Evidence / example formatting ────────────────────────────────────────────

def format_span_lines(spans, label: str) -> str:
    """Format one polarity's evidence spans as a labelled bullet list."""
    if not spans:
        return f"{label}: None identified"
    lines = [f"{label}:"]
    for key, category, text in spans:
        if category:
            lines.append(f'- {key} ({ABCD_NAMES.get(key, key)}): "{text}" ({category})')
        else:
            lines.append(f'- {key} ({ABCD_NAMES.get(key, key)}): "{text}"')
    return "\n".join(lines)


def evidence_text_block(post: Post) -> str:
    """Both polarities' gold evidence spans, formatted for a prompt."""
    return (
        format_span_lines(post.evidence["adaptive"], "Adaptive")
        + "\n"
        + format_span_lines(post.evidence["maladaptive"], "Maladaptive")
    )


def span_texts(post: Post, polarity: str) -> list[str]:
    return [t for _, _, t in post.evidence[polarity]]


# ── Leave-one-timeline-out (LOTO) example filtering ──────────────────────────

def loto_candidates(target: Post, db_posts: list[Post]) -> list[Post]:
    """Candidate in-context examples, never from the target post's own timeline.

    Applied by every system that draws examples (Aquarius, BULUSI, EAIonFlux,
    ISM random sampling) for both Task B and Task C.
    """
    return [p for p in db_posts if p.timeline_id != target.timeline_id]


# ── Generation backends ───────────────────────────────────────────────────────

class VLLMBackend:
    """Local-weights vLLM generation with tensor parallelism across GPUs.

    model_path must point at weights already downloaded to the cluster (e.g.
    via huggingface-cli download); vLLM loads them from disk. Set
    HF_HUB_OFFLINE=1 in the environment to guarantee no network calls.
    """

    def __init__(self, model_path: str, tensor_parallel_size: int = 4,
                 dtype: str = "auto", quantization: str | None = None,
                 gpu_memory_utilization: float = 0.9, max_model_len: int | None = None,
                 seed: int | None = None, trust_remote_code: bool = True,
                 temperature: float = 0.1, top_p: float = 1.0, max_tokens: int = 600):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        print(f"[vLLM] Loading model from local weights: {model_path} "
              f"(tensor_parallel_size={tensor_parallel_size}, dtype={dtype}"
              + (f", quantization={quantization}" if quantization else "") + ")")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        # Effective context window (used by drop_until_fits when the caller
        # did not pass --max_model_len, so over-long prompts are shortened
        # instead of crashing a whole vLLM batch).
        try:
            self.max_model_len = self.llm.llm_engine.model_config.max_model_len
        except AttributeError:
            self.max_model_len = max_model_len

    def generate(self, batch_messages: list[list[dict]],
                 sampling_overrides: list[dict] | None = None) -> list[str]:
        """Generate for a batch of chat messages. Each item is a list of
        {"role", "content"} dicts. sampling_overrides (optional, one dict per
        request, e.g. {"seed": r} for ISM's sampling rounds) adjusts
        SamplingParams per request.
        """
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False,
                                                add_generation_prompt=True)
            for m in batch_messages
        ]
        if sampling_overrides:
            from vllm import SamplingParams
            params = [
                SamplingParams(temperature=self.temperature, top_p=self.top_p,
                               max_tokens=self.max_tokens, **ov)
                for ov in sampling_overrides
            ]
            try:
                outputs = self.llm.generate(prompts, params)
            except TypeError:
                # Older vLLM: fall back to shared sampling params.
                outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def count_tokens(self, messages: list[dict]) -> int:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        return len(self.tokenizer.encode(text))


class OllamaBackend:
    """Local ollama daemon generation (single-GPU smoke runs)."""

    def __init__(self, model: str, host: str | None = None,
                 temperature: float = 0.1, max_tokens: int = 600,
                 system_message: str | None = None):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama
        self.model = model
        self.options = {"temperature": temperature, "num_predict": max_tokens}
        self.max_model_len = None
        self.system_message = system_message

    def generate(self, batch_messages: list[list[dict]],
                 sampling_overrides: list[dict] | None = None) -> list[str]:
        out = []
        for messages in batch_messages:
            msgs = ([{"role": "system", "content": self.system_message}]
                    if self.system_message else []) + list(messages)
            resp = self.client.chat(model=self.model, messages=msgs,
                                    options=self.options)
            out.append(resp["message"]["content"].strip())
        return out

    def count_tokens(self, messages: list[dict]) -> int:
        # Rough estimate for budget purposes (~4 chars/token); ollama does not
        # expose a tokenizer.
        text = " ".join(m["content"] for m in messages)
        return len(text) // 4


class MockBackend:
    """Deterministic pseudo-summaries for functional testing (no models).

    Output varies with the prompt hash so candidate-selection logic (ISM's
    KDE) can be exercised.
    """

    def __init__(self, max_tokens: int = 600, max_model_len: int | None = None,
                 **kwargs):
        print("[MockBackend] Using mock generation (no LLM loaded)")
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len

    def generate(self, batch_messages: list[list[dict]],
                 sampling_overrides: list[dict] | None = None) -> list[str]:
        out = []
        for messages in batch_messages:
            text = " ".join(m["content"] for m in messages)
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            out.append(
                f"MOCK SUMMARY {h % 10000}: The self-states in this post "
                f"fluctuate between adaptive and maladaptive (variant {h % 7}). "
                f"The person expresses {['distress', 'hope', 'anger', 'grief'][h % 4]} "
                f"alongside {'connection to others' if h % 2 else 'withdrawal'}, "
                f"and finds it difficult to cope with their situation."
            )
        return out

    def count_tokens(self, messages: list[dict]) -> int:
        text = " ".join(m["content"] for m in messages)
        return len(text) // 4


def make_backend(backend: str, model: str, temperature: float, max_tokens: int,
                 system_message: str | None = None, host: str | None = None,
                 tensor_parallel_size: int = 4, dtype: str = "auto",
                 quantization: str | None = None, gpu_memory_utilization: float = 0.9,
                 max_model_len: int | None = None, seed: int | None = None,
                 top_p: float = 1.0):
    if backend == "vllm":
        return VLLMBackend(model, tensor_parallel_size=tensor_parallel_size,
                           dtype=dtype, quantization=quantization,
                           gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_model_len, seed=seed,
                           temperature=temperature, top_p=top_p,
                           max_tokens=max_tokens)
    if backend == "ollama":
        return OllamaBackend(model, host=host, temperature=temperature,
                             max_tokens=max_tokens, system_message=system_message)
    if backend == "mock":
        return MockBackend(max_tokens=max_tokens, max_model_len=max_model_len)
    raise ValueError(f"Unknown backend: {backend}")


# ── Embedding models (retrieval / KDE) ───────────────────────────────────────

class MockEmbedder:
    """Deterministic pseudo-embeddings (hash-seeded, L2-normalized) so the
    retrieval and KDE code paths can be smoke-tested without model weights."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False,
               **kwargs):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(self.dim)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out)


class MockCrossEncoder:
    """Deterministic pseudo cross-encoder scores for smoke tests."""

    def predict(self, pairs):
        return np.array([
            int(hashlib.md5((a + "||" + b).encode("utf-8")).hexdigest()[:8], 16) / 2**32
            for a, b in pairs])


def make_embedder(model_name: str, mock: bool = False, device: str | None = None):
    """SentenceTransformer embedder, or a MockEmbedder when mock=True."""
    if mock:
        print(f"[MockEmbedder] Standing in for {model_name}")
        return MockEmbedder()
    from sentence_transformers import SentenceTransformer
    print(f"[Embedding] Loading {model_name}" + (f" on {device}" if device else ""))
    return SentenceTransformer(model_name, device=device)


def free_gpu_memory():
    """Release cached GPU memory after retrieval models are deleted, so the
    vLLM engine loaded afterwards can claim it."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def timeline_text(tl: Timeline, sep: str = "\n\n") -> str:
    """All posts of a timeline concatenated in chronological order."""
    return sep.join(p.text for p in tl.posts)


# ── Resumable checkpointing ──────────────────────────────────────────────────

class Checkpoint:
    """JSONL checkpoint of per-post results, appended after every batch and
    skipped on restart. Key = (timeline_id, post_id); stored as a 2-list."""

    def __init__(self, path: str):
        self.path = path
        self.records: dict[tuple, dict] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.records[tuple(rec["_key"])] = rec
            print(f"[Checkpoint] Resuming: {len(self.records)} completed posts "
                  f"found in {path}")

    def has(self, timeline_id: str, post_id: str) -> bool:
        return (timeline_id, post_id) in self.records

    def get(self, timeline_id: str, post_id: str) -> dict | None:
        return self.records.get((timeline_id, post_id))

    def append(self, timeline_id: str, post_id: str, record: dict):
        record = {"_key": [timeline_id, post_id], "timeline_id": timeline_id,
                  "post_id": post_id, **record}
        self.records[(timeline_id, post_id)] = record
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Official submission-format writer ────────────────────────────────────────

def write_task_b_submission(timelines: list[Timeline],
                            records: dict[tuple, dict] | Checkpoint,
                            out_path: str):
    """Write the official submission-format JSON (same structure as dev.json /
    the gold data): {timeline_id: {"timeline_level": {"summary": ""},
    "post_level": {post_id: {"adaptive_evidence", "maladaptive_evidence",
    "summary", "wellbeing_score}}}}.

    Gold evidence spans and gold well-being scores are filled in (the repo
    convention: Task A outputs are gold), so both the official run.py and
    eval/task31_eval.py can score the file directly. Posts without a generated
    summary get an empty template (they are skipped by the scorers or scored
    at the empty-prediction floor).
    """
    if isinstance(records, Checkpoint):
        records = records.records
    sub = {}
    for tl in timelines:
        sub[tl.timeline_id] = {"timeline_level": {"summary": ""}, "post_level": {}}
        for p in tl.posts:
            rec = records.get((tl.timeline_id, p.post_id))
            sub[tl.timeline_id]["post_level"][p.post_id] = {
                "adaptive_evidence": span_texts(p, "adaptive"),
                "maladaptive_evidence": span_texts(p, "maladaptive"),
                "summary": (rec or {}).get("summary", "") or "",
                "wellbeing_score": p.wellbeing,
            }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=2, ensure_ascii=False)
    n = sum(1 for r in records.values() if r.get("summary"))
    print(f"[Submission] Wrote {out_path} ({n} generated summaries; "
          f"{sum(len(t.posts) for t in timelines)} posts total)")


def write_task_c_submission(timelines: list[Timeline],
                            records: dict[tuple, dict] | Checkpoint,
                            out_path: str):
    """Write the official submission-format JSON for Task C (timeline-level):
    {timeline_id: {"timeline_level": {"summary": predicted},
    "post_level": {post_id: empty template}}}.

    Post-level blocks are filled with gold evidence/well-being and an empty
    summary (the same structure as the repo's existing c_*.json submissions);
    score with run.py --tasks C.
    """
    if isinstance(records, Checkpoint):
        records = records.records
    sub = {}
    for tl in timelines:
        rec = records.get((tl.timeline_id, "__timeline__"))
        sub[tl.timeline_id] = {
            "timeline_level": {"summary": (rec or {}).get("summary", "") or ""},
            "post_level": {},
        }
        for p in tl.posts:
            sub[tl.timeline_id]["post_level"][p.post_id] = {
                "adaptive_evidence": span_texts(p, "adaptive"),
                "maladaptive_evidence": span_texts(p, "maladaptive"),
                "summary": "",
                "wellbeing_score": p.wellbeing,
            }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=2, ensure_ascii=False)
    n = sum(1 for r in records.values() if r.get("summary"))
    print(f"[Submission] Wrote {out_path} ({n} timeline summaries)")


# ── Generation-settings log ─────────────────────────────────────────────────

def write_settings_log(out_path: str, title: str,
                       settings: list[tuple[str, object, str]],
                       notes: list[str]):
    """Write a Markdown log of every generation/retrieval setting.

    settings: (name, value, provenance) triples where provenance says whether
    the value is from the paper ("paper") or our documented default
    ("our default — paper silent"). This log is the source for the paper's
    reimplementation section.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# {title}", "", f"Run started: {ts}", "",
             "| Setting | Value | Source |", "|---|---|---|"]
    for name, value, prov in settings:
        lines.append(f"| {name} | `{value}` | {prov} |")
    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in notes]
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Settings log] Wrote {out_path}")


# ── Prompt token-budget guard ────────────────────────────────────────────────

def prompt_token_count(backend, messages: list[dict]) -> int:
    return backend.count_tokens(messages)


def drop_until_fits(build, backend, n_examples: int, max_model_len: int | None,
                    max_tokens: int, min_examples: int = 0) -> tuple[list, int]:
    """Repeatedly rebuild `build(k)` with fewer examples (k = n_examples-1, ...)
    until the prompt fits the context window. Returns (prompt_inputs, k_used).

    `build(k)` must return the object passed to backend.generate for a k-example
    prompt (a messages list). If even min_examples does not fit, returns the
    min_examples variant anyway (vLLM will raise if it truly cannot fit —
    better to fail loudly than silently skip posts).

    If max_model_len is None, the backend's own context window is used when it
    exposes one (VLLMBackend does), so the guard is active by default on the
    cluster. Callers must record k_used so shortened prompts are reported.
    """
    if max_model_len is None:
        max_model_len = getattr(backend, "max_model_len", None)
    if max_model_len is None:
        return build(n_examples), n_examples
    k = n_examples
    budget = max_model_len - max_tokens - 64
    while k > min_examples:
        candidate = build(k)
        if prompt_token_count(backend, candidate) <= budget:
            return candidate, k
        k -= 1
    return build(min_examples), min_examples


# ── Shared CLI ───────────────────────────────────────────────────────────────

def add_common_args(parser, default_model: str, default_output: str,
                    default_temperature: float = 0.1,
                    default_batch_size: int = 16):
    """Arguments every system script shares (data, backend, decoding, task)."""
    parser.add_argument("--data_dir", required=True,
                        help="Directory of shared-task timeline JSON files.")
    parser.add_argument("--output_dir", default=default_output)
    parser.add_argument("--task", default="bc", choices=["b", "c", "bc"],
                        help="b = post-level, c = timeline-level, bc = both "
                             "(one model load; Task B runs first).")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "ollama", "mock"],
                        help="mock = no LLM and stand-in embedding models "
                             "(smoke tests only).")
    parser.add_argument("--model", default=default_model,
                        help=f"Local weights path or id. Default: {default_model}.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Context window. Default: the model's own (read "
                             "from vLLM); the prompt-length guard uses it.")
    parser.add_argument("--temperature", type=float, default=default_temperature)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N targets (posts for B, timelines "
                             "for C) — smoke testing.")


def backend_from_args(args):
    backend = make_backend(
        args.backend, args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype, quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, seed=args.seed, top_p=args.top_p,
    )
    return backend


def common_settings(args) -> list[tuple[str, object, str]]:
    """Settings-log rows shared by every system."""
    return [
        ("backend", args.backend, "our setup (vLLM, local weights)"),
        ("tensor_parallel_size", args.tensor_parallel_size, "our setup"),
        ("max_model_len", args.max_model_len or "model default",
         "our setup (prompt-length guard reads the effective window)"),
        ("max_tokens", args.max_tokens, "our default — paper silent"),
        ("top_p", args.top_p, "our default — paper silent"),
        ("seed", args.seed, "our setup"),
    ]


# ── Timeline-level (Task C) prompt-length guard ─────────────────────────────

# Per-post word caps tried, in order, once in-context examples have been
# reduced to the minimum. Truncating every post (rather than dropping the
# oldest posts) keeps the whole trajectory — beginning, middle, end — visible,
# which is what the Task C prompts ask the model to describe.
POST_WORD_CAPS = [None, 400, 200, 100, 50, 25]


def truncate_words(text: str, cap: int | None) -> str:
    if cap is None:
        return text
    words = text.split()
    if len(words) <= cap:
        return text
    return " ".join(words[:cap]) + " [...]"


def fit_timeline_prompt(build, backend, n_examples: int,
                        max_model_len: int | None, max_tokens: int,
                        min_examples: int = 0, keep_examples_down_to_cap: int = 100):
    """Fit a timeline-level prompt into the context window.

    `build(k, cap)` returns the messages for k in-context examples with every
    post text (examples and target) truncated to `cap` words (None = full).

    Order of shortening: with all n_examples kept, post texts are truncated
    through POST_WORD_CAPS down to `keep_examples_down_to_cap` words; only if
    that still does not fit is an example dropped and the caps retried. At
    min_examples the caps continue down to the smallest. The in-context
    examples define each system, so they are kept as long as posts can stay
    reasonably long. Returns (messages, k_used, word_cap_used); callers record
    both so any shortened prompt is reported.
    """
    if max_model_len is None:
        max_model_len = getattr(backend, "max_model_len", None)
    if max_model_len is None:
        return build(n_examples, None), n_examples, None
    budget = max_model_len - max_tokens - 64
    soft_caps = [c for c in POST_WORD_CAPS
                 if c is None or c >= keep_examples_down_to_cap]
    for k in range(n_examples, min_examples - 1, -1):
        caps = POST_WORD_CAPS if k == min_examples else soft_caps
        for cap in caps:
            msgs = build(k, cap)
            if prompt_token_count(backend, msgs) <= budget:
                return msgs, k, cap
    return build(min_examples, POST_WORD_CAPS[-1]), min_examples, POST_WORD_CAPS[-1]


def task_c_targets(timelines: list[Timeline], limit: int | None = None) -> list[Timeline]:
    """Timelines with a gold timeline summary (the ones Task C is scored on)."""
    out = [tl for tl in timelines if tl.summary]
    return out[:limit] if limit else out
