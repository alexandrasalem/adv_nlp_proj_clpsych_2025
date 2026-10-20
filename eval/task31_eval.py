"""CLPsych 2026 Task 3.1-style evaluation, adapted for CLPsych 2025 Task B results.

Recreates the CLPsych 2026 shared task Task 3.1 evaluation (sequence-summary
quality) over this repo's Task B post-level summaries, so the CLPsych 2025
systems can be ranked with the 2026 metric suite.

Source of the evaluation logic (fetched 2026-10-19):
    Maria-Liakata-NLP-Group/CLPsych-2026, evaluation/Task3.1_Scripts/
    (evaluate_task3.py, nli_scorer.py, bertscore_scorer.py,
     TASK3_1_EVALUATION_LOGIC.md)
That code expects CLPsych 2026 Task 3.1 sequence predictions
((timeline_id, sequence_id) -> summary) scored against 2026 gold summaries.
This adaptation keeps the metric definitions, models, edge cases, and the
350-word truncation rule identical, but reads CLPsych 2025 Task B data:

    * systems = official submission-format JSONs
      ({timeline_id: {"timeline_level": {...}, "post_level": {post_id:
        {"summary", ...}}}}), i.e. exactly what clpsych2025/evaluation/run.py
      and edit_output_for_official_eval.py produce/consume;
    * gold = clpsych2025/evaluation/data/dev.json
      (post_level[post_id]["summary_sents"] per run.py);
    * each post with a non-empty gold summary is one "sequence".

Metrics (identical to Task 3.1):
    CS   = 1 - mean contradiction       (DeBERTa-v3-large-mnli-fever-anli-ling-wanli,
                                         gold sentence = premise, prediction = hypothesis)
                                       higher = better
    CT   = mean per-predicted-sentence MAX contradiction
                                       lower = better
    ROUGE-L Recall  (rouge-score pkg, Porter stemmer)   higher = better
    BERTScore Recall (per-gold-sentence max F1 against predicted sentences,
                      microsoft/deberta-xlarge-mnli, rescale_with_baseline=True)
                                       higher = better

Predictions are truncated to the first 350 words before ANY scoring.

Ranking (per user / shared-task style):
    1. rank systems on each of the four metrics (ties -> average rank);
    2. average the four RANKS per system; lower mean rank = better.
    The official CLPsych-2026 Codabench key `t3_rank` is just `t3_cs`
    (Task 3.1 Ranking = t3_cs, higher = better) — this is also reported for
    reference, but the headline ranking here is the rank-average.

Usage:
    python eval/task31_eval.py                       # scores every *.json in
                                                    # clpsych2025/evaluation/submissions_dev
    python eval/task31_eval.py --submissions_dir clpsych2025/evaluation/official_run
    python eval/task31_eval.py --submissions path/to/BLUE.json --no_constant
    python eval/task31_eval.py --mock               # no models, functional test
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

MAX_PRED_WORDS = 350

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Label order for this model: 0=entailment, 1=neutral, 2=contradiction
CONTRADICTION_IDX = 2
BERTSCORE_MODEL_NAME = "microsoft/deberta-xlarge-mnli"

# The repo's degenerate constant string (see PROJECT_CONTEXT.md §3.1).
DEFAULT_CONSTANT_TEXT = (
    "The self-states in this timeline fluctuate between adaptive and "
    "maladaptive, this person finds it difficult to cope with their situation."
)


# ---------------------------------------------------------------------------
# Text utilities (from Task3.1_Scripts/evaluate_task3.py)
# ---------------------------------------------------------------------------

def truncate_to_words(text, max_words=MAX_PRED_WORDS):
    """Truncate text to first max_words words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def sentence_tokenize(text):
    """Split text into sentences using NLTK."""
    import nltk
    if not text or not text.strip():
        return []
    return [s.strip() for s in nltk.sent_tokenize(text.strip()) if s.strip()]


def ensure_nltk_data():
    """Download NLTK punkt tokenizer if not present."""
    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find("tokenizers/" + resource)
        except (LookupError, OSError):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ROUGE-L Recall (Google rouge-score package, Porter stemmer)
# ---------------------------------------------------------------------------

_ROUGE_SCORER = None


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def rouge_l_recall(gold_text, pred_text):
    if not gold_text or not pred_text:
        return 0.0
    scorer = _get_rouge_scorer()
    return float(scorer.score(gold_text, pred_text)["rougeL"].recall)


# ---------------------------------------------------------------------------
# NLI scorer (CS / CT) — ported from Task3.1_Scripts/nli_scorer.py
# ---------------------------------------------------------------------------

class MockNLIScorer:
    """Deterministic mock scorer for functional testing without the NLI model
    (same pseudo-scoring as the official Task3.1_Scripts mock)."""

    def __init__(self):
        print("[MockNLIScorer] Using mock scorer (no NLI model loaded)")
        ensure_nltk_data()

    @staticmethod
    def _pseudo_contradiction(premise, hypothesis):
        import hashlib
        p_set = set(premise.lower().split())
        h_set = set(hypothesis.lower().split())
        if not p_set or not h_set:
            return 0.5
        overlap = len(p_set & h_set) / max(len(p_set | h_set), 1)
        h = int(hashlib.md5((premise + hypothesis).encode()).hexdigest()[:8], 16)
        noise = (h % 100) / 1000.0
        return max(0.0, min(1.0, (1.0 - overlap) * 0.6 + noise))

    def score_sequence(self, gold_sents, pred_sents):
        if not pred_sents:
            return {"cs": 0.0, "ct": 1.0}
        if not gold_sents:
            return {"cs": 1.0, "ct": 0.0}
        n_pred, n_gold = len(pred_sents), len(gold_sents)
        prob_matrix = np.zeros((n_pred, n_gold))
        for pi, ps in enumerate(pred_sents):
            for gi, gs in enumerate(gold_sents):
                prob_matrix[pi, gi] = self._pseudo_contradiction(gs, ps)
        cs = float(1.0 - prob_matrix.mean(axis=1).mean())
        ct = float(prob_matrix.max(axis=1).mean())
        return {"cs": cs, "ct": ct}


class NLIScorer:
    """Computes NLI contradiction scores between gold and predicted summaries.

    CS (Consistency Score) = 1 - mean contradiction probability (higher=better):
        CS = 1/(|S||G|) * sum_s sum_g (1 - NLI(Contradict|g, s))
    CT (Contradiction Top) = mean over predicted sentences of the max
    contradiction probability across gold sentences (lower=better).

    NLI direction: gold sentence = premise, predicted sentence = hypothesis.
    """

    def __init__(self, model_name=NLI_MODEL_NAME, device=None, batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        print("Loading NLI model: %s (device: %s)" % (model_name, self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        ensure_nltk_data()

    def _get_contradiction_probs(self, premise_hypothesis_pairs):
        import torch

        if not premise_hypothesis_pairs:
            return []

        all_probs = []
        for i in range(0, len(premise_hypothesis_pairs), self.batch_size):
            batch = premise_hypothesis_pairs[i: i + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                contradiction_probs = probs[:, CONTRADICTION_IDX].cpu().numpy()
                all_probs.extend(contradiction_probs.tolist())

        return all_probs

    def score_sequence(self, gold_sents, pred_sents):
        if not pred_sents:
            # Empty prediction: worst possible scores
            return {"cs": 0.0, "ct": 1.0}
        if not gold_sents:
            # Empty gold (shouldn't happen): no contradiction measurable
            return {"cs": 1.0, "ct": 0.0}

        pairs = []
        pair_indices = []
        for pi, ps in enumerate(pred_sents):
            for gi, gs in enumerate(gold_sents):
                pairs.append((gs, ps))  # premise=gold, hypothesis=pred
                pair_indices.append((pi, gi))

        contra_probs = self._get_contradiction_probs(pairs)

        n_pred = len(pred_sents)
        n_gold = len(gold_sents)
        prob_matrix = np.zeros((n_pred, n_gold))
        for idx, (pi, gi) in enumerate(pair_indices):
            prob_matrix[pi, gi] = contra_probs[idx]

        # CS: consistency = 1 - mean contradiction (higher = better)
        cs = float(1.0 - prob_matrix.mean(axis=1).mean())
        # CT: max contradiction per predicted sentence, averaged (lower = better)
        ct = float(prob_matrix.max(axis=1).mean())

        return {"cs": cs, "ct": ct}


# ---------------------------------------------------------------------------
# BERTScore Recall — ported from Task3.1_Scripts/bertscore_scorer.py
# ---------------------------------------------------------------------------

class MockBERTScoreScorer:
    """Deterministic mock for functional testing without the BERTScore model."""

    def __init__(self):
        print("[MockBERTScoreScorer] Using mock scorer (no BERTScore model loaded)")

    @staticmethod
    def _pseudo_bertscore(gold, pred):
        import hashlib
        g_set = set(gold.lower().split())
        p_set = set(pred.lower().split())
        if not g_set or not p_set:
            return 0.0
        overlap = len(g_set & p_set) / max(len(g_set | p_set), 1)
        h = int(hashlib.md5((gold + pred).encode()).hexdigest()[:8], 16)
        noise = (h % 100) / 2000.0
        return min(1.0, overlap * 0.8 + 0.1 + noise)

    def score_sequence(self, gold_sents, pred_sents):
        if not pred_sents or not gold_sents:
            return {"bertscore_recall": 0.0}
        recalls = []
        for gs in gold_sents:
            scores = [self._pseudo_bertscore(gs, ps) for ps in pred_sents]
            recalls.append(max(scores))
        return {"bertscore_recall": float(np.mean(recalls))}


class BERTScoreScorer:
    """Computes BERTScore Recall between gold and predicted summary sentences.

    For each gold sentence, compute BERTScore F1 against all predicted sentences,
    take the max. Average across gold sentences.
    """

    def __init__(self, model_name=BERTSCORE_MODEL_NAME, rescale_with_baseline=True):
        from bert_score import BERTScorer

        self.model_name = model_name
        self.rescale_with_baseline = rescale_with_baseline

        print("Loading BERTScore model: %s" % model_name)
        self.scorer = BERTScorer(
            model_type=self.model_name,
            lang="en",
            rescale_with_baseline=self.rescale_with_baseline,
        )

    def score_sequence(self, gold_sents, pred_sents):
        if not pred_sents or not gold_sents:
            return {"bertscore_recall": 0.0}

        recalls = []
        for gold_sent in gold_sents:
            # Score each predicted sentence against this gold sentence
            _, _, F = self.scorer.score(pred_sents, [gold_sent] * len(pred_sents))
            recalls.append(F.max().item())

        return {"bertscore_recall": float(np.mean(recalls))}


# ---------------------------------------------------------------------------
# Data loading (CLPsych 2025 Task B, gold via the official scorer's schema)
# ---------------------------------------------------------------------------

def load_gold_post_summaries(gold_file):
    """Load gold post-level summary sentences from dev.json.

    Mirrors clpsych2025/evaluation/run.py: iterate
    gold[timeline]["timeline_level"]["post_ids"]; a post is evaluated only if
    gold[timeline]["post_level"][post_id]["summary_sents"] is non-empty.

    Returns list of (timeline_id, post_id, gold_sents) in stable order.
    """
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_posts = []
    for timeline_id in sorted(gold_data.keys()):
        datum = gold_data[timeline_id]
        for post_id in datum["timeline_level"]["post_ids"]:
            gold_sents = datum["post_level"][post_id].get("summary_sents", [])
            if gold_sents:
                gold_posts.append((timeline_id, post_id, gold_sents))
    return gold_posts


def load_submission_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def system_name_from_path(path):
    """run.py's parse_filename: 'BLUE.json' -> 'BLUE_BLUE'; anything else ->
    the file stem. For the ranking table we keep the full stem (unique)."""
    return os.path.splitext(os.path.basename(path))[0]


def build_pseudo_submissions(gold_posts, constant_text):
    """Pseudo-systems scored like real submissions (summary per gold post).

    gold     -> the gold summary itself (sanity anchor; CS/ROUGE/BERTScore ~ 1)
    constant -> the repo's degenerate constant string for every post
    empty    -> empty string for every post (the metric floor)
    """
    gold_sub = {}
    constant_sub = {}
    empty_sub = {}
    for timeline_id, post_id, gold_sents in gold_posts:
        gold_sub.setdefault(timeline_id, {"post_level": {}})["post_level"][post_id] = {
            "summary": " ".join(gold_sents)
        }
        constant_sub.setdefault(timeline_id, {"post_level": {}})["post_level"][post_id] = {
            "summary": constant_text
        }
        empty_sub.setdefault(timeline_id, {"post_level": {}})["post_level"][post_id] = {
            "summary": ""
        }
    return {"gold": gold_sub, "constant": constant_sub, "empty": empty_sub}


def get_pred_summary(submission, timeline_id, post_id):
    """Fetch a post's predicted summary; missing post/timeline -> empty string."""
    try:
        return submission[timeline_id]["post_level"][post_id].get("summary", "") or ""
    except (KeyError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# Per-system evaluation
# ---------------------------------------------------------------------------

def evaluate_system(system_name, submission, gold_posts, nli, bsc,
                    enable_rouge=True, enable_bertscore=True, limit=None):
    """Score one system against the gold posts. Returns (per_post_records, means)."""
    records = []
    n_truncated = 0
    n_missing = 0

    eval_posts = gold_posts if limit is None else gold_posts[:limit]

    for timeline_id, post_id, gold_sents in eval_posts:
        raw_pred = get_pred_summary(submission, timeline_id, post_id)
        if not raw_pred:
            n_missing += 1

        pred_summary = truncate_to_words(raw_pred, MAX_PRED_WORDS)
        truncated = len(raw_pred.split()) > MAX_PRED_WORDS
        if truncated:
            n_truncated += 1

        pred_sents = sentence_tokenize(pred_summary)

        nli_scores = nli.score_sequence(gold_sents, pred_sents)
        rec = {
            "system": system_name,
            "timeline_id": timeline_id,
            "post_id": post_id,
            "cs": nli_scores["cs"],
            "ct": nli_scores["ct"],
            "n_gold_sents": len(gold_sents),
            "n_pred_sents": len(pred_sents),
            "pred_words": len(pred_summary.split()),
            "truncated": int(truncated),
        }
        if enable_rouge:
            rec["rouge_l_recall"] = rouge_l_recall(" ".join(gold_sents), pred_summary)
        if enable_bertscore:
            rec["bertscore_recall"] = bsc.score_sequence(gold_sents, pred_sents)["bertscore_recall"]
        records.append(rec)

    df = pd.DataFrame(records)
    means = {"system": system_name, "n_posts": len(records)}
    means["mean_cs"] = float(df["cs"].mean())
    means["mean_ct"] = float(df["ct"].mean())
    if enable_rouge:
        means["mean_rouge_l_recall"] = float(df["rouge_l_recall"].mean())
    if enable_bertscore:
        means["mean_bertscore_recall"] = float(df["bertscore_recall"].mean())
    if n_truncated:
        print("  [%s] truncated %d/%d predictions to %d words" % (
            system_name, n_truncated, len(records), MAX_PRED_WORDS))
    if n_missing:
        print("  [%s] WARNING: %d/%d gold posts have no/empty predicted summary" % (
            system_name, n_missing, len(records)))
    return records, means


# ---------------------------------------------------------------------------
# Ranking: per-metric ranks, then the average of the ranks
# ---------------------------------------------------------------------------

METRIC_COLS = {
    # result column -> (means column, higher_is_better)
    "rank_cs": ("mean_cs", True),
    "rank_ct": ("mean_ct", False),          # lower = better
    "rank_rouge": ("mean_rouge_l_recall", True),
    "rank_bert": ("mean_bertscore_recall", True),
}


def add_ranks(by_system, enable_rouge=True, enable_bertscore=True):
    """Rank systems per metric (ties -> average rank, rank 1 = best), then
    average the ranks. Also report the official t3_rank (= t3_cs) reference."""
    active = [(rcol, mcol, higher)
              for rcol, (mcol, higher) in METRIC_COLS.items()
              if (rcol != "rank_rouge" or enable_rouge)
              and (rcol != "rank_bert" or enable_bertscore)]
    for rcol, mcol, higher in active:
        by_system[rcol] = by_system[mcol].rank(ascending=not higher, method="average")
    rank_cols = [rcol for rcol, _, _ in active]
    by_system["mean_rank"] = by_system[rank_cols].mean(axis=1)
    # Final ranking: lowest average rank = best; ties share the average rank.
    by_system["final_rank"] = by_system["mean_rank"].rank(method="average")
    # Official CLPsych-2026 Task 3.1 ranking key: t3_rank = t3_cs (the CS
    # value itself, higher = better). Reported for reference.
    by_system["t3_rank_cs"] = by_system["mean_cs"]
    return by_system.sort_values("final_rank")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLPsych 2026 Task 3.1-style evaluation over CLPsych 2025 Task B submissions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--submissions_dir", default="clpsych2025/evaluation/submissions_dev",
                        help="Directory of submission-format JSONs to score (default: "
                             "clpsych2025/evaluation/submissions_dev). Ignored if "
                             "--submissions is given.")
    parser.add_argument("--submissions", nargs="+", default=None,
                        help="Explicit submission JSON file(s) to score (overrides "
                             "--submissions_dir).")
    parser.add_argument("--pattern", default="*",
                        help="Glob pattern for files in --submissions_dir (default: all *.json).")
    parser.add_argument("--gold_file", default="clpsych2025/evaluation/data/dev.json",
                        help="Gold dev.json (same file the official scorer uses).")
    parser.add_argument("--output_dir", default="outputs/task31_eval",
                        help="Where to write the per-post / per-system CSVs.")
    parser.add_argument("--device", default=None,
                        help="Device for the NLI model (cuda/mps/cpu, auto-detected if omitted).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for NLI inference (default: 32).")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock scorers (no models, for functional testing).")
    parser.add_argument("--no_rouge", action="store_true",
                        help="Disable ROUGE-L Recall (faster evaluation).")
    parser.add_argument("--no_bertscore", action="store_true",
                        help="Disable BERTScore Recall (skips the ~1.4GB model).")
    parser.add_argument("--no_gold", action="store_true",
                        help="Exclude the gold-summary pseudo-system from the ranking.")
    parser.add_argument("--no_constant", action="store_true",
                        help="Exclude the constant-string pseudo-system from the ranking.")
    parser.add_argument("--no_empty", action="store_true",
                        help="Exclude the empty-string pseudo-system from the ranking.")
    parser.add_argument("--constant_text", default=DEFAULT_CONSTANT_TEXT,
                        help="Constant string for the constant pseudo-system.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N gold posts (smoke testing).")
    args = parser.parse_args()

    enable_rouge = not args.no_rouge
    enable_bertscore = not args.no_bertscore

    # --- Collect systems ---------------------------------------------------
    if args.submissions:
        sub_paths = args.submissions
    else:
        sub_paths = sorted(
            glob.glob(os.path.join(args.submissions_dir, args.pattern + ".json"))
            if not args.pattern.endswith(".json")
            else glob.glob(os.path.join(args.submissions_dir, args.pattern))
        )
    if not sub_paths:
        print("ERROR: no submission files found.", file=sys.stderr)
        sys.exit(1)

    gold_posts = load_gold_post_summaries(args.gold_file)
    print("Gold posts with summaries: %d (from %s)" % (len(gold_posts), args.gold_file))
    if args.limit:
        print("Limiting to first %d gold posts (--limit)." % args.limit)

    systems = []
    for path in sub_paths:
        name = system_name_from_path(path)
        systems.append((name, load_submission_file(path)))
        print("Loaded submission: %-40s (%s)" % (name, path))
    for name, sub in build_pseudo_submissions(gold_posts, args.constant_text).items():
        if (name == "gold" and not args.no_gold) or \
           (name == "constant" and not args.no_constant) or \
           (name == "empty" and not args.no_empty):
            systems.append((name, sub))
            print("Added pseudo-system: %s" % name)

    # --- Scorers ------------------------------------------------------------
    ensure_nltk_data()
    if args.mock:
        nli = MockNLIScorer()
        bsc = MockBERTScoreScorer() if enable_bertscore else None
    else:
        nli = NLIScorer(device=args.device, batch_size=args.batch_size)
        bsc = BERTScoreScorer() if enable_bertscore else None

    # --- Evaluate ------------------------------------------------------------
    all_records = []
    all_means = []
    for name, submission in systems:
        print("\nScoring system: %s" % name)
        records, means = evaluate_system(
            name, submission, gold_posts, nli, bsc,
            enable_rouge=enable_rouge, enable_bertscore=enable_bertscore,
            limit=args.limit,
        )
        all_records.extend(records)
        all_means.append(means)

    per_post = pd.DataFrame(all_records)
    by_system = pd.DataFrame(all_means)
    by_system = add_ranks(by_system, enable_rouge, enable_bertscore)

    # --- Write outputs -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    per_post_path = os.path.join(args.output_dir, f"task31_per_post_{ts}.csv")
    by_system_path = os.path.join(args.output_dir, f"task31_by_system_{ts}.csv")
    per_post.to_csv(per_post_path, index=False)
    by_system.to_csv(by_system_path, index=False)

    # --- Print summary -------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("TASK 3.1-STYLE EVALUATION — Task B systems ranked by average rank "
          "(CS, CT, ROUGE-L recall, BERTScore recall)")
    print("=" * 100)
    show = by_system.copy()
    for col in ["mean_cs", "mean_ct", "mean_rouge_l_recall", "mean_bertscore_recall"]:
        if col in show.columns:
            show[col] = show[col].map(lambda v: f"{v:.4f}")
    for col in ["rank_cs", "rank_ct", "rank_rouge", "rank_bert", "mean_rank"]:
        if col in show.columns:
            show[col] = show[col].map(lambda v: f"{v:.2f}")
    print(show.to_string(index=False))
    print("\nRank 1 = best. CS/ROUGE-L/BERTScore: higher = better; CT: lower = better.")
    print("mean_rank = average of the four per-metric ranks (ties -> average rank).")
    print("final_rank orders systems by mean_rank; t3_rank_cs is the official "
          "CLPsych-2026 key (= mean CS, higher = better).")
    print("\nPer-post scores:   %s" % per_post_path)
    print("System rankings:   %s" % by_system_path)


if __name__ == "__main__":
    main()
