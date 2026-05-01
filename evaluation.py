"""
CLPsych 2025 Evaluation Metrics
================================

Evaluation metrics for the subtasks our pipeline performs:

    Task A.3 — ABCD classification of evidence spans
               Metric: accuracy + macro-F1 over the 6 ABCD keys
               (also stratified by polarity and a full classification report)

    Task B   — Post-Level Summarization
               Metrics: Consistency (CS), Contradiction (CT),
               Evidence Alignment (EA), all computed via NLI
               (DeBERTa-v3-large-mnli-fever-anli-ling-wanli).

    Task C   — Timeline-Level Summarization
               Metrics: Consistency (CS), Contradiction (CT)

Task A.1 (Evidence Extraction) and Task A.2 (Well-being Score Prediction) are
intentionally NOT evaluated here: our pipeline uses gold evidence spans and
gold well-being scores as inputs and does not generate predictions for those
tasks.

Usage:
    # Task A.3 from pipeline.py output
    a3 = evaluator.evaluate_task_a3("./outputs/task_a3_predictions.json")

    # Task B from pipeline.py output
    b = evaluator.evaluate_pipeline_output(
        "./outputs/task_b_results.json", mode="both"
    )

Requirements:
    pip install torch transformers nltk scikit-learn --break-system-packages
"""

import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Configuration ────────────────────────────────────────────────────────────

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_LABELS = ["entailment", "neutral", "contradiction"]

ABCD_KEYS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]


# ── Result Containers ────────────────────────────────────────────────────────

@dataclass
class TaskA3Results:
    accuracy: float
    macro_f1: float
    accuracy_adaptive: float
    accuracy_maladaptive: float
    n_spans: int
    n_unknown: int
    report_str: str

    def __str__(self):
        return (
            f"Task A.3 Results:\n"
            f"  Accuracy:              {self.accuracy:.4f}\n"
            f"  Macro F1:              {self.macro_f1:.4f}\n"
            f"  Accuracy (adaptive):   {self.accuracy_adaptive:.4f}\n"
            f"  Accuracy (maladaptive):{self.accuracy_maladaptive:.4f}\n"
            f"  n_spans: {self.n_spans}  n_unknown: {self.n_unknown}\n"
            f"\n{self.report_str}"
        )


@dataclass
class TaskBResults:
    consistency: float
    contradiction: float
    evidence_alignment: float
    n_posts: int
    per_post_cs: list
    per_post_ct: list
    per_post_ea: list

    def __str__(self):
        return (
            f"Task B Results:\n"
            f"  Consistency (CS):        {self.consistency:.4f}\n"
            f"  Contradiction (CT):      {self.contradiction:.4f}\n"
            f"  Evidence Alignment (EA): {self.evidence_alignment:.4f}\n"
            f"  n_posts: {self.n_posts}"
        )


@dataclass
class TaskCResults:
    consistency: float
    contradiction: float
    n_timelines: int
    per_timeline_cs: list
    per_timeline_ct: list

    def __str__(self):
        return (
            f"Task C Results:\n"
            f"  Consistency (CS):   {self.consistency:.4f}\n"
            f"  Contradiction (CT): {self.contradiction:.4f}\n"
            f"  n_timelines: {self.n_timelines}"
        )


# ── Sentence Splitter ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences (nltk punkt with regex fallback)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)
    except ImportError:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]


# ── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluator for CLPsych 2025 subtasks A.3, B, and C.

    The NLI model is loaded lazily on first use of any Task B / Task C metric.
    Task A.3 does not require the NLI model.
    """

    def __init__(self, device: str = None, batch_size: int = 16):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size

        self._nli_model = None
        self._nli_tokenizer = None

        print(f"Evaluator initialized (device={self.device})")

    # ── NLI Setup ────────────────────────────────────────────────────────

    def _load_nli(self):
        if self._nli_model is not None:
            return
        print(f"Loading NLI model: {NLI_MODEL}")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self._nli_model.to(self.device)
        self._nli_model.eval()
        print("NLI model loaded")

    def _nli_predict(
        self, premises: list[str], hypotheses: list[str]
    ) -> np.ndarray:
        """Returns array (n_pairs, 3) with [entailment, neutral, contradiction] probs."""
        self._load_nli()
        all_probs = []
        n = len(premises)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            inputs = self._nli_tokenizer(
                premises[start:end], hypotheses[start:end],
                return_tensors="pt", truncation=True, padding=True, max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = self._nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def _get_nli_label_index(self, label: str) -> int:
        self._load_nli()
        id2label = self._nli_model.config.id2label
        for idx, name in id2label.items():
            if label.lower() in name.lower():
                return idx
        raise ValueError(f"Label '{label}' not found in: {id2label}")

    # ── Task A.3: ABCD Classification ────────────────────────────────────

    def evaluate_task_a3(self, predictions_path: str) -> TaskA3Results:
        """
        Evaluate Task A.3 from the JSON pipeline.py writes to
        `<output_dir>/task_a3_predictions.json`.

        Expected JSON shape:
            {
                "spans": [
                    {"gold": "A", "predicted": "B-O", "polarity": "adaptive", ...},
                    ...
                ]
            }

        Computes accuracy, macro-F1 over the 6 ABCD keys, and per-polarity
        accuracy. Spans predicted as "UNKNOWN" count as incorrect.
        """
        print(f"\nEvaluating Task A.3 from {predictions_path}")
        with open(predictions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        spans = data.get("spans", [])
        if not spans:
            raise ValueError(f"No spans found in {predictions_path}")

        gold = [s["gold"] for s in spans]
        pred = [s["predicted"] for s in spans]

        accuracy = accuracy_score(gold, pred)
        macro_f1 = f1_score(gold, pred, labels=ABCD_KEYS, average="macro", zero_division=0)
        report_str = classification_report(
            gold, pred,
            labels=ABCD_KEYS,
            zero_division=0,
        )

        # Per-polarity
        def acc_subset(polarity):
            sub = [(g, p) for s, g, p in zip(spans, gold, pred) if s.get("polarity") == polarity]
            if not sub:
                return float("nan")
            return accuracy_score([g for g, _ in sub], [p for _, p in sub])

        results = TaskA3Results(
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            accuracy_adaptive=float(acc_subset("adaptive")),
            accuracy_maladaptive=float(acc_subset("maladaptive")),
            n_spans=len(spans),
            n_unknown=sum(1 for p in pred if p == "UNKNOWN"),
            report_str=report_str,
        )
        print(results)
        return results

    # ── Task B: NLI-based Summary Metrics ────────────────────────────────

    def _compute_consistency(
        self, pred_sentences: list[str], gold_sentences: list[str]
    ) -> float:
        """
        CS = (1 / (|S| * |G|)) * sum_{s in S} sum_{g in G}
             (1 - P(Contradiction | premise=g, hypothesis=s))

        Premise = gold sentence, hypothesis = predicted sentence.
        """
        if not pred_sentences or not gold_sentences:
            return 0.0
        contra_idx = self._get_nli_label_index("contradiction")
        premises, hypotheses = [], []
        for s in pred_sentences:
            for g in gold_sentences:
                premises.append(g)
                hypotheses.append(s)
        probs = self._nli_predict(premises, hypotheses)
        return float(np.mean(1.0 - probs[:, contra_idx]))

    def _compute_contradiction(
        self, pred_sentences: list[str], gold_sentences: list[str]
    ) -> float:
        """
        CT = (1 / |S|) * sum_{s in S} max_{g in G}
             P(Contradiction | premise=g, hypothesis=s)
        """
        if not pred_sentences or not gold_sentences:
            return 0.0
        contra_idx = self._get_nli_label_index("contradiction")
        max_contras = []
        for s in pred_sentences:
            probs = self._nli_predict(gold_sentences, [s] * len(gold_sentences))
            max_contras.append(float(probs[:, contra_idx].max()))
        return float(np.mean(max_contras))

    def _compute_evidence_alignment(
        self, pred_sentences: list[str], evidence_spans: list[str]
    ) -> float:
        """
        EA = (1 / |H|) * sum_{h in H} max_{s in S}
             P(Entailment | premise=h, hypothesis=s)

        Evidence is the premise, predicted summary sentence is the hypothesis.
        """
        if not evidence_spans or not pred_sentences:
            return 0.0
        entail_idx = self._get_nli_label_index("entailment")
        max_entails = []
        for h in evidence_spans:
            probs = self._nli_predict([h] * len(pred_sentences), pred_sentences)
            max_entails.append(float(probs[:, entail_idx].max()))
        return float(np.mean(max_entails))

    def evaluate_task_b(
        self,
        predicted_summaries: list[str],
        gold_summaries: list[str],
        evidence_spans: list[list[str]] = None,
    ) -> TaskBResults:
        self._load_nli()
        assert len(predicted_summaries) == len(gold_summaries), \
            f"Length mismatch: {len(predicted_summaries)} vs {len(gold_summaries)}"

        print(f"Evaluating Task B ({len(predicted_summaries)} posts)...")

        per_post_cs, per_post_ct, per_post_ea = [], [], []

        for i in range(len(predicted_summaries)):
            pred_sents = split_sentences(predicted_summaries[i])
            gold_sents = split_sentences(gold_summaries[i])

            if not pred_sents or not gold_sents:
                per_post_cs.append(0.0)
                per_post_ct.append(1.0)
                per_post_ea.append(0.0)
                continue

            per_post_cs.append(self._compute_consistency(pred_sents, gold_sents))
            per_post_ct.append(self._compute_contradiction(pred_sents, gold_sents))

            if evidence_spans and i < len(evidence_spans) and evidence_spans[i]:
                per_post_ea.append(
                    self._compute_evidence_alignment(pred_sents, evidence_spans[i])
                )
            else:
                per_post_ea.append(float("nan"))

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(predicted_summaries)}] "
                      f"CS={np.mean(per_post_cs):.4f} "
                      f"CT={np.mean(per_post_ct):.4f}")

        ea_valid = [v for v in per_post_ea if not np.isnan(v)]
        mean_ea = float(np.mean(ea_valid)) if ea_valid else float("nan")

        results = TaskBResults(
            consistency=float(np.mean(per_post_cs)),
            contradiction=float(np.mean(per_post_ct)),
            evidence_alignment=mean_ea,
            n_posts=len(predicted_summaries),
            per_post_cs=per_post_cs,
            per_post_ct=per_post_ct,
            per_post_ea=per_post_ea,
        )
        print(results)
        return results

    # ── Task C: Timeline-Level ───────────────────────────────────────────

    def evaluate_task_c(
        self,
        predicted_summaries: list[str],
        gold_summaries: list[str],
    ) -> TaskCResults:
        self._load_nli()
        assert len(predicted_summaries) == len(gold_summaries), \
            f"Length mismatch: {len(predicted_summaries)} vs {len(gold_summaries)}"

        print(f"Evaluating Task C ({len(predicted_summaries)} timelines)...")

        per_tl_cs, per_tl_ct = [], []
        for i in range(len(predicted_summaries)):
            pred_sents = split_sentences(predicted_summaries[i])
            gold_sents = split_sentences(gold_summaries[i])

            if not pred_sents or not gold_sents:
                per_tl_cs.append(0.0)
                per_tl_ct.append(1.0)
                continue

            cs = self._compute_consistency(pred_sents, gold_sents)
            ct = self._compute_contradiction(pred_sents, gold_sents)
            per_tl_cs.append(cs)
            per_tl_ct.append(ct)
            print(f"  Timeline {i+1}/{len(predicted_summaries)}: CS={cs:.4f} CT={ct:.4f}")

        results = TaskCResults(
            consistency=float(np.mean(per_tl_cs)),
            contradiction=float(np.mean(per_tl_ct)),
            n_timelines=len(predicted_summaries),
            per_timeline_cs=per_tl_cs,
            per_timeline_ct=per_tl_ct,
        )
        print(results)
        return results

    # ── Pipeline Output Convenience ──────────────────────────────────────

    def evaluate_pipeline_output(
        self,
        results_path: str,
        mode: str = "zero_shot",
    ) -> dict:
        """
        Evaluate Task B from `pipeline.py`'s task_b_results.json output.

        Args:
            results_path: Path to task_b_results.json or task_b_cv_results.json
            mode: "zero_shot" or "one_shot"
        """
        print(f"\nLoading pipeline results from: {results_path}")
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        summary_key = f"{mode}_summary"
        valid = [
            r for r in results
            if r.get("gold_summary") and r.get(summary_key)
        ]
        print(f"Found {len(valid)} posts with both gold and {mode} summaries")
        if not valid:
            return {}

        predicted = [r[summary_key] for r in valid]
        gold = [r["gold_summary"] for r in valid]
        evidence = None
        if "evidence_texts" in valid[0]:
            evidence = [r.get("evidence_texts", []) for r in valid]

        b = self.evaluate_task_b(predicted, gold, evidence)

        return {
            "mode": mode,
            "n_evaluated": len(valid),
            "task_b": {
                "consistency": b.consistency,
                "contradiction": b.contradiction,
                "evidence_alignment": b.evidence_alignment,
            },
        }

    def compare_modes(self, results_path: str) -> dict:
        """Compare zero-shot vs one-shot summaries from pipeline output."""
        print("\n" + "=" * 60)
        print("COMPARISON: Zero-Shot vs One-Shot (Structural Retrieval)")
        print("=" * 60)

        zs = self.evaluate_pipeline_output(results_path, mode="zero_shot")
        os_ = self.evaluate_pipeline_output(results_path, mode="one_shot")
        comparison = {"zero_shot": zs, "one_shot": os_}

        if zs and os_:
            zs_b = zs.get("task_b", {})
            os_b = os_.get("task_b", {})
            print(f"\n{'Metric':<25} {'Zero-Shot':>12} {'One-Shot':>12} {'Delta':>12}")
            print("-" * 61)
            for metric in ["consistency", "contradiction", "evidence_alignment"]:
                zs_val = zs_b.get(metric, float("nan"))
                os_val = os_b.get(metric, float("nan"))
                delta = (os_val - zs_val) if not (np.isnan(zs_val) or np.isnan(os_val)) else float("nan")
                # CS / EA: higher is better; CT: lower is better
                better = "+" if (metric != "contradiction" and delta > 0) or \
                              (metric == "contradiction" and delta < 0) else ""
                print(f"  {metric:<23} {zs_val:>12.4f} {os_val:>12.4f} {better}{delta:>11.4f}")

        return comparison


# ── CLI ──────────────────────────────────────────────────────────────────────

def _extract_timestamp(predictions_path: str) -> str:
    """
    Extract the timestamp suffix from a pipeline output filename.

    `task_a3_predictions_1714501234.567.json`  →  `1714501234.567`
    `task_b_results_1714501234.567.json`       →  `1714501234.567`
    `task_b_cv_results_1714501234.567.json`    →  `1714501234.567`

    Raises ValueError if the filename doesn't match the expected pattern.
    """
    import re
    stem = os.path.basename(predictions_path)
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    m = re.search(r"_(\d+(?:\.\d+)?)$", stem)
    if not m:
        raise ValueError(
            f"Could not extract timestamp from filename '{predictions_path}'. "
            f"Expected something like 'task_a3_predictions_<timestamp>.json'."
        )
    return m.group(1)


def _eval_output_path(predictions_path: str, task: str) -> str:
    """Compose the evaluation output path next to the predictions file."""
    timestamp = _extract_timestamp(predictions_path)
    out_dir = os.path.dirname(os.path.abspath(predictions_path))
    return os.path.join(out_dir, f"task_{task}_evaluation_{timestamp}.json")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLPsych 2025 Evaluation Metrics")
    parser.add_argument("--task", type=str, default="b",
                        choices=["a3", "b"],
                        help="Which task to evaluate")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to pipeline output JSON "
                             "(task_a3_predictions_<ts>.json for --task a3, "
                             "task_b_results_<ts>.json for --task b)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["zero_shot", "one_shot", "both"],
                        help="Task B only: which summary mode to evaluate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detects if not set)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for NLI inference")

    args = parser.parse_args()
    evaluator = Evaluator(device=args.device, batch_size=args.batch_size)

    output_path = _eval_output_path(args.results, args.task)

    if args.task == "a3":
        results = evaluator.evaluate_task_a3(args.results)
        payload = {
            "source": os.path.basename(args.results),
            "accuracy": results.accuracy,
            "macro_f1": results.macro_f1,
            "accuracy_adaptive": results.accuracy_adaptive,
            "accuracy_maladaptive": results.accuracy_maladaptive,
            "n_spans": results.n_spans,
            "n_unknown": results.n_unknown,
        }

    elif args.task == "b":
        if args.mode == "both":
            comparison = evaluator.compare_modes(args.results)
            payload = {"source": os.path.basename(args.results), **comparison}
        else:
            results = evaluator.evaluate_pipeline_output(args.results, mode=args.mode)
            payload = {"source": os.path.basename(args.results), **results}

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
