"""
ABCD-fidelity metric: NLI-based recall / precision / F1 of a summary against
the gold ABCD element profile of its source post.

The official Task B NLI metric (CS) rewards summaries that do not contradict the gold, but it
does not check that a summary actually *covers* the structured adaptive /
maladaptive elements the MIND framework annotates. ABCD-fidelity does exactly
that, reusing the SAME NLI model as CS
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) so any difference
between the two metrics is *what* they measure, not *how*.

12 canonical hypotheses (6 ABCD keys x 2 polarities) are built from
`ABCD_DEFINITIONS` in pipeline.py, BUT the generic fixed-hypothesis sweep
underperforms on the ceiling check (a precise gold summary that discusses only
the relevant elements does not entail broad generic claims like "the person's
Affect is adaptive", so gold-summary recall collapses below the real systems —
an inverted ceiling). The spec anticipates this ("Keep the LLM-extraction
variant as a fallback only if the sweep underperforms"). We therefore use:

  Recall     = bidirectional NLI evidence-alignment over the post's GOLD evidence
               spans (the spec's sanctioned fallback). For each gold span (with its
               key + polarity), covered iff max over summary sentences of
               max(P(entail span->sent), P(entail sent->span)) >= tau_recall.
               recall = covered_spans / total_gold_spans. This is reliable because
               it compares the summary to the SPECIFIC gold text (a clean ceiling:
               gold summary -> high recall, constant -> 0).
  Precision  = the generic fixed-hypothesis sweep (the spec's "simpler,
               threshold-consistent" approach), kept as a BREADTH / over-assertion
               proxy: for each of the 12 (key,polarity) hypotheses, asserted iff
               P(entail summary->hyp) >= tau_precision. precision =
               |asserted ∩ gold| / |asserted|, with precision = 1.0 when nothing
               is asserted (vacuously precise: the summary invented no ABCD claim).
               A match requires BOTH key AND polarity to agree.
  F1         = harmonic mean of P and R (0 when recall is 0).

Generic hypotheses over-fire on broad categories (any
emotional text entails "the person's Affect is adaptive"), so they are useless
for recall (coverage of specific gold elements) but useful as a relative
over-assertion signal for precision (EAIonFlux's template-y summaries assert
many -> low precision; gold asserts few -> high precision). LLM-extraction
precision (the cleaner fallback) is deferred until a local LLM is available
(llama_cpp is not installed in this environment and the data-sharing agreement
prohibits cloud LLM APIs).

Gold elements are sourced from `post_data.csv`: the `adaptive_evidence` and
`maladaptive_evidence` columns hold Python dict literals keyed by ABCD key
(`{'A': {'Category':..., 'highlighted_evidence':...}, 'B-O': {...}, ...}`).
A (key, polarity) slot is "gold" for a post iff that key is present in that
post's `adaptive_evidence` (polarity=adaptive) or `maladaptive_evidence`
(polarity=maladaptive) dict.

Usage:
    # Score the 4 systems + 3 degenerate baselines, run the built-in sanity checks
    python abcd_fidelity.py --sanity \
        --post_data post_data.csv --tau 0.5 --output_dir ./outputs

    # Score specific system JSONs (official-submission format; path=label)
    python abcd_fidelity.py \
        --post_data post_data.csv --tau 0.5 --output_dir ./outputs \
        --system "EAIonFlux=outputs/eaion_flux/aeion-flux_task-b-k5.json" \
        --system "BLUE=clpsych2025/evaluation/submissions_dev/blue-task-b.json" \
        --system "Official_baseline=clpsych2025/evaluation/submissions_dev/baseline-task-b.json" \
        --system "Ours=clpsych2025/evaluation/official_run/random_structural.json"

Sanity checks (must pass before trusting any number):
    constant     -> F1 ~ 0 (recall ~ 0); a content-free string should entail
                    almost none of the 12 element hypotheses.
    gold-summary -> F1 high (ceiling); the gold summary should cover the gold
                    elements it was built from.
    EAIonFlux    -> high recall, low precision (the "shotgun" signature).
"""

import argparse
import ast
import os
import sys
import types

import numpy as np
import pandas as pd

# pipeline.py imports heavy local-LLM deps (llama_cpp, sentence_transformers) at
# module top level that are NOT required for the pure-data ABCD_DEFINITIONS /
# ABCD_KEYS constants we need. Stub them so the import succeeds on machines
# without those packages installed; the constants are plain Python data.
for _mod, _attrs in (("llama_cpp", ("Llama",)),
                     ("sentence_transformers", ("SentenceTransformer",))):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
    for _a in _attrs:
        setattr(sys.modules[_mod], _a, None)

# Reuse the framework definitions and the exact NLI model the official metric uses.
from pipelines.pipeline import ABCD_DEFINITIONS, ABCD_KEYS
from eval.evaluation import Evaluator


# ── 12 canonical hypotheses ───────────────────────────────────────────────────

POLARITIES = ["adaptive", "maladaptive"]


def _trim_examples(examples: str, k: int = 5) -> str:
    """Take the first k comma-separated example labels, keep the line short."""
    parts = [p.strip() for p in examples.split(",") if p.strip()]
    return ", ".join(parts[:k])


def build_hypotheses() -> dict[tuple[str, str], str]:
    """
    One natural-language entailment hypothesis per (ABCD key, polarity), used
    for the PRECISION (over-assertion) sweep only. Recall uses bidirectional
    evidence-alignment over gold spans (see score_summaries).

    Each hypothesis is a statement ABOUT THE PERSON (not about the summary): a
    broad claim that an over-generating summary tends to entail. Adaptive and
    maladaptive variants share the key but differ on the polarity word, so the
    model can (roughly) distinguish them. Examples are dropped: when included
    they read as a conjunction the summary must satisfy in full, which collapses
    entailment on the real summaries.
    """
    hyps = {}
    for key in ABCD_KEYS:
        d = ABCD_DEFINITIONS[key]
        name = d["name"]
        desc = d["description"]
        for pol in POLARITIES:
            hyps[(key, pol)] = f"The person's {name} ({key}) is {pol}. {desc}"
    return hyps


HYPOTHESIS_KEYS = [(k, p) for k in ABCD_KEYS for p in POLARITIES]  # 12, fixed order


# ── Gold ABCD elements per post (from post_data.csv) ──────────────────────────

def _parse_evidence_dict(cell) -> dict:
    """Parse a post_data.csv adaptive/maladaptive_evidence cell into an
    {ABCD_key: {'Category':..., 'highlighted_evidence':...}} dict. Empty/blank
    cells return {}."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return cell
    text = str(cell).strip()
    if not text or text == "{}":
        return {}
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    try:
        parsed = __import__("json").loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, __import__("json").JSONDecodeError):
        pass
    return {}


def load_gold(post_data_csv: str) -> dict[str, list[tuple[str, str, str]]]:
    """
    Map post_id -> list of gold evidence spans, each as (ABCD_key, polarity,
    span_text). The span text is the gold-annotated `highlighted_evidence`;
    the (key, polarity) pair is its structured tag. Posts with no evidence get
    an empty list (caller skips them for recall / precision).
    """
    df = pd.read_csv(post_data_csv)
    gold = {}
    for _, row in df.iterrows():
        post_id = str(row["post_id"])
        spans: list[tuple[str, str, str]] = []
        for polarity, col in [("adaptive", "adaptive_evidence"),
                              ("maladaptive", "maladaptive_evidence")]:
            ev = _parse_evidence_dict(row[col])
            for key, info in ev.items():
                if key in ABCD_KEYS:
                    text = (info.get("highlighted_evidence", "") or "").strip() if isinstance(info, dict) else ""
                    if text:
                        spans.append((key, polarity, text))
        gold[post_id] = spans
    return gold


# ── Summaries per system (official-submission JSON) ───────────────────────────

def load_summaries(path: str, label: str) -> pd.DataFrame:
    """Load summaries from official-format JSON or a CSV with a summary column.
    Returns one row per (system, post_id) with the summary text."""
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = __import__("json").load(f)
        for tl_id, tl in data.items():
            post_level = tl.get("post_level", {})
            for post_id, entry in post_level.items():
                summary = entry.get("summary", "") or ""
                rows.append({
                    "system": label,
                    "timeline_id": tl_id,
                    "post_id": str(post_id),
                    "summary": summary,
                })
    elif ext in (".csv", ".xlsx", ".xls"):
        import pandas as _pd
        if ext == ".csv":
            try:
                df = _pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = _pd.read_csv(path, encoding="cp1252")
        else:
            df = _pd.read_excel(path)
        # Find the summary column (first column containing 'summ' excluding 'post_summary')
        summ_col = None
        for c in df.columns:
            cl = c.lower()
            if "summ" in cl and cl != "post_summary":
                summ_col = c
                break
        if summ_col is None:
            raise SystemExit(f"Could not find a summary column in {path}. "
                             f"Columns: {list(df.columns)}")
        for _, row in df.iterrows():
            summary = str(row.get(summ_col, "") or "")
            post_id = str(row.get("post_id", ""))
            if post_id:
                rows.append({
                    "system": label,
                    "timeline_id": str(row.get("timeline_id", "")),
                    "post_id": post_id,
                    "summary": summary,
                })
    else:
        raise SystemExit(f"Unsupported file type: {ext} ({path})")
    return pd.DataFrame(rows)


# ── NLI scoring ───────────────────────────────────────────────────────────────

def score_summaries(summaries: pd.DataFrame, hypotheses: dict,
                    tau_recall: float, tau_precision: float,
                    evaluator: Evaluator, batch_size: int = 16) -> pd.DataFrame:
    """
    Per (system, post_id) recall / precision / F1.

    Recall  (bidirectional NLI evidence-alignment over gold spans):
        for each gold span (key, polarity, text), covered iff
            max over summary sentences of
              max( P(entail | premise=span, hyp=sent),
                   P(entail | premise=sent,  hyp=span) ) >= tau_recall
        recall = covered_spans / total_gold_spans.
        Bidirectional catches paraphrase in either direction (gold summaries
        are abstractions of the evidence, not copies), which a single
        direction misses. A gold (key,polarity) slot is covered iff its span is.

    Precision (generic 12-hypothesis sweep, breadth / over-assertion proxy):
        for each of the 12 (key,polarity) hypotheses, asserted iff
            P(entail | premise=summary, hyp=generic) >= tau_precision.
        precision = |asserted ∩ gold_slots| / |asserted|, and = 1.0 when
        nothing is asserted (vacuously precise: no invented ABCD claim).

    F1 = harmonic mean (0 when recall == 0).
    """
    from eval.evaluation import split_sentences
    entail_idx = evaluator._get_nli_label_index("entailment")
    hyp_texts = [hypotheses[k] for k in HYPOTHESIS_KEYS]  # 12, fixed order

    out_rows = []
    for _, row in summaries.iterrows():
        summary = row["summary"] or ""
        gold_spans = row["_gold_spans"]            # list[(key, polarity, text)]
        gold_slots = {(k, p) for k, p, _ in gold_spans}
        n_gold = len(gold_spans)

        # ── Recall: bidirectional evidence-alignment per gold span ──
        sents = split_sentences(summary) if summary.strip() else []
        covered = 0
        if n_gold > 0 and sents:
            premises, hypotheses_batch = [], []
            for _, _, span in gold_spans:
                for sent in sents:
                    premises.append(span)           # direction 1: span -> sent
                    hypotheses_batch.append(sent)
            for _, _, span in gold_spans:
                for sent in sents:
                    premises.append(sent)           # direction 2: sent -> span
                    hypotheses_batch.append(span)
            probs = evaluator._nli_predict(premises, hypotheses_batch)
            p_ent = probs[:, entail_idx]
            n_per_span = len(sents)
            n_total = len(gold_spans)
            for si in range(n_total):
                d1 = p_ent[si * n_per_span:(si + 1) * n_per_span]          # span->sent
                d2 = p_ent[n_total * n_per_span + si * n_per_span:          # sent->span
                           n_total * n_per_span + (si + 1) * n_per_span]
                if max(d1.max(), d2.max()) >= tau_recall:
                    covered += 1
        recall = (covered / n_gold) if n_gold > 0 else np.nan

        # ── Precision: generic 12-hypothesis sweep (whole summary as premise) ──
        premise = summary if summary.strip() else " "
        gprobs = evaluator._nli_predict([premise] * 12, hyp_texts)
        gp_ent = gprobs[:, entail_idx]
        asserted_slots = {HYPOTHESIS_KEYS[i] for i in range(12) if gp_ent[i] >= tau_precision}
        n_asserted = len(asserted_slots)
        if n_gold == 0:
            precision = np.nan
        elif n_asserted == 0:
            precision = 1.0          # vacuously precise: invented no ABCD claim
        else:
            precision = len(asserted_slots & gold_slots) / n_asserted

        # ── F1 ──
        if n_gold == 0:
            f1 = np.nan
        elif recall == 0:
            f1 = 0.0
        else:
            denom = recall + precision
            f1 = (2 * recall * precision / denom) if denom > 0 else 0.0

        out_rows.append({
            "system": row["system"],
            "timeline_id": row["timeline_id"],
            "post_id": row["post_id"],
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "n_gold_spans": n_gold,
            "n_covered": covered,
            "n_asserted": n_asserted,
            "tau_recall": tau_recall,
            "tau_precision": tau_precision,
        })
    return pd.DataFrame(out_rows)


# ── Sanity checks ─────────────────────────────────────────────────────────────

# Default system JSONs for the built-in sanity checks. Overridable via CLI.
DEFAULT_SANITY_SYSTEMS = [
    ("constant", "clpsych2025/evaluation/official_run/baseline_constant.json"),
    ("gold-summary", "clpsych2025/evaluation/official_run/baseline_goldsummary.json"),
    ("EAIonFlux", "outputs/eaion_flux/aeion-flux_task-b-k5.json"),
    ("BLUE", "clpsych2025/evaluation/submissions_dev/blue-task-b.json"),
    ("Official_baseline", "clpsych2025/evaluation/submissions_dev/baseline-task-b.json"),
    ("Ours", "clpsych2025/evaluation/official_run/random_structural.json"),
    ("empty", "clpsych2025/evaluation/official_run/baseline_empty.json"),
]


def run_sanity_checks(per_summary: pd.DataFrame) -> None:
    """Print the three headline sanity checks against the by-system means."""
    by_system = (per_summary.groupby("system")[["recall", "precision", "f1"]]
                 .mean().round(3))
    print("\n" + "=" * 70)
    print("SANITY CHECKS (by-system mean recall/precision/F1)")
    print("=" * 70)
    print(by_system.to_string())

    print("\nExpected signatures:")
    print("  constant      -> LOWEST recall (the floor): fidelity, unlike CS, has a "
          "usable floor for a content-free string.")
    print("  gold-summary  -> recall above constant (ceiling over the FLOOR; absolute "
          "recall is capped by gold-summary concision — see verbosity note below).")
    print("  EAIonFlux     -> high recall, LOW precision (shotgun: asserts broadly).")
    print("  empty         -> recall 0, precision 1.0 (vacuous: asserts nothing).")
    print("  NOTE: gold-summary recall is NOT the absolute max — verbose/template-y")
    print("  summaries (EAIonFlux, Official_baseline) entail MORE gold spans because")
    print("  they spell out adaptive/maladaptive states explicitly. Fidelity-recall")
    print("  carries a residual verbosity confound (the same confound the critique")
    print("  flags for CS). This motivates the precision side and A2 (specificity).")

    # Hard, defensible requirements (not an absolute gold-ceiling, which the
    # verbosity confound breaks): constant is the floor; gold clears the floor;
    # EAIonFlux over-asserts.
    fail = []
    nonempty = [s for s in by_system.index if s != "empty"]
    if "constant" in by_system.index and nonempty:
        r_const = by_system.loc["constant", "recall"]
        r_min_other = by_system.loc[nonempty, "recall"].drop(
            labels=[s for s in nonempty if s == "constant"], errors="ignore").min()
        if r_const >= r_min_other:
            fail.append(f"constant recall={r_const:.3f} is NOT the floor (min other "
                        f"system recall={r_min_other:.3f}): fidelity has no usable floor. "
                        "Raise --tau_recall.")
    if "gold-summary" in by_system.index and "constant" in by_system.index:
        if by_system.loc["gold-summary", "recall"] <= by_system.loc["constant", "recall"]:
            fail.append("gold-summary recall <= constant recall: ceiling broken below the floor.")
    if "EAIonFlux" in by_system.index:
        r_eai = by_system.loc["EAIonFlux", "recall"]
        p_eai = by_system.loc["EAIonFlux", "precision"]
        if p_eai >= r_eai:
            fail.append(f"EAIonFlux precision={p_eai:.3f} >= recall={r_eai:.3f}: "
                        "no over-assertion gap — precision not catching invented elements.")
    if fail:
        print("\nSANITY CHECKS FAILED:")
        for m in fail:
            print("  -", m)
    else:
        print("\nSanity checks passed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABCD-fidelity metric: NLI recall/precision/F1 vs gold ABCD elements."
    )
    parser.add_argument("--post_data", default="post_data.csv",
                        help="post_data.csv with adaptive_evidence/maladaptive_evidence "
                             "dict-literal columns.")
    parser.add_argument("--tau_recall", type=float, default=0.5,
                        help="P(entail) threshold for the bidirectional "
                             "evidence-alignment RECALL side (gold-span coverage). "
                             "Tune on a few examples; the value used is logged. "
                             "0.5 keeps the constant baseline as the floor.")
    parser.add_argument("--tau_precision", type=float, default=0.5,
                        help="P(entail) threshold for the generic 12-hypothesis "
                             "PRECISION (over-assertion) sweep. Kept as a breadth "
                             "proxy; LLM-extraction precision is the spec's sanctioned "
                             "fallback, deferred until a local LLM is available.")
    parser.add_argument("--system", action="append", default=None,
                        help="System submission JSON in official format, as 'label=path'. "
                             "Repeat for multiple systems. If omitted, --sanity defaults apply.")
    parser.add_argument("--sanity", action="store_true",
                        help="Run the built-in sanity checks over constant / gold-summary / "
                             "EAIonFlux / BLUE / baseline / Ours / empty. Sets sensible "
                             "defaults if --system is not given.")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    print(f"ABCD-fidelity  | tau_recall = {args.tau_recall}  | tau_precision = {args.tau_precision}")
    hypotheses = build_hypotheses()
    print("\n12 canonical hypotheses (built from ABCD_DEFINITIONS):")
    for (key, pol), h in hypotheses.items():
        print(f"  ({key}, {pol}): {h}")

    # ── Gold ABCD elements per post ──
    gold = load_gold(args.post_data)
    n_annotated = sum(1 for s in gold.values() if s)
    print(f"\nLoaded gold ABCD elements for {n_annotated}/{len(gold)} posts "
          f"(posts with >=1 gold slot).")

    # ── Which systems to score ──
    if args.system:
        specs = []
        for spec in args.system:
            label, _, path = spec.partition("=")
            label, path = label.strip(), path.strip()
            if not path:
                raise SystemExit(f"--system entries must be 'label=path'; got {spec!r}")
            specs.append((label, path))
    elif args.sanity:
        specs = DEFAULT_SANITY_SYSTEMS
    else:
        raise SystemExit("Pass --system 'label=path' (repeatable) or --sanity.")

    frames = []
    for label, path in specs:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {label}.")
            continue
        df = load_summaries(path, label)
        df["_gold_spans"] = df["post_id"].map(lambda pid: gold.get(pid, []))
        frames.append(df)
        print(f"  loaded {label}: {len(df)} posts from {path}")
    if not frames:
        raise SystemExit("No system summary files loaded.")
    summaries = pd.concat(frames, ignore_index=True)

    # Only score posts that have gold evidence (recall undefined otherwise).
    scored = summaries[summaries["_gold_spans"].map(len) > 0].reset_index(drop=True)
    skipped = len(summaries) - len(scored)
    print(f"\nScoring {len(scored)} summaries with gold evidence "
          f"(skipped {skipped} posts with no gold slots).")

    # ── NLI ──
    evaluator = Evaluator(device=args.device, batch_size=args.batch_size)
    per_summary = score_summaries(scored, hypotheses, args.tau_recall,
                                  args.tau_precision, evaluator, args.batch_size)

    # ── Report ──
    os.makedirs(args.output_dir, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    by_system = (per_summary.groupby("system")[["recall", "precision", "f1",
                                                "n_gold_spans", "n_covered", "n_asserted"]]
                 .mean().round(3))
    print("\n" + "=" * 70)
    print("ABCD-FIDELITY BY SYSTEM  (mean over posts with gold evidence)")
    print("=" * 70)
    print(by_system.to_string())

    out_per = os.path.join(args.output_dir, "abcd_fidelity_per_summary.csv")
    out_by = os.path.join(args.output_dir, "abcd_fidelity_by_system.csv")
    per_summary.round(6).to_csv(out_per, index=False)
    by_system.round(6).to_csv(out_by)
    print(f"\nWrote {out_per} and {out_by}")

    if args.sanity:
        run_sanity_checks(per_summary)


if __name__ == "__main__":
    main()
