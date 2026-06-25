"""
Specificity / informativeness analysis of generated summaries.

Motivation
----------
The NLI-based CLPsych Task B/C metric (mean consistency) rewards summaries that
avoid contradicting the gold summary. A fluent-but-generic summary — e.g. the
`baseline_constant` sentence, or a "condensed regurgitation" — can score high
precisely because it says little that *could* be contradicted. This script turns
that qualitative observation into a measured property: it computes per-summary
specificity signals and shows whether high-consistency summaries are
systematically more generic.

Everything here is pure computation over the generated summaries (model outputs),
so it needs no annotators and never touches the protected source data.

Specificity panel (per summary; pooled corpus = all loaded summaries)
---------------------------------------------------------------------
    n_tokens             length in word tokens
    ttr                  type-token ratio (unique/total) — lexical diversity
    mattr                moving-average TTR (window 50) — length-robust diversity
    content_word_ratio   share of tokens that are content words (non-stopword)
    mean_idf             mean corpus IDF of the summary's word types. HIGH = uses
                         rarer, more post-specific words; LOW = generic vocabulary
    hedge_density        density of vagueness/hedge markers (may, seems, some...)
    abcd_term_density    density of self-state / ABCD framework vocabulary
    bigram_commonness    mean share of the corpus that reuses each of the
                         summary's bigrams. HIGH = boilerplate shared across many
                         summaries (cf. MSLR's self-repetition finding)

The informative-vs-generic axis is multi-signal on purpose: a system can score
high on `abcd_term_density` (it name-drops ABCD labels) yet low on `mean_idf` and
high on `bigram_commonness` (it is still generic) — exactly the BLUE pattern.

Inputs
------
One or more --source specs, each `name=path#field`:
    name    a system label used in the output tables
    path    a JSON file (list of records) or a CSV file
    field   the summary field/column; omit (drop the `#field`) to auto-detect
Use '#' (not ':') as the field separator so Windows drive letters survive.

    --source "eaionflux_k5=outputs/task_b_eaionflux_k5_<ts>.json#one_shot_summary"
    --source "oneshot=outputs/task_b_results_gold_hybrid-a0.50_<ts>.json#one_shot_summary"
    --source "gold=outputs/task_b_results_gold_hybrid-a0.50_<ts>.json#gold_summary"
    --include_constant            # add the baseline_constant sentence as its own system

Optionally join consistency scores to get the headline correlation:
    --cs_by_system cs.csv         # columns: system, cs   (or set --cs_col)
reports Spearman(specificity_metric, CS) across systems for every metric.

Outputs (to --output_dir)
    specificity_per_summary.csv   one row per summary, all metrics
    specificity_by_system.csv     one row per system, mean of each metric
                                  (feeds rank_correlation.py --level teams --metrics_csv)

Usage
    python specificity_analysis.py \
        --source "eaionflux_k5=outputs/task_b_eaionflux_k5_x.json#one_shot_summary" \
        --source "oneshot=outputs/task_b_results_x.json#one_shot_summary" \
        --include_constant --cs_by_system cs_by_system.csv
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from make_degenerate_baselines import CONSTANT_SUMMARY
except Exception:  # keep a single source of truth, but stay importable if it moves
    CONSTANT_SUMMARY = (
        "The person is experiencing emotional distress and is finding it "
        "difficult to cope with their situation."
    )

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

# Hedge / vagueness markers — generic summaries lean on these.
HEDGES = {
    "may", "might", "maybe", "perhaps", "possibly", "possible", "seem", "seems",
    "seemed", "appear", "appears", "appeared", "suggest", "suggests", "could",
    "would", "generally", "often", "sometimes", "somewhat", "tend", "tends",
    "likely", "presumably", "apparently", "relatively", "fairly", "rather",
    "various", "certain", "some", "overall", "potential", "potentially",
}

# Self-state / ABCD framework vocabulary — domain informativeness.
# TODO: update with keywords from paper (and potentially keywords I added)
ABCD_TERMS = {
    "affect", "affective", "emotion", "emotions", "emotional", "behavior",
    "behaviour", "behavioral", "behaviours", "cognition", "cognitive", "desire",
    "desires", "need", "needs", "adaptive", "maladaptive", "self-state",
    "self-states", "dominant", "interplay", "well-being", "wellbeing",
    "relatedness", "autonomy", "competence", "self-criticism", "self-compassion",
    "avoidance", "self-care", "self-harm", "rumination", "hopelessness",
}

CONTENT_STOP = set(ENGLISH_STOP_WORDS)

PANEL = [
    "n_tokens", "ttr", "mattr", "content_word_ratio", "mean_idf",
    "hedge_density", "abcd_term_density", "bigram_commonness",
]
# Higher = more specific/informative for these; lower for the rest.
HIGHER_IS_MORE_SPECIFIC = {"ttr", "mattr", "content_word_ratio", "mean_idf", "abcd_term_density"}


# ── tokenization ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def bigrams(tokens: list[str]) -> list[tuple]:
    return list(zip(tokens, tokens[1:]))


# ── source loading ────────────────────────────────────────────────────────────

_AUTO_JSON_FIELDS = [
    "predicted_summary", "one_shot_summary", "summary",
    "zero_shot_summary", "gold_summary",
]


def _auto_id(rec: dict, i: int) -> str:
    if "timeline_id" in rec and "post_index" in rec:
        return f'{rec["timeline_id"]}#{rec["post_index"]}'
    if "post_id" in rec and rec["post_id"]:
        return str(rec["post_id"])
    if "timeline_id" in rec:
        return str(rec["timeline_id"])
    return str(i)


def load_source(spec: str) -> tuple[str, list[tuple]]:
    """Parse `name=path#field` and return (system_name, [(id, summary), ...])."""
    if "=" not in spec:
        raise SystemExit(f"--source must look like name=path#field; got {spec!r}")
    name, rest = spec.split("=", 1)
    if "#" in rest:
        path, field = rest.rsplit("#", 1)
    else:
        path, field = rest, None
    name = name.strip()
    path = path.strip()

    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".json":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON list of records (got {type(data).__name__}). "
                             "Official-submission dicts are not supported here.")
        if field is None:
            field = next((k for k in _AUTO_JSON_FIELDS if data and k in data[0]), None)
            if field is None:
                raise SystemExit(f"{path}: could not auto-detect a summary field; "
                                 f"pass name=path#field. Keys: {list(data[0].keys()) if data else []}")
        for i, rec in enumerate(data):
            rows.append((_auto_id(rec, i), rec.get(field, "") or ""))
    elif ext == ".csv":
        df = pd.read_csv(path)
        if field is None:
            cand = [c for c in df.columns if "summ" in c.lower()]
            if not cand:
                raise SystemExit(f"{path}: no column matching 'summ'; pass name=path#column.")
            field = cand[0]
        if field not in df.columns:
            raise SystemExit(f"{path}: no column {field!r}. Columns: {list(df.columns)}")
        for i, val in enumerate(df[field].tolist()):
            rows.append((str(i), "" if pd.isna(val) else str(val)))
    else:
        raise SystemExit(f"{path}: unsupported extension {ext!r} (use .json or .csv).")

    print(f"  loaded {len(rows):4d} summaries for system '{name}' (field={field})")
    return name, rows


# ── corpus-level statistics ───────────────────────────────────────────────────

def build_corpus_stats(all_texts: list[str]):
    """Return (idf_lookup, default_idf, bigram_doc_freq, n_docs) computed over the pooled summaries."""
    nonempty = [t for t in all_texts if t.strip()]
    n_docs = max(len(nonempty), 1)

    # IDF over word types via sklearn (smooth_idf default; same vocabulary basis).
    vect = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    if nonempty:
        vect.fit(nonempty)
        idf = dict(zip(vect.get_feature_names_out(), vect.idf_))
    else:
        idf = {}
    default_idf = max(idf.values()) if idf else 1.0  # unseen token -> max specificity

    # Document frequency of bigrams (how many summaries contain each bigram).
    bg_df = Counter()
    for t in nonempty:
        for bg in set(bigrams(tokenize(t))):
            bg_df[bg] += 1

    return idf, default_idf, bg_df, n_docs


# ── per-summary metrics ───────────────────────────────────────────────────────

def mattr(tokens: list[str], window: int = 50) -> float:
    """
    moving average type token ratio, statistical measure of the lexical diversity
    and vocab richness in a document, the moving average part uses the average of a 
    window that is run over the text.
    NOTE: try 35, 40, and 45 if 50 doesnt work. Do not go below 30 or above 50.
    """
    if not tokens:
        return float("nan")
    if len(tokens) <= window:
        # Just the TTR when the window is too long
        return len(set(tokens)) / len(tokens)
    # full MATTR
    # amount of unique tokens / window (50) done in a loop 
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def summary_metrics(text: str, idf, default_idf, bg_df, n_docs) -> dict:
    """
    full summary metrics for the text, returns a dict with everything."""
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return {m: float("nan") for m in PANEL} | {"n_tokens": 0}

    # number of sentences
    n_sent = max(1, len(re.findall(r"[.!?]+", text)))
    # unique token set
    types = set(toks)
    # content words (stop words removed)
    content = [t for t in toks if t not in CONTENT_STOP]
    # vaugeness words (may, seem, appears, etc.)
    hedges = sum(1 for t in toks if t in HEDGES)
    # ABCD specific terms
    abcd = sum(1 for t in toks if t in ABCD_TERMS)
    # mean inverse document frequency
    mean_idf = float(np.mean([idf.get(t, default_idf) for t in types]))

    bgs = bigrams(toks)
    if bgs:
        bigram_commonness = float(np.mean([bg_df.get(bg, 1) / n_docs for bg in bgs]))
    else:
        bigram_commonness = float("nan")

    return {
        "n_tokens": n,
        "n_sentences": n_sent,
        "ttr": len(types) / n,
        "mattr": mattr(toks),
        "content_word_ratio": len(content) / n,
        "mean_idf": mean_idf,
        "hedge_density": hedges / n,
        "abcd_term_density": abcd / n,
        "bigram_commonness": bigram_commonness,
    }


# ── CS join (optional headline) ───────────────────────────────────────────────

def report_cs_correlation(by_system: pd.DataFrame, cs_path: str, cs_col: str):

    cs = pd.read_csv(cs_path)
    id_col = cs.columns[0] if "system" not in cs.columns else "system"
    cs = cs.set_index(id_col)
    cs.index = cs.index.astype(str)
    if cs_col not in cs.columns:
        # fall back to the first numeric column
        numeric = cs.select_dtypes(include="number")
        if numeric.empty:
            raise SystemExit(f"{cs_path}: no numeric CS column found.")
        cs_col = numeric.columns[0]
    cs_series = cs[cs_col]

    joined = by_system.join(cs_series.rename("CS"), how="inner")
    n = len(joined)
    print("\n" + "=" * 70)
    print(f"SPECIFICITY vs CONSISTENCY  ({cs_col}, n={n} systems with both)")
    print("=" * 70)
    if n < 5:
        print(f"NOTE: only {n} systems matched the CS file. Treat rho as a descriptive "
              f"effect size; the p-values below use scipy's asymptotic approximation and "
              f"are not reliable at this sample size (report rho, not significance).")
    print(joined.round(4).to_string())
    print("\nSpearman rho of each specificity metric vs CS "
          "(negative for 'more specific' metrics => higher CS rewards vaguer summaries):")
    for m in PANEL:
        pair = joined[[m, "CS"]].dropna()
        if len(pair) >= 2 and pair[m].nunique() > 1 and pair["CS"].nunique() > 1:
            r, p = spearmanr(pair[m], pair["CS"])
            arrow = "(higher=more specific)" if m in HIGHER_IS_MORE_SPECIFIC else "(higher=more generic)"
            star = " *" if (not np.isnan(p) and p < 0.05) else ""
            print(f"  {m:<22} rho={r:+.3f}  p={p:.3f}{star}   {arrow}")
        else:
            print(f"  {m:<22} rho=  nan  (insufficient variation)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Specificity / informativeness analysis of generated summaries."
    )
    parser.add_argument("--source", action="append", default=[],
                        help="name=path#field (repeatable). field optional (auto-detected).")
    parser.add_argument("--include_constant", action="store_true",
                        help="Add the baseline_constant sentence as a one-summary system.")
    parser.add_argument("--cs_by_system", default=None,
                        help="CSV of system-level consistency to correlate against "
                             "(columns: system, cs).")
    parser.add_argument("--cs_col", default="cs", help="CS column name in --cs_by_system.")
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    if not args.source and not args.include_constant:
        raise SystemExit("Give at least one --source (or --include_constant).")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading summary sources:")
    sources = [load_source(s) for s in args.source]
    if args.include_constant:
        sources.append(("constant", [("constant#0", CONSTANT_SUMMARY)]))

    # Pool every summary to compute corpus-relative statistics (IDF, bigram DF).
    all_texts = [text for _, rows in sources for _, text in rows]
    idf, default_idf, bg_df, n_docs = build_corpus_stats(all_texts)
    print(f"\nPooled corpus: {len(all_texts)} summaries, {n_docs} non-empty "
          f"(IDF + bigram stats computed over the pool).")

    # Per-summary metrics.
    per_rows = []
    for system, rows in sources:
        for sid, text in rows:
            m = summary_metrics(text, idf, default_idf, bg_df, n_docs)
            m["system"] = system
            m["id"] = sid
            per_rows.append(m)
    per_df = pd.DataFrame(per_rows)
    cols = ["system", "id", "n_tokens", "n_sentences"] + \
           [c for c in PANEL if c != "n_tokens"]
    per_df = per_df[[c for c in cols if c in per_df.columns]]

    # Aggregate to system level (NaN-aware means; empty summaries don't poison).
    by_system = per_df.groupby("system")[PANEL].mean()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 70)
    print("SPECIFICITY BY SYSTEM (mean per summary)")
    print("=" * 70)
    print(by_system.round(4).to_string())
    print("\nReading guide: ttr / mattr / content_word_ratio / mean_idf / "
          "abcd_term_density -> higher = more specific. hedge_density / "
          "bigram_commonness -> higher = more generic.")

    per_path = os.path.join(args.output_dir, "specificity_per_summary.csv")
    sys_path = os.path.join(args.output_dir, "specificity_by_system.csv")
    per_df.round(6).to_csv(per_path, index=False)
    by_system.round(6).to_csv(sys_path)
    print(f"\nWrote {per_path} and {sys_path}")
    print("Tip: feed specificity_by_system.csv into rank_correlation.py "
          "(--level teams --metrics_csv) alongside CS to put specificity and "
          "consistency in one Spearman matrix.")

    if args.cs_by_system:
        report_cs_correlation(by_system, args.cs_by_system, args.cs_col)


if __name__ == "__main__":
    main()
