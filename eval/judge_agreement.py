"""
LLM-judge vs human agreement: ICC (+ Spearman / Pearson) between the LLM-as-a-judge
scores and the human consensus ratings, per evaluation question.

Treats the human consensus (averaged across raters) and the LLM judge as two
"raters" of the same items (summaries) and reports:
    ICC(2,1)  two-way random, absolute agreement, single rater   [headline]
    ICC(3,1)  two-way mixed,  consistency,        single rater
    ICC(2,k)  / ICC(3,k)      reliability of the AVERAGE of the two raters
    Spearman rho, Pearson r (each with p), and the mean human/judge scores.

Supports:
  - Multiple human rater files (--human, nargs="+", with path=label syntax)
  - .xlsx, .csv, .json file formats
  - Encoding fallback (utf-8 → cp1252)
  - 'X/Y' fraction string parsing (q3b, q3d)
  - Excel datetime-corrupted fraction recovery (4/4 → April 4 → 1.0)
  - Column-name typo tolerance (q3b_unsupported_elements → q3c_unsupported_elements)
  - Duplicate row dropping per rater file
  - The judge file can have all posts; only posts matching the human set are used

Usage:
    python judge_agreement.py `
        --judge outputs/llm_judge_gptoss/llm_judge_per_summary.csv `
        --human "rHuman_Evals_main/selected_blue_..._alex.xlsx=BLUE" `
                "rHuman_Evals_main/selected_blue_..._sophie.csv=BLUE" `
                "rHuman_Evals_main/N_BLUE.xlsx=BLUE" `
                [... all rater files for all systems ...] `
        --judgment_cols q1_overall_quality,q2_no_contradiction,q3_coverage,q3a_dominant_self_state,q3b_element_recall,q3c_unsupported_elements,q3d_polarity_correctness `
        --human_lower_is_better unsupported `
        --output_dir ./outputs/judge_agreement
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import f as fdist
from scipy.stats import pearsonr, spearmanr

JUDGE_COLS_DEFAULT = [
    "q1_overall_quality", "q2_no_contradiction", "q3_coverage",
    "q3a_dominant_self_state", "q3b_element_recall",
    "q3c_unsupported_elements", "q3d_polarity_correctness",
]

# Column aliases: requested name -> acceptable alternatives
COL_ALIASES = {
    "q3c_unsupported_elements": ["q3c_unsupported_elements", "q3b_unsupported_elements"],
}


# ── ICC (validated against Shrout & Fleiss 1979) ──────────────────────────────

def icc_full(X: np.ndarray, alpha: float = 0.05) -> dict:
    """All six ICCs plus 95% CIs for an n-items x k-raters matrix (k=2 here)."""
    X = np.asarray(X, float)
    n, k = X.shape
    gm = X.mean()
    SSR = k * ((X.mean(1) - gm) ** 2).sum()
    SSC = n * ((X.mean(0) - gm) ** 2).sum()
    SST = ((X - gm) ** 2).sum()
    SSE = SST - SSR - SSC
    SSW = SST - SSR
    msb = SSR / (n - 1)
    msj = SSC / (k - 1)
    mse = SSE / ((n - 1) * (k - 1))
    msw = SSW / (n * (k - 1))
    dfb, dfe, dfw = n - 1, (n - 1) * (k - 1), n * (k - 1)

    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icc2 = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
    icc3 = (msb - mse) / (msb + (k - 1) * mse)
    icc1k = (msb - msw) / msb
    icc2k = (msb - mse) / (msb + (msj - mse) / n)
    icc3k = (msb - mse) / msb

    fl = (msb / mse) / fdist.ppf(1 - alpha / 2, dfb, dfe)
    fu = (msb / mse) * fdist.ppf(1 - alpha / 2, dfe, dfb)
    icc3_lo, icc3_hi = (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)

    a = k * icc2 / (n * (1 - icc2))
    b = 1 + k * icc2 * (n - 1) / (n * (1 - icc2))
    v = (a * msj + b * mse) ** 2 / ((a * msj) ** 2 / (k - 1) + (b * mse) ** 2 / dfe)
    f2u = fdist.ppf(1 - alpha / 2, n - 1, v)
    f2l = fdist.ppf(1 - alpha / 2, v, n - 1)
    icc2_lo = n * (msb - f2u * mse) / (f2u * (k * msj + (k * n - k - n) * mse) + n * msb)
    icc2_hi = n * (f2l * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f2l * msb)

    p_val = float(fdist.sf(msb / mse, dfb, dfe))

    return {
        "ICC2_1": icc2, "ICC2_1_lo": icc2_lo, "ICC2_1_hi": icc2_hi,
        "ICC3_1": icc3, "ICC3_1_lo": icc3_lo, "ICC3_1_hi": icc3_hi,
        "ICC2_k": icc2k, "ICC3_k": icc3k, "ICC1_1": icc1, "ICC1_k": icc1k,
        "F_p_value": p_val,
    }


# ── Loading / coercion ────────────────────────────────────────────────────────

def load_ratings(path: str) -> pd.DataFrame:
    """Load a .xlsx, .xls, .csv, or .json file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return pd.read_json(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric, parsing 'X/Y' fractions and recovering datetime-corrupted
    fractions (Excel auto-converts 4/4 to April 4). NS/blank → NaN.

    Handles three datetime corruption modes:
    1. Series dtype is datetime64 (all values corrupted) — pd.to_numeric gives
       huge nanosecond integers; we recover month/day from each value.
    2. Series dtype is object with mixed Timestamp + string values.
    3. Series dtype is object with Python datetime.datetime values.
    """
    # If the entire series is datetime64, pd.to_numeric would give nanosecond
    # integers (e.g. 1.7e18). Detect and recover BEFORE calling to_numeric.
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.Series(np.nan, index=s.index, dtype=float)
        for idx in s.index:
            v = s[idx]
            if pd.notna(v):
                ts = pd.Timestamp(v)
                m, d = ts.month, ts.day
                out[idx] = m / d if d != 0 else np.nan
        return out

    out = pd.to_numeric(s, errors="coerce")
    mask = out.isna() & s.notna()
    # Also catch values where to_numeric "succeeded" but gave a huge number
    # (datetime serial). These are > 1e6; fractions should be in [0, 1].
    if out.notna().any():
        suspicious = out.abs() > 1e6
        mask = mask | suspicious
    if mask.any():
        for idx in mask[mask].index:
            v = s[idx]
            # Recover from datetime corruption
            if isinstance(v, (pd.Timestamp,)):
                m, d = v.month, v.day
                out[idx] = m / d if d != 0 else np.nan
                continue
            import datetime as _dt
            if isinstance(v, _dt.datetime):
                m, d = v.month, v.day
                out[idx] = m / d if d != 0 else np.nan
                continue
            parts = str(v).strip().split("/")
            if len(parts) == 2:
                try:
                    num, den = float(parts[0]), float(parts[1])
                    out[idx] = num / den if den != 0 else np.nan
                except ValueError:
                    pass
    return out


def resolve_col(df: pd.DataFrame, requested: str) -> str | None:
    """Find the actual column name, tolerating aliases/typos."""
    if requested in df.columns:
        return requested
    if requested in COL_ALIASES:
        for alt in COL_ALIASES[requested]:
            if alt in df.columns:
                return alt
    return None


def load_human_files(human_specs: list[str], judgment_cols: list[str],
                     id_col: str = "post_id") -> pd.DataFrame:
    """Load multiple human rater files with path=label syntax, coerce all judgment
    columns to numeric, drop duplicates, and return a stacked DataFrame with
    columns: system, post_id, + judgment_cols."""
    frames = []
    for spec in human_specs:
        path, _, label = spec.partition("=")
        path, label = path.strip(), (label.strip() or None)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        df = load_ratings(path)
        df.columns = df.columns.astype(str).str.strip()

        # Drop duplicate rows (some files have doubled posts)
        if id_col in df.columns and df.duplicated(subset=[id_col]).any():
            n_before = len(df)
            df = df.drop_duplicates(subset=[id_col], keep="first")
            print(f"    Dropped {n_before - len(df)} duplicate rows from {os.path.basename(path)}")

        # Coerce judgment columns
        for c in judgment_cols:
            actual = resolve_col(df, c)
            if actual is not None:
                df[c] = coerce_numeric(df[actual])
                if actual != c:
                    print(f"    Column alias: using '{actual}' for '{c}' in {os.path.basename(path)}")

        # Stamp system label
        if label is not None:
            df["system"] = label
        elif "system" not in df.columns:
            raise SystemExit(f"File {path} has no 'system' column and no =label given.")

        # Keep only needed columns
        keep = ["system", id_col] + [c for c in judgment_cols if c in df.columns]
        frames.append(df[keep])
        print(f"  loaded {label or '?'}: {len(df)} rows from {os.path.basename(path)}")

    if not frames:
        raise SystemExit("No human files loaded.")
    return pd.concat(frames, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICC (and Spearman/Pearson) between LLM-judge and human consensus."
    )
    parser.add_argument("--human", nargs="+", required=True,
                        help="Human rater files: 'path=label' (repeatable). "
                             "Multiple files for the same system are averaged to consensus.")
    parser.add_argument("--judge", required=True,
                        help="LLM-judge per-summary CSV (must have 'system' column).")
    parser.add_argument("--judgment_cols", default=None,
                        help="Comma-separated question column names (default: q1–q3d).")
    parser.add_argument("--id_col", default="post_id",
                        help="Join key column (default: post_id).")
    parser.add_argument("--human_lower_is_better", default=None,
                        help="Comma-separated substrings for lower-is-better human columns "
                             "(e.g. 'unsupported'). Matching columns are negated so "
                             "higher=better for both judge and human before ICC.")
    parser.add_argument("--judge_system_map", default=None,
                        help="Comma-separated old=new pairs to relabel the judge file's "
                             "system column (e.g. 'Official_baseline=Official_Baseline'). "
                             "Use when judge and human files use different capitalization.")
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    judgment_cols = (args.judgment_cols.split(",") if args.judgment_cols
                     else JUDGE_COLS_DEFAULT)
    judgment_cols = [c.strip() for c in judgment_cols]
    lower_subs = tuple(s.strip() for s in args.human_lower_is_better.split(",")) if args.human_lower_is_better else ()

    # ── Load human files ──
    print("Loading human rater files:")
    human = load_human_files(args.human, judgment_cols, args.id_col)

    # Average to consensus per (system, post_id)
    human_cons = human.groupby(["system", args.id_col], dropna=False)[judgment_cols].mean()
    n_human = len(human_cons)
    print(f"\nHuman consensus: {n_human} (system, post) pairs "
          f"(from {len(human)} rater rows)")

    # Apply lower-is-better orientation to human
    if lower_subs:
        for c in human_cons.columns:
            if any(s in c.lower() for s in lower_subs):
                human_cons[c] = -human_cons[c]
                print(f"  Human column '{c}' negated (lower-is-better)")

    # ── Load judge file ──
    judge = load_ratings(args.judge)
    judge.columns = judge.columns.astype(str).str.strip()
    print(f"\nJudge file: {len(judge)} rows, systems: {list(judge['system'].unique())}")

    # Relabel judge system names if needed (e.g. capitalization mismatch)
    if args.judge_system_map:
        remap = {}
        for pair in args.judge_system_map.split(","):
            old, _, new = pair.partition("=")
            remap[old.strip()] = new.strip()
        judge["system"] = judge["system"].map(lambda s: remap.get(str(s), str(s)))
        print(f"  Relabeled judge systems: {remap}")
        print(f"  Judge systems after relabel: {list(judge['system'].unique())}")

    # Coerce judge judgment columns (should already be numeric, but just in case)
    for c in judgment_cols:
        if c in judge.columns:
            judge[c] = coerce_numeric(judge[c])

    # Average judge per (system, post_id) — in case of duplicates
    judge_cons = judge.groupby(["system", args.id_col], dropna=False)[judgment_cols].mean()

    # Apply lower-is-better orientation to judge (same columns)
    if lower_subs:
        for c in judge_cons.columns:
            if any(s in c.lower() for s in lower_subs):
                judge_cons[c] = -judge_cons[c]

    # ── Join ──
    merged = human_cons.join(judge_cons, how="inner", lsuffix="_h", rsuffix="_j")
    # The join creates duplicated columns; pick the right ones
    # Actually both have the same column names, so after join they get suffixes.
    # Let's do it more carefully:
    merged = pd.DataFrame(index=human_cons.index)
    for c in judgment_cols:
        merged[c + "_human"] = human_cons[c]
        merged[c + "_judge"] = judge_cons[c]
    merged = merged.dropna()

    n_paired = len(merged)
    print(f"\nPaired items (system, post_id) with both human and judge: {n_paired}")
    if n_paired < 3:
        raise SystemExit(f"Only {n_paired} paired items — need >= 3. Check system labels and post_ids.")

    # ── Compute ICC + correlations per question ──
    rows = []
    for c in judgment_cols:
        h = merged[c + "_human"]
        j = merged[c + "_judge"]
        pair = pd.DataFrame({"human": h, "judge": j}).dropna()
        n = len(pair)
        if n < 3:
            print(f"\n[{c}] only {n} paired items — skipped.")
            continue
        if pair["human"].nunique() < 2 or pair["judge"].nunique() < 2:
            print(f"\n[{c}] insufficient variation (n={n}) — skipped.")
            continue

        X = pair[["human", "judge"]].to_numpy()
        icc = icc_full(X)
        sr, sp = spearmanr(pair["human"], pair["judge"])
        pr, pp = pearsonr(pair["human"], pair["judge"])

        rows.append({
            "question": c, "n": n,
            "human_mean": pair["human"].mean(), "judge_mean": pair["judge"].mean(),
            "mean_abs_diff": (pair["human"] - pair["judge"]).abs().mean(),
            "spearman_rho": sr, "spearman_p": sp,
            "pearson_r": pr, "pearson_p": pp,
            **icc,
        })

        print(f"\n[{c}]  n={n}")
        print(f"  ICC(2,1) absolute  = {icc['ICC2_1']:.3f}  "
              f"95% CI [{icc['ICC2_1_lo']:.3f}, {icc['ICC2_1_hi']:.3f}]  p={icc['F_p_value']:.4g}")
        print(f"  ICC(3,1) consistency = {icc['ICC3_1']:.3f}  "
              f"95% CI [{icc['ICC3_1_lo']:.3f}, {icc['ICC3_1_hi']:.3f}]")
        print(f"  ICC(2,k)={icc['ICC2_k']:.3f}  ICC(3,k)={icc['ICC3_k']:.3f}")
        print(f"  Spearman={sr:+.3f} (p={sp:.4g})   Pearson={pr:+.3f} (p={pp:.4g})")
        print(f"  means: human={pair['human'].mean():.2f}  judge={pair['judge'].mean():.2f}  "
              f"|diff|={(pair['human']-pair['judge']).abs().mean():.2f}")

    if not rows:
        raise SystemExit("No question had >= 3 paired items with variation; nothing written.")

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "judge_agreement.csv")
    out.round(6).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
