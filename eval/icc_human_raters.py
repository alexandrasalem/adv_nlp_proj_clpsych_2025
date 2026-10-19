"""
Inter-annotator agreement (ICC) across human raters for the CLPsych Task B
human evaluation.

Computes ICC(A,k) (two-way random, absolute agreement, average measures) and
ICC(A,1) (single-rater) per question, per system, and pooled across all
systems — following Shrout & Fleiss (1979) / McGraw & Wong (1996).

Supports any number of raters (k ≥ 2). Raters are identified by filename
(e.g. alex.xlsx, sophie.csv, nasrin.csv). Files can be .xlsx or .csv.

The script loads each rater's file per system, joins them by post_id into an
n-posts × k-raters matrix per question, drops posts where any rater has a
non-numeric value (NS, blank, NaN), and computes ICC on the remaining matrix.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import f as fdist


JUDGE_COLS_DEFAULT = [
    "q1_overall_quality", "q2_no_contradiction", "q3_coverage",
    "q3a_dominant_self_state", "q3b_element_recall",
    "q3c_unsupported_elements", "q3d_polarity_correctness",
]


def load_rater_file(path: str) -> pd.DataFrame:
    """Load a .xlsx, .xls, or .csv file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric, parsing 'X/Y' fractions to proportions. NS/blank → NaN.
    Also handles Excel date-corrupted fractions (X/Y auto-converted to month/day).
    Handles datetime64 dtype, mixed object dtype, and Python datetime."""
    # If the entire series is datetime64, recover BEFORE to_numeric
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
    # Also catch datetime serials (huge numbers)
    if out.notna().any():
        suspicious = out.abs() > 1e6
        mask = mask | suspicious
    if mask.any():
        import datetime as _dt
        for idx in mask[mask].index:
            v = s[idx]
            if isinstance(v, (pd.Timestamp,)):
                m, d = v.month, v.day
                out[idx] = m / d if d != 0 else np.nan
                continue
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


def icc_full(X: np.ndarray, alpha: float = 0.05) -> dict:
    """
    All six ICCs plus 95% CIs for the single-rater forms, for an n-items x k-raters
    matrix. Formulas: Shrout & Fleiss (1979) / McGraw & Wong (1996).
    Handles k ≥ 2 raters with complete data (no NaN — drop rows with any NaN
    before calling).
    """
    X = np.asarray(X, float)
    n, k = X.shape
    if n < 2 or k < 2:
        return {"error": f"n={n}, k={k} — need at least 2 items and 2 raters"}

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

    # ICC(3,1) CI
    fl = (msb / mse) / fdist.ppf(1 - alpha / 2, dfb, dfe)
    fu = (msb / mse) * fdist.ppf(1 - alpha / 2, dfe, dfb)
    icc3_lo = (fl - 1) / (fl + k - 1)
    icc3_hi = (fu - 1) / (fu + k - 1)

    # ICC(2,1) CI (McGraw & Wong)
    a = k * icc2 / (n * (1 - icc2))
    b = 1 + k * icc2 * (n - 1) / (n * (1 - icc2))
    v = (a * msj + b * mse) ** 2 / ((a * msj) ** 2 / (k - 1) + (b * mse) ** 2 / dfe)
    f2u = fdist.ppf(1 - alpha / 2, n - 1, v)
    f2l = fdist.ppf(1 - alpha / 2, v, n - 1)
    icc2_lo = n * (msb - f2u * mse) / (f2u * (k * msj + (k * n - k - n) * mse) + n * msb)
    icc2_hi = n * (f2l * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f2l * msb)

    p_val = float(fdist.sf(msb / mse, dfb, dfe))

    return {
        "n_items": n, "k_raters": k,
        "ICC_A1": icc2, "ICC_A1_lo": icc2_lo, "ICC_A1_hi": icc2_hi,  # two-way random, absolute, single
        "ICC_Ak": icc2k,  # two-way random, absolute, average (the headline number)
        "ICC_C1": icc3, "ICC_C1_lo": icc3_lo, "ICC_C1_hi": icc3_hi,  # two-way random, consistency, single
        "ICC_Ck": icc3k,  # two-way random, consistency, average
        "ICC_1k": icc1k,  # one-way random, average
        "F_p_value": p_val,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inter-annotator agreement (ICC) across human raters."
    )
    parser.add_argument("--rater", action="append", default=[],
                        help="'system=rater=path' or 'rater=path' (repeatable). "
                             "Files for the same system + different raters are joined "
                             "by post_id. If no system is given, 'pooled' is used. "
                             "Example: --rater 'BLUE=alex=path.xlsx' "
                             "--rater 'BLUE=sophie=path.csv'")
    parser.add_argument("--system_col", default=None,
                        help="Column name identifying the system (e.g. 'system'). If "
                             "provided, files are grouped by system. If omitted, all "
                             "files are treated as one pooled group.")
    parser.add_argument("--id_col", default="post_id",
                        help="Column name for the per-summary id (default: post_id).")
    parser.add_argument("--judgment_cols", default=None,
                        help="Comma-separated judgment column names (default: q1–q3d).")
    parser.add_argument("--drop_any_nan", action="store_true", default=True,
                        help="Drop posts where ANY rater has NaN/NS (default: True). "
                             "This gives a complete matrix for ICC.")
    parser.add_argument("--output_dir", default="./outputs/icc")
    args = parser.parse_args()

    if not args.rater:
        raise SystemExit("Pass at least one --rater 'name=path'.")

    judgment_cols = (args.judgment_cols.split(",") if args.judgment_cols
                     else JUDGE_COLS_DEFAULT)
    judgment_cols = [c.strip() for c in judgment_cols]

    # Load all rater files
    # Group by system if --system_col is provided, else treat as pooled
    # Each rater file may contain ratings for one or multiple systems
    all_data = []  # list of (rater_name, system_name, post_id, {col: value})
    for spec in args.rater:
        # Parse: system=rater=path OR rater=path (no system → "pooled")
        parts = spec.split("=", 2)
        if len(parts) == 3:
            sys_name, rater_name, path = parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            sys_name, rater_name, path = "pooled", parts[0].strip(), parts[1].strip()
        else:
            raise SystemExit(f"--rater entries must be 'system=rater=path' or 'rater=path'; got {spec!r}")

        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping rater {rater_name}.")
            continue
        df = load_rater_file(path)
        df.columns = df.columns.astype(str).str.strip()
        # Tolerate column-name typos: q3b_unsupported_elements -> q3c_unsupported_elements
        if "q3b_unsupported_elements" in df.columns and "q3c_unsupported_elements" not in df.columns:
            df = df.rename(columns={"q3b_unsupported_elements": "q3c_unsupported_elements"})
        # Drop duplicate rows (some files have doubled rows)
        if df.duplicated(subset=[args.id_col]).any() if args.id_col in df.columns else False:
            n_before = len(df)
            df = df.drop_duplicates(subset=[args.id_col], keep="first")
            print(f"    Dropped {n_before - len(df)} duplicate rows from {os.path.basename(path)}")
        # Coerce judgment columns to numeric
        for c in judgment_cols:
            if c in df.columns:
                df[c] = coerce_numeric(df[c])
        # Each row is one post rated by this rater for this system
        for _, row in df.iterrows():
            pid = str(row.get(args.id_col, ""))
            if pid:
                ratings = {c: row.get(c, np.nan) for c in judgment_cols if c in df.columns}
                all_data.append((rater_name, sys_name, pid, ratings))

    if not all_data:
        raise SystemExit("No data loaded.")

    # Build matrices: (system, question) -> {post_id -> {rater -> value}}
    from collections import defaultdict
    by_sys_q = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for rater, sys_name, pid, ratings in all_data:
        for q, val in ratings.items():
            by_sys_q[sys_name][q][pid][rater] = val

    # Also build a pooled-across-systems matrix per question
    by_q_pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for rater, sys_name, pid, ratings in all_data:
        # Use (system, post_id) as the unique item key for pooling
        item_key = f"{sys_name}#{pid}"
        for q, val in ratings.items():
            by_q_pooled[q][item_key][rater] = val

    raters = sorted(set(r for r, _, _, _ in all_data))
    systems = sorted(set(s for _, s, _, _ in all_data))
    print(f"Raters: {raters}")
    print(f"Systems: {systems}")
    print(f"Judgment columns: {judgment_cols}")
    print()

    results = []

    # Per-system ICC
    for sys_name in systems:
        print(f"\n{'=' * 60}")
        print(f"SYSTEM: {sys_name}")
        print(f"{'=' * 60}")
        for q in judgment_cols:
            post_raters = by_sys_q[sys_name].get(q, {})
            if not post_raters:
                continue
            # Build matrix: rows = posts, cols = raters
            # Drop posts where any rater has NaN
            matrix = {}
            n_dropped = 0
            for pid, rater_vals in post_raters.items():
                row = [rater_vals.get(r, np.nan) for r in raters]
                if any(np.isnan(v) for v in row):
                    n_dropped += 1
                    continue
                matrix[pid] = row

            n_items = len(matrix)
            if n_items < 2:
                print(f"  {q}: n={n_items} (after dropping {n_dropped} with NaN) — too few items")
                continue

            X = np.array(list(matrix.values()))
            icc = icc_full(X)
            if "error" in icc:
                print(f"  {q}: {icc['error']}")
                continue

            icc["system"] = sys_name
            icc["question"] = q
            results.append(icc)
            sig = "*" if icc["F_p_value"] < 0.05 else ""
            print(f"  {q:<35} n={n_items:3d} (dropped {n_dropped})  "
                  f"k={icc['k_raters']}  "
                  f"ICC(A,k)={icc['ICC_Ak']:.3f}  "
                  f"ICC(A,1)={icc['ICC_A1']:.3f} "
                  f"[{icc['ICC_A1_lo']:.3f}, {icc['ICC_A1_hi']:.3f}]  "
                  f"p={icc['F_p_value']:.4f}{sig}")

    # Pooled across all systems
    print(f"\n{'=' * 60}")
    print(f"POOLED (all systems)")
    print(f"{'=' * 60}")
    for q in judgment_cols:
        post_raters = by_q_pooled.get(q, {})
        if not post_raters:
            continue
        matrix = {}
        n_dropped = 0
        for item_key, rater_vals in post_raters.items():
            row = [rater_vals.get(r, np.nan) for r in raters]
            if any(np.isnan(v) for v in row):
                n_dropped += 1
                continue
            matrix[item_key] = row

        n_items = len(matrix)
        if n_items < 2:
            print(f"  {q}: n={n_items} — too few items")
            continue

        X = np.array(list(matrix.values()))
        icc = icc_full(X)
        if "error" in icc:
            print(f"  {q}: {icc['error']}")
            continue

        icc["system"] = "pooled"
        icc["question"] = q
        results.append(icc)
        sig = "*" if icc["F_p_value"] < 0.05 else ""
        print(f"  {q:<35} n={n_items:3d} (dropped {n_dropped})  "
              f"k={icc['k_raters']}  "
              f"ICC(A,k)={icc['ICC_Ak']:.3f}  "
              f"ICC(A,1)={icc['ICC_A1']:.3f} "
              f"[{icc['ICC_A1_lo']:.3f}, {icc['ICC_A1_hi']:.3f}]  "
              f"p={icc['F_p_value']:.4f}{sig}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "icc_human_raters.csv")
    # Reorder columns
    front = ["system", "question", "n_items", "k_raters",
             "ICC_Ak", "ICC_A1", "ICC_A1_lo", "ICC_A1_hi",
             "ICC_Ck", "ICC_C1", "ICC_C1_lo", "ICC_C1_hi",
             "ICC_1k", "F_p_value"]
    out_df = out_df.reindex(columns=front + [c for c in out_df.columns if c not in front])
    out_df.round(6).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\nInterpretation:")
    print("  ICC(A,k) = two-way random, absolute agreement, average measures")
    print("    — the headline number. ≥0.6 = adequate, ≥0.75 = good, ≥0.9 = excellent")
    print("  ICC(A,1) = single-rater form with 95% CI")
    print("  Posts with NaN/NS on any rater are dropped (complete-case ICC)")


if __name__ == "__main__":
    main()
