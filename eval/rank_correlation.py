"""
Rank-correlation analysis between automated-metric rankings and human rankings.

This follows the MSLR precedent (Wang et al., 2023, "Automated Metrics for Medical
Multi-Document Summarization Disagree with Human Evaluations",
https://arxiv.org/pdf/2305.13693): build a system-level ranking from each
automated metric by averaging its instance-level values across all instances for
each system, then compute the Spearman rank correlation (rho) between every
metric's system ranking and the human ranking (and between metrics). The headline
is whether the NLI-based CLPsych metrics order systems the way humans do.

Three units of comparison are supported (see --level), matching the project's needs:

  conditions  The repository's own conditions, scored by the local eval harness.
              Input is one or more long-format results CSVs written by
              clpsych2025/evaluation/run.py (columns: timeline_id, post_id,
              metric, task, value, team_name, submission_id). A "system" is a
              (team_name, submission_id) pair by default; values are averaged
              across instances per system per metric. This is the natural place
              for the EAIonFlux reproduction, the degenerate baselines, and the
              label-source x retrieval sweep to be ranked side by side.

  teams       The official shared-task teams, supplied as a pre-aggregated wide
              CSV via --metrics_csv (one row per team; first column = team id;
              remaining columns = system-level metric means). Use this to mirror
              MSLR's "leaderboard systems" setting when you only have published
              system-level numbers rather than per-instance scores.

  instances   Pooled per-summary correlation (the powered companion to the
              system-level result, which is limited to a handful of systems).
              Joins each summary's per-instance metric value (from the results
              CSV, keyed on post_id for Task B / timeline_id for Task C) to that
              same summary's human rating, then reports one Spearman rho (with a
              real p-value over many summaries) per (metric, human judgment). Needs
              --results plus a PER-SUMMARY --human file (system + per-summary id +
              judgment columns); see human_rankings_instance_template.csv.

When humans annotated only a subset of posts (e.g. 30 random posts per system),
the MSLR-faithful system-level comparison averages the metrics over THAT SAME
subset, not over all generated summaries. Pass --restrict_to_human with --level
conditions and a PER-SUMMARY --human file: the metrics are inner-joined to the
annotated summaries (restricting them to the subset), both metrics and human
ratings are aggregated to system means over that subset, and the run additionally
prints the pooled instance-level correlation from the same data — so one command
gives both the (descriptive, small-n) system-level ranks and the (powered,
many-summary) instance-level rho.

Human rankings come from a file you provide (--human), since they are not part of
this repository. See human_rankings_template.csv for the schema:
    * a system-id column (default `system`, set with --human_system_col) whose
      values match the system ids produced above, and
    * one or more human judgment columns (e.g. human_overall, human_fluency).
Use --human_kind to say whether those columns are scores (higher = better, the
default) or ranks (1 = best).

Metric/score direction: rankings are oriented so that rank 1 = best for every
source before correlating, so a positive rho always means "agrees with". By
default any metric whose name contains 'contradiction' or 'mse' is treated as
lower-is-better; everything else as higher-is-better. Override with
--lower_is_better / --higher_is_better. Inferred directions are printed so you
can sanity-check them.

Outputs (to --output_dir):
    system_scores.csv          systems x sources, oriented so higher = better
    system_ranks.csv           systems x sources, integer ranks (1 = best)  [MSLR Table 1]
    spearman_rho_matrix.csv    sources x sources Spearman rho               [MSLR Figure 1]
    spearman_pvalue_matrix.csv sources x sources two-sided p-values
    spearman_heatmap.png       optional rho heatmap (with --heatmap)

Usage:
    # Rank internal conditions from local eval output, Task B only, vs your human file
    python rank_correlation.py --level conditions --task B \
        --results clpsych2025/evaluation/results/results_dev_<ts>.csv \
        --human my_human_rankings.csv --heatmap

    # Rank the official teams from a published system-level metrics table
    python rank_correlation.py --level teams \
        --metrics_csv official_team_metrics.csv \
        --human official_team_human.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Substrings that flag a metric as lower-is-better (rank 1 = lowest value).
DEFAULT_LOWER_IS_BETTER = ("contradiction", "mse")


# ── Load the system x metric table ────────────────────────────────────────────

def load_conditions(results_paths: list[str], task: str, system_key: str) -> pd.DataFrame:
    """
    Aggregate long-format eval CSV(s) into a wide system x metric table of means.

    Each row of the input is one (instance, metric) value tagged with team_name /
    submission_id. We average `value` per system per metric.
    """
    frames = [pd.read_csv(p) for p in results_paths]
    df = pd.concat(frames, ignore_index=True)

    required = {"metric", "value", "team_name", "submission_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"results CSV is missing columns {missing}; got {list(df.columns)}")

    if task != "all":
        if "task" not in df.columns:
            raise SystemExit("--task was set but the results CSV has no 'task' column.")
        df = df[df["task"].astype(str) == task]
        if df.empty:
            raise SystemExit(f"No rows with task == {task!r} in the results CSV(s).")

    if system_key == "team_name":
        df["system"] = df["team_name"].astype(str)
    elif system_key == "submission_id":
        df["system"] = df["submission_id"].astype(str)
    else:  # both
        df["system"] = df["team_name"].astype(str) + "_" + df["submission_id"].astype(str)

    wide = (df.groupby(["system", "metric"])["value"].mean()
              .unstack("metric"))
    return wide


def load_teams(metrics_csv: str) -> pd.DataFrame:
    """Load a wide system-level metrics table (one row per team; first col = team id)."""
    df = pd.read_csv(metrics_csv)
    id_col = df.columns[0]
    df = df.set_index(id_col)
    df.index.name = "system"
    # Keep only numeric metric columns.
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise SystemExit(f"No numeric metric columns found in {metrics_csv}.")
    return numeric


# ── Human rankings ────────────────────────────────────────────────────────────

def _read_human_files(human_paths, system_col="system") -> pd.DataFrame:
    """Read and row-concatenate one or more human rating files (CSV/JSON).

    Each entry may be 'path' or 'path=label'. When a label is given, a system
    column named system_col is stamped with that label for all rows from that
    file — use this when per-system files carry their system identity in the
    filename rather than in a column.
    """
    if isinstance(human_paths, str):
        human_paths = [human_paths]
    frames = []
    for spec in human_paths:
        path, _, label = spec.partition("=")
        path = path.strip()
        label = label.strip() or None
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            f = pd.read_json(path)
        elif ext in (".xlsx", ".xls"):
            f = pd.read_excel(path)
        else:
            # Try UTF-8 first (standard), fall back to cp1252 (Excel's default
            # on Windows — many human-annotated CSVs are exported from Excel).
            try:
                f = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                f = pd.read_csv(path, encoding="cp1252")
        f.columns = f.columns.astype(str).str.strip()  # tolerate stray header whitespace
        if label is not None:
            f[system_col] = label
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _parse_alias(spec: str) -> dict:
    """Parse a 'human=results,human=results,...' alias spec into a dict."""
    alias = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--system_alias entries must be 'human=results'; got {pair!r}")
        k, v = pair.split("=", 1)
        alias[k.strip()] = v.strip()
    return alias


def _apply_alias(index, alias):
    """Relabel human system ids to results system ids before joining."""
    if alias:
        return index.map(lambda s: alias.get(str(s), str(s)))
    return index


def _coerce_judgment_series(s: pd.Series) -> pd.Series:
    """Coerce a human judgment column to numeric, parsing 'X/Y' fractions.

    Handles rating columns stored as fraction strings (e.g. '3/5' element recall)
    that pd.to_numeric would silently drop to NaN. 'X/0' yields NaN (undefined).
    Also handles Excel date-corrupted fractions: when a user types '4/4' in
    Excel, it may be auto-converted to a date (e.g. 2026-04-04). We recover the
    fraction from month/day. Handles datetime64 dtype (all values corrupted),
    mixed object dtype (some Timestamp + some string), and Python datetime.
    """
    # If the entire series is datetime64, recover BEFORE to_numeric (which would
    # give nanosecond integers).
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
    # (datetime serial). Fractions should be in [0, 1]; 1-5 for Likert.
    if out.notna().any():
        suspicious = out.abs() > 1e6
        mask = mask | suspicious
    if mask.any():
        import datetime as _dt
        def _frac(v):
            # Recover from datetime corruption
            if isinstance(v, (pd.Timestamp,)):
                m, d = v.month, v.day
                return m / d if d != 0 else np.nan
            if isinstance(v, _dt.datetime):
                m, d = v.month, v.day
                return m / d if d != 0 else np.nan
            parts = str(v).strip().split("/")
            if len(parts) == 2:
                try:
                    num, den = float(parts[0]), float(parts[1])
                except ValueError:
                    return np.nan
                return num / den if den != 0 else np.nan
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan
        parsed = s[mask].map(_frac)
        out.loc[mask] = parsed
        if parsed.notna().any():
            print(f"  Parsed 'X/Y' fraction strings -> proportions in column "
                  f"'{s.name}' ({parsed.notna().sum()} values).")
    return out


def _select_judgments(df: pd.DataFrame, judgment_cols) -> pd.DataFrame:
    """Return human judgment columns as a numeric DataFrame.

    If judgment_cols is given, use exactly those (coerced to numeric, with 'X/Y'
    fraction strings parsed to proportions); otherwise fall back to all numeric
    columns. Pass judgment_cols to EXCLUDE numeric metadata (post_index,
    well_being_score, ...) from being treated as ratings.

    Tolerates common column-name typos: 'q3b_unsupported_elements' is accepted
    as an alias for 'q3c_unsupported_elements' (a typo found in one rater file).
    """
    # Column alias map: requested name -> list of acceptable actual names
    ALIASES = {
        "q3c_unsupported_elements": ["q3c_unsupported_elements", "q3b_unsupported_elements"],
    }
    if judgment_cols:
        actual_cols = []
        missing = []
        for c in judgment_cols:
            if c in df.columns:
                actual_cols.append(c)
            elif c in ALIASES:
                found = [a for a in ALIASES[c] if a in df.columns]
                if found:
                    actual_cols.append(found[0])
                    if found[0] != c:
                        print(f"  Column alias: using '{found[0]}' for requested '{c}'.")
                else:
                    missing.append(c)
            else:
                missing.append(c)
        if missing:
            raise SystemExit(f"--human_judgment_cols not found in human file: {missing}")
        sub = df[actual_cols].apply(_coerce_judgment_series)
        # Restore requested names so downstream columns match
        sub.columns = judgment_cols
    else:
        sub = df.select_dtypes(include="number")
    return sub.dropna(axis=1, how="all")


def load_human(human_paths, system_col: str, kind: str,
               judgment_cols=None, alias=None, lower_is_better=()) -> pd.DataFrame:
    """
    Load human judgment columns indexed by system id, oriented so higher = better.
    Accepts one or more files; multiple rows per system (e.g. several annotators)
    are averaged to the per-system consensus.

    kind == 'score'  values are scores, higher is better (kept as-is), unless a
                     column name matches a lower_is_better substring (then negated).
    kind == 'rank'   values are ranks, 1 = best (negated so higher = better);
                     lower_is_better is ignored in this mode.

    judgment_cols restricts which columns are treated as judgments (default: all
    numeric). alias relabels human system ids to match the results system ids
    formed by --system_key, so human rows can join to metric rows. lower_is_better
    is a tuple of substrings flagging score-kind columns where lower = better.
    """
    df = _read_human_files(human_paths, system_col)

    if system_col not in df.columns:
        raise SystemExit(
            f"Human file has no '{system_col}' column (columns: {list(df.columns)}). "
            f"Set --human_system_col."
        )
    df = df.set_index(system_col)
    df.index = _apply_alias(df.index.astype(str), alias)
    df.index.name = "system"

    numeric = _select_judgments(df, judgment_cols)
    if numeric.empty:
        raise SystemExit("No numeric human judgment columns found in the human file(s).")

    # Collapse multiple rows per system (annotators / per-system means) to the mean.
    n_before = len(numeric)
    numeric = numeric.groupby(level="system").mean()
    if len(numeric) < n_before:
        print(f"Averaged {n_before} human rows -> {len(numeric)} systems (annotator consensus).")

    if kind == "rank":
        numeric = -numeric  # lower rank = better -> negate so higher = better
    elif lower_is_better:
        # Score kind: negate columns flagged lower-is-better so all are higher=better.
        flip = [c for c in numeric.columns
                if any(s in c.lower() for s in lower_is_better)]
        if flip:
            numeric = numeric.copy()
            numeric[flip] = -numeric[flip]
            print(f"Human columns oriented lower-is-better (negated): {flip}")
    # Prefix to keep human columns visually distinct from metric columns.
    return numeric.add_prefix("human::")


# ── Instance-level loaders (per-summary, not per-system) ──────────────────────

def load_conditions_instances(results_paths: list[str], task: str, system_key: str,
                              instance_key: str) -> pd.DataFrame:
    """
    Per-instance metric table from long-format eval CSV(s): index = (system,
    instance), columns = metrics. `instance_key` is the results-CSV column that
    identifies a scored unit (post_id for Task B; timeline_id for Task C).
    """
    frames = [pd.read_csv(p) for p in results_paths]
    df = pd.concat(frames, ignore_index=True)

    required = {"metric", "value", "team_name", "submission_id", instance_key}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"results CSV is missing columns {missing}; got {list(df.columns)}")

    if task != "all":
        if "task" not in df.columns:
            raise SystemExit("--task was set but the results CSV has no 'task' column.")
        df = df[df["task"].astype(str) == task]

    df = df[df[instance_key].notna() & (df[instance_key].astype(str).str.strip() != "")]
    if df.empty:
        raise SystemExit(f"No per-instance rows with a non-empty {instance_key!r}. "
                         f"For Task C, pass --instance_key timeline_id.")

    if system_key == "team_name":
        df["system"] = df["team_name"].astype(str)
    elif system_key == "submission_id":
        df["system"] = df["submission_id"].astype(str)
    else:
        df["system"] = df["team_name"].astype(str) + "_" + df["submission_id"].astype(str)
    df["instance"] = df[instance_key].astype(str)

    wide = (df.groupby(["system", "instance", "metric"])["value"].mean()
              .unstack("metric"))
    return wide


def load_human_instances(human_paths, system_col: str, id_col: str,
                         kind: str, judgment_cols=None, alias=None,
                         lower_is_better=()) -> pd.DataFrame:
    """
    Per-summary human judgments: index = (system, instance), columns = human::*,
    oriented so higher = better. Accepts one or more files; multiple ratings of
    the same (system, post) — e.g. several annotators, whether in one file or in
    separate per-annotator files — are averaged to the per-summary consensus.
    The file(s) need a system column and a per-summary id column.

    judgment_cols restricts which columns are treated as judgments (default: all
    numeric minus the id columns). alias relabels human system ids to match the
    results system ids formed by --system_key.
    """
    df = _read_human_files(human_paths, system_col)

    for col in (system_col, id_col):
        if col not in df.columns:
            raise SystemExit(f"Per-summary human file needs a {col!r} column "
                             f"(columns: {list(df.columns)}). Set --human_system_col / --human_id_col.")
    sys_idx = _apply_alias(df[system_col].astype(str), alias)
    inst_idx = df[id_col].astype(str)
    if judgment_cols:
        numeric = _select_judgments(df, judgment_cols)
    else:
        # Drop the id columns before selecting judgments so a numeric id can't be
        # mistaken for a rating.
        numeric = df.drop(columns=[system_col, id_col]).select_dtypes(include="number")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        raise SystemExit("No numeric human judgment columns found in the human file(s).")
    numeric.index = pd.MultiIndex.from_arrays([sys_idx, inst_idx], names=["system", "instance"])

    # Collapse multiple annotator rows per (system, instance) to the mean (consensus).
    n_before = len(numeric)
    numeric = numeric.groupby(level=["system", "instance"]).mean()
    if len(numeric) < n_before:
        raters = n_before / max(len(numeric), 1)
        print(f"Averaged {n_before} human ratings -> {len(numeric)} unique summaries "
              f"(annotator consensus, ~{raters:.1f} raters/summary).")

    if kind == "rank":
        numeric = -numeric
    elif lower_is_better:
        flip = [c for c in numeric.columns
                if any(s in c.lower() for s in lower_is_better)]
        if flip:
            numeric = numeric.copy()
            numeric[flip] = -numeric[flip]
            print(f"Human columns oriented lower-is-better (negated): {flip}")
    return numeric.add_prefix("human::")


def load_merged_instances(args) -> tuple[pd.DataFrame, list, list]:
    """
    Inner-join per-instance metrics and per-summary human ratings on
    (system, instance). The inner join restricts the metrics to exactly the
    summaries that were human-annotated. Returns (merged, metric_cols, human_cols).
    """
    if not args.results:
        raise SystemExit("This mode requires --results <csv> [csv ...].")
    if not args.human:
        raise SystemExit("This mode requires a PER-SUMMARY --human file "
                         "(system + per-summary id + judgment columns; "
                         "see human_rankings_instance_template.csv).")
    id_col = args.human_id_col or args.instance_key
    metrics = load_conditions_instances(args.results, args.task, args.system_key, args.instance_key)
    human = load_human_instances(args.human, args.human_system_col, id_col, args.human_kind,
                                 args.human_judgment_cols, args.alias_map,
                                 args.human_lower_is_better)
    merged = metrics.join(human, how="inner")
    if merged.empty:
        raise SystemExit("No summaries matched. Check that system ids and "
                         f"{id_col!r} line up between the results CSV and the human file.")
    return merged, list(metrics.columns), list(human.columns)


def instance_level_report(merged, metric_cols, human_cols, lower, higher,
                          output_dir, task_label, heatmap=False):
    """Pooled per-summary Spearman of each metric vs each human judgment; prints + saves."""
    oriented, directions = orient_metrics(merged, lower, higher)
    print("\nMetric directions (positive rho => metric agrees with human):")
    for c in metric_cols:
        print(f"  {c:<40} {directions[c]}-is-better")

    rows = []
    for hc in human_cols:
        for mc in metric_cols:
            pair = oriented[[mc, hc]].dropna()
            if len(pair) >= 3 and pair[mc].nunique() > 1 and pair[hc].nunique() > 1:
                r, p = spearmanr(pair[mc], pair[hc])
            else:
                r, p = np.nan, np.nan
            rows.append({"human": hc, "metric": mc, "rho": r, "p_value": p, "n": len(pair)})

    corr = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 70)
    print(f"INSTANCE-LEVEL SPEARMAN  (pooled over summaries; task={task_label})")
    print("=" * 70)
    for hc in human_cols:
        print(f"\n  vs {hc}:")
        sub = corr[corr["human"] == hc].sort_values("rho", ascending=False, na_position="last")
        for _, r in sub.iterrows():
            star = " *" if (not np.isnan(r["p_value"]) and r["p_value"] < 0.05) else ""
            rtxt = "  nan" if np.isnan(r["rho"]) else f"{r['rho']:+.3f}"
            ptxt = "nan" if np.isnan(r["p_value"]) else f"{r['p_value']:.4g}"
            print(f"    {r['metric']:<40} rho={rtxt}  p={ptxt}  n={int(r['n'])}{star}")

    os.makedirs(output_dir, exist_ok=True)
    corr.round(6).to_csv(os.path.join(output_dir, "instance_correlations.csv"), index=False)
    oriented.round(6).to_csv(os.path.join(output_dir, "instance_scores.csv"))
    print(f"\nWrote instance_correlations.csv and instance_scores.csv to {output_dir}/")

    if heatmap:
        # Wide rho grid: rows = metric, columns = human judgment. Each cell is
        # the Spearman rho of that metric vs that human judgment (pooled over
        # summaries). Square-capable save_heatmap handles the non-square shape.
        grid = (corr.pivot(index="metric", columns="human", values="rho")
                   .reindex(index=metric_cols, columns=human_cols))
        hm_path = os.path.join(output_dir, "instance_heatmap.png")
        save_heatmap(grid, hm_path,
                     title="Instance-level Spearman rho (metric x human judgment)")
        print(f"Wrote {hm_path}")


def run_instance_level(args, lower, higher):
    """--level instances: pooled per-summary Spearman across all annotated summaries."""
    merged, metric_cols, human_cols = load_merged_instances(args)
    print(f"Matched {len(merged)} summaries (system x {args.instance_key}) with both "
          f"metric and human scores.")
    instance_level_report(merged, metric_cols, human_cols, lower, higher,
                          args.output_dir, args.task, heatmap=args.heatmap)


# ── Orientation, ranking, correlation ─────────────────────────────────────────

def orient_metrics(table: pd.DataFrame, lower_substrs, higher_substrs) -> tuple[pd.DataFrame, dict]:
    """
    Return (oriented_table, directions) where every column is oriented so that
    higher = better, and directions maps column -> 'higher'/'lower'.
    Metric columns flagged lower-is-better are negated; human:: columns are
    already oriented higher = better by load_human.
    """
    oriented = table.copy()
    directions = {}
    for col in table.columns:
        if col.startswith("human::"):
            directions[col] = "higher"
            continue
        name = col.lower()
        lower = any(s in name for s in lower_substrs)
        higher = any(s in name for s in higher_substrs)
        is_lower = lower and not higher
        directions[col] = "lower" if is_lower else "higher"
        if is_lower:
            oriented[col] = -table[col]
    return oriented, directions


def ranks_table(oriented: pd.DataFrame) -> pd.DataFrame:
    """Integer-ish ranks (1 = best) per source; ties share the average rank."""
    # Higher oriented value = better -> rank 1 to the max.
    return oriented.rank(ascending=False, method="average")


def spearman_matrices(oriented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman rho and two-sided p-value matrices across the sources."""
    cols = list(oriented.columns)
    rho = pd.DataFrame(np.nan, index=cols, columns=cols)
    pval = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        rho.loc[a, a] = 1.0
        pval.loc[a, a] = 0.0
        for b in cols[i + 1:]:
            pair = oriented[[a, b]].dropna()
            if len(pair) >= 2 and pair[a].nunique() > 1 and pair[b].nunique() > 1:
                r, p = spearmanr(pair[a], pair[b])
            else:
                r, p = np.nan, np.nan
            rho.loc[a, b] = rho.loc[b, a] = r
            pval.loc[a, b] = pval.loc[b, a] = p
    return rho, pval


def save_heatmap(rho: pd.DataFrame, out_path: str, title: str = "Spearman rho"):
    """Save a heatmap of a rho DataFrame. Handles non-square matrices (e.g. the
    instance-level metric x human-judgment grid) as well as square ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = rho.shape
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n_cols), max(5, 0.45 * n_rows)))
    im = ax.imshow(rho.values.astype(float), vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(rho.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(rho.index, fontsize=7)
    for i in range(n_rows):
        for j in range(n_cols):
            v = rho.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MSLR-style Spearman rank correlation: metric rankings vs human rankings."
    )
    parser.add_argument("--level", choices=["conditions", "teams", "instances"], default="conditions",
                        help="System-level: rank internal conditions (--results) or official teams "
                             "(--metrics_csv). instances: pooled per-summary correlation (--results + "
                             "a per-summary --human file).")
    parser.add_argument("--results", nargs="+", default=None,
                        help="conditions: long-format results CSV(s) from run.py.")
    parser.add_argument("--metrics_csv", default=None,
                        help="teams: wide CSV, one row per team, first col = team id.")
    parser.add_argument("--task", choices=["B", "C", "all"], default="all",
                        help="conditions: restrict to a single task before aggregating. "
                             "Ranking B-only and C-only systems together is rarely meaningful, "
                             "so pick B or C unless your CSV is single-task.")
    parser.add_argument("--system_key", choices=["team_name", "submission_id", "both"],
                        default="both",
                        help="conditions: how to form the system id from the CSV.")
    parser.add_argument("--human", nargs="+", default=None,
                        help="Human rating file(s) (CSV/JSON). Each entry may be 'path' or "
                             "'path=label' — when a label is given, a system column is stamped "
                             "with that label for all rows from that file (use this when "
                             "per-system files carry their identity in the filename). Pass "
                             "several (e.g. one per annotator) and rows for the same "
                             "system/summary are averaged to the consensus. "
                             "See human_rankings_template.csv.")
    parser.add_argument("--human_system_col", default="system",
                        help="Name of the system-id column in the human file.")
    parser.add_argument("--human_kind", choices=["score", "rank"], default="score",
                        help="Are human columns scores (higher=better) or ranks (1=best)?")
    parser.add_argument("--human_lower_is_better", default=None,
                        help="Comma-separated substrings flagging human judgment "
                             "columns as lower-is-better (e.g. unsupported,contradiction). "
                             "Only applies when --human_kind score; ranks are already "
                             "1=best. Matching columns are negated so higher=better.")
    parser.add_argument("--instance_key", default="post_id",
                        help="instances: results-CSV column identifying a scored summary "
                             "(post_id for Task B, timeline_id for Task C).")
    parser.add_argument("--human_id_col", default=None,
                        help="instances: per-summary id column in the human file "
                             "(defaults to --instance_key).")
    parser.add_argument("--human_judgment_cols", default=None,
                        help="Comma-separated human judgment columns to use (e.g. "
                             "q1_overall_quality,q2_no_contradiction,q3_coverage). By default ALL "
                             "numeric columns are treated as judgments — set this to exclude numeric "
                             "metadata like post_index / well_being_score.")
    parser.add_argument("--system_alias", default=None,
                        help="Map human system labels to the results system ids formed by "
                             "--system_key, as comma-separated human=results pairs, e.g. "
                             "eaion_flux=aeion-flux_task-b-k5,BLUE=blue-task-b_blue-task-b. Use this "
                             "when the human files label systems differently than the results CSV.")
    parser.add_argument("--restrict_to_human", action="store_true",
                        help="conditions: use a PER-SUMMARY --human file and restrict the "
                             "metric averages to the human-annotated summaries only "
                             "(MSLR-faithful). Aggregates both metrics and human ratings to "
                             "system means over that subset, and also reports the pooled "
                             "instance-level correlation from the same data.")
    parser.add_argument("--lower_is_better", default=None,
                        help="Comma-separated substrings forcing lower-is-better "
                             f"(default: {','.join(DEFAULT_LOWER_IS_BETTER)}).")
    parser.add_argument("--higher_is_better", default=None,
                        help="Comma-separated substrings forcing higher-is-better "
                             "(overrides lower-is-better on a clash).")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also save spearman_heatmap.png.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Orientation config (rank 1 = best) is shared by every mode.
    lower = tuple(s.strip() for s in args.lower_is_better.split(",")) if args.lower_is_better \
        else DEFAULT_LOWER_IS_BETTER
    higher = tuple(s.strip() for s in args.higher_is_better.split(",")) if args.higher_is_better \
        else ()

    # Parsed human-side config shared by system-level and instance-level paths.
    args.alias_map = _parse_alias(args.system_alias) if args.system_alias else None
    args.human_judgment_cols = ([c.strip() for c in args.human_judgment_cols.split(",")]
                                if args.human_judgment_cols else None)
    args.human_lower_is_better = (tuple(s.strip() for s in args.human_lower_is_better.split(","))
                                  if args.human_lower_is_better else ())

    if args.level == "instances":
        run_instance_level(args, lower, higher)
        return

    restrict = args.level == "conditions" and args.restrict_to_human
    restrict_data = None  # (merged, metric_cols, human_cols) when restrict is on

    # ── Build the system x metric (+ human) table ──
    if restrict:
        merged, metric_cols, human_cols = load_merged_instances(args)
        n_sys = merged.index.get_level_values("system").nunique()
        print(f"Restricted to {len(merged)} annotated summaries across {n_sys} systems; "
              f"metrics and human ratings aggregated to system means over that subset.")
        table = merged.groupby(level="system").mean()
        restrict_data = (merged, metric_cols, human_cols)
    else:
        if args.level == "conditions":
            if not args.results:
                raise SystemExit("--level conditions requires --results <csv> [csv ...].")
            metrics = load_conditions(args.results, args.task, args.system_key)
        else:
            if not args.metrics_csv:
                raise SystemExit("--level teams requires --metrics_csv <csv>.")
            metrics = load_teams(args.metrics_csv)

        # ── Attach human rankings (joined on system id) ──
        table = metrics
        human_cols = []
        if args.human:
            human = load_human(args.human, args.human_system_col, args.human_kind,
                               args.human_judgment_cols, args.alias_map,
                               args.human_lower_is_better)
            human_cols = list(human.columns)
            table = metrics.join(human, how="outer")
            matched = metrics.index.intersection(human.index)
            print(f"Joined human rankings: {len(matched)}/{len(metrics.index)} systems matched "
                  f"by id. Unmatched metric systems: {sorted(set(metrics.index) - set(human.index))}")
            if len(matched) < 3:
                print("WARNING: fewer than 3 systems matched the human file — Spearman rho will be "
                      "unstable or undefined. Check that system ids line up.")
        else:
            print("No --human file given: reporting metric-vs-metric agreement only "
                  "(the MSLR analysis without the human column).")

    # ── Orient everything so higher = better ──
    oriented, directions = orient_metrics(table, lower, higher)

    print("\nMetric directions (rank 1 = best):")
    for col in table.columns:
        tag = "human (higher=better)" if col.startswith("human::") else f"{directions[col]}-is-better"
        print(f"  {col:<40} {tag}")

    if len(table.index) < 3:
        print(f"\nWARNING: only {len(table.index)} systems — rank correlation needs >= 3 "
              "to be meaningful.")

    # ── Ranks table + correlation matrices ──
    ranks = ranks_table(oriented)
    rho, pval = spearman_matrices(oriented)

    # ── Report ──
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\n" + "=" * 70)
    print("SYSTEM RANKS (1 = best)   [MSLR Table 1 analogue]")
    print("=" * 70)
    print(ranks.round(1).to_string())

    print("\n" + "=" * 70)
    print("SPEARMAN rho BETWEEN RANKING SOURCES   [MSLR Figure 1 analogue]")
    print("=" * 70)
    print(rho.round(3).to_string())

    if human_cols:
        print("\n--- Headline: each automated metric vs human ranking(s) ---")
        metric_cols = [c for c in oriented.columns if not c.startswith("human::")]
        for hc in human_cols:
            print(f"\n  vs {hc}:")
            pairs = [(mc, rho.loc[mc, hc], pval.loc[mc, hc]) for mc in metric_cols]
            for mc, r, p in sorted(pairs, key=lambda t: (np.nan_to_num(t[1], nan=-2)), reverse=True):
                star = "" if (np.isnan(p)) else (" *" if p < 0.05 else "")
                rtxt = "  nan" if np.isnan(r) else f"{r:+.3f}"
                ptxt = "nan" if np.isnan(p) else f"{p:.3f}"
                print(f"    {mc:<40} rho={rtxt}  p={ptxt}{star}")

    # ── Save ──
    oriented.round(6).to_csv(os.path.join(args.output_dir, "system_scores.csv"))
    ranks.round(3).to_csv(os.path.join(args.output_dir, "system_ranks.csv"))
    rho.round(6).to_csv(os.path.join(args.output_dir, "spearman_rho_matrix.csv"))
    pval.round(6).to_csv(os.path.join(args.output_dir, "spearman_pvalue_matrix.csv"))
    print(f"\nWrote system_scores.csv, system_ranks.csv, spearman_rho_matrix.csv, "
          f"spearman_pvalue_matrix.csv to {args.output_dir}/")

    if args.heatmap:
        hm_path = os.path.join(args.output_dir, "spearman_heatmap.png")
        save_heatmap(rho, hm_path)
        print(f"Wrote {hm_path}")

    # ── Restrict mode also emits the pooled instance-level correlation ──
    if restrict_data is not None:
        print("\n" + "#" * 70)
        print("# POOLED INSTANCE-LEVEL CORRELATION (powered companion, n = summaries)")
        print("#" * 70)
        merged, metric_cols, human_cols = restrict_data
        instance_level_report(merged, metric_cols, human_cols, lower, higher,
                              args.output_dir, args.task, heatmap=args.heatmap)


if __name__ == "__main__":
    main()
