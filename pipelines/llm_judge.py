"""
LLM-as-a-judge: rate the 4 systems' summaries with the SAME instrument the human
annotators use, so judge scores can be joined to human ratings later (B5) via
`judge_agreement.py`.

The instrument combines two sources:
  - q1–q3 (overall quality, no-contradiction, coverage): 1-5 Likert with full
    anchors, taken VERBATIM from `prompt_judges.txt` — the human annotator rubric.
  - q3a–q3d (ABCD fidelity sub-questions): taken VERBATIM from the ABCD Fidelity
    Human Evaluation Prompt spec (docx). These use the CORRECT scales:
      q3a_dominant_self_state     0-2   (0=wrong, 1=partial, 2=match; NS=not stated)
      q3b_element_recall          "X/Y" fraction string (e.g. "3/5")
      q3c_unsupported_elements    int >= 0
      q3d_polarity_correctness    "X/Y" fraction string (e.g. "4/4")

The judge also receives the gold evidence annotations (the ABCD elements with
their adaptive/maladaptive polarity) per the docx spec: "Using the element
reference table and the post's Gold Evidence Annotations and Gold Summary."

The judge is a LOCAL model via `ollama` (the data-sharing agreement prohibits
cloud LLM APIs). Default: `gpt-oss:20b` with thinking enabled.

Only the 30 human-eval posts per system (120 total) are judged — the same posts
the human raters scored.

Usage:
    python llm_judge.py --model gpt-oss:20b --output_dir ./outputs/llm_judge_gptoss
"""

import argparse
import ast
import json
import os
import time

import numpy as np
import pandas as pd

# The 4 systems to judge. These match the human-eval set.
DEFAULT_SYSTEMS = [
    ("EAIonFlux", "outputs/eaion_flux/aeion-flux_task-b-k5.json"),
    ("BLUE", "outputs/BLUE/post_data_task_b_blue_baseline.csv"),
    ("Official_baseline", "outputs/Baseline/post_data_task_b_official_baseline.csv"),
    ("Ours", "clpsych2025/evaluation/official_run/random_structural.json"),
]

JUDGE_COLS = [
    "q1_overall_quality", "q2_no_contradiction", "q3_coverage",
    "q3a_dominant_self_state", "q3b_element_recall",
    "q3c_unsupported_elements", "q3d_polarity_correctness",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_evidence_dict(cell) -> dict:
    """Parse a post_data.csv adaptive/maladaptive_evidence cell into an
    {ABCD_key: {'Category':..., 'highlighted_evidence':...}} dict."""
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
    return {}


def load_post_data(post_data_csv: str) -> dict[str, dict]:
    """post_id -> {'post': str, 'gold': str, 'evidence': str}.
    'evidence' is a formatted text block of the gold ABCD evidence annotations."""
    df = pd.read_csv(post_data_csv)
    out = {}
    for _, row in df.iterrows():
        pid = str(row["post_id"])
        gold = str(row.get("post_summary", "") or "").strip()
        post = str(row.get("post", "") or "").strip()
        if not (gold and post):
            continue
        # Parse and format the gold evidence
        adaptive = _parse_evidence_dict(row.get("adaptive_evidence"))
        maladaptive = _parse_evidence_dict(row.get("maladaptive_evidence"))
        evidence_lines = []
        if adaptive:
            evidence_lines.append("Adaptive states:")
            for key, info in adaptive.items():
                cat = info.get("Category", "") if isinstance(info, dict) else ""
                ev = info.get("highlighted_evidence", "") if isinstance(info, dict) else str(info)
                evidence_lines.append(f"  - {key} ({cat}): \"{ev}\"")
        else:
            evidence_lines.append("Adaptive states: (none)")
        if maladaptive:
            evidence_lines.append("Maladaptive states:")
            for key, info in maladaptive.items():
                cat = info.get("Category", "") if isinstance(info, dict) else ""
                ev = info.get("highlighted_evidence", "") if isinstance(info, dict) else str(info)
                evidence_lines.append(f"  - {key} ({cat}): \"{ev}\"")
        else:
            evidence_lines.append("Maladaptive states: (none)")
        out[pid] = {
            "post": post,
            "gold": gold,
            "evidence": "\n".join(evidence_lines),
        }
    return out


def load_summaries(path: str, label: str) -> list[dict]:
    """Load summaries from official-format JSON or a CSV with a summary column.
    Returns list of {system, post_id, summary} (non-empty only)."""
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tl_id, tl in data.items():
            post_level = tl.get("post_level", {}) if isinstance(tl, dict) else {}
            for post_id, entry in post_level.items():
                summary = (entry or {}).get("summary", "") or ""
                if summary.strip():
                    rows.append({"system": label, "post_id": str(post_id), "summary": summary})
    elif ext in (".csv", ".xlsx", ".xls"):
        if ext == ".csv":
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="cp1252")
        else:
            df = pd.read_excel(path)
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
            if summary.strip() and post_id:
                rows.append({"system": label, "post_id": post_id, "summary": summary})
    else:
        raise SystemExit(f"Unsupported file type: {ext} ({path})")
    return rows


# ── Prompt + judge ───────────────────────────────────────────────────────────

# The combined rubric: q1-q3 from prompt_judges.txt (verbatim) + q3a-q3d from
# the ABCD Fidelity Human Evaluation Prompt spec (docx, verbatim).
RUBRIC = """You are an expert clinical psychologist. Your task is to evaluate an LLM-generated summary of a Reddit post that describes the interplay between adaptive and maladaptive self-states that occur in it. The summary should identify the dominant self-state, and describe how core aspects of it influence the rest.

The summary should describe the post author's self-state using the ABCD framework:
A (Affect): emotions and feelings
B (Behavior): actions and conduct toward self or others
C (Cognition): thoughts, beliefs, and perceptions about self or others
D (Desire/Need): psychological needs and motivations

A self-state is defined as:
Adaptive: positive emotions, behaviors, or thoughts that support the author's psychological needs
Maladaptive: negative emotions, behaviors, or thoughts that hinder the author's psychological needs

You are judging how faithfully a system summary reflects the ABCD elements that are actually present in this post, based on the gold Evidence annotations. An ABCD element is one of six psychological dimensions, each either adaptive or maladaptive:
A — Affect: the emotion expressed
  Adaptive examples: Calm/Laid back, Emotional Pain/Grieving, Content/Happy, Vigor/Energetic, Justifiable Anger/Assertive Anger, Proud
  Maladaptive examples: Anxious/Tense/Fearful, Depressed/Desperate/Hopeless, Mania, Apathetic/Don’t care/Blunted, Angry (Agressive, Disgust, Contempt), Ashamed/Guilty
B-O — Behavior toward Other: how they act toward others
  Adaptive examples: Requesting-Advice, Expressing-Emotions, Reaching-Out-for-Support, Sharing-Experiences, Relating behavior, Autonomous behavior
  Maladaptive examples: Avoidant Communication, Social Isolation, Aggressive Deflection, Fear of Abandonment, Fight or flight behavior, Overcontrolled/controlling behavior
B-S — Behavior toward Self: how they act toward themselves
  Adaptive examples: Self-Advocacy, Self-Improvement, Self-care, Positive Self-Talk
  Maladaptive examples: Self-Neglect, Substance Abuse, Self-Sabotage, Self-Isolation, Avoidance Behaviors, Self-Harm, Suicidal Intention or actions, Social Withdrawal
C-O — Cognition of Other: how they perceive others
  Adaptive examples: Perceptions of Support, Perception of the other as related, Perception of the other as facilitating autonomy/competence needs
  Maladaptive examples: Social Isolation, Feeling Unappreciated, Feeling Disrespected, Perception of the other as detached or over attached, Perception of the other as blocking autonomy needs
C-S — Cognition of Self: how they perceive themselves
  Adaptive examples: Self-Confidence, Self-Acceptance, Self-Validation, Self-Awareness, Self-compassion
  Maladaptive examples: Self-Blame, Self-criticism, Low Self-Esteem, Self-Loathing, Helplessness, Self-Doubt
D — Desire: what they want, need, fear, or expect
  Adaptive examples: Desire for Normalcy and Acceptance, Desire for Change and Improvement, Desire for Happiness and Fulfillment, Relatedness, Autonomy and adaptive control, Competence, Self-esteem, Self-care
  Maladaptive examples: Desire for death, Loss of motivation, Existential crisis, Hopelessness, Expectation that relatedness needs will not be met, Expectation that autonomy needs will not be met, Expectation that competence needs will not be met 

Important: a summary does NOT need to name these letters or use clinical terms. A summary that says "the maladaptive states include the author feeling on edge about the interview" HAS referenced Maladaptive Affect (anxiety), even though it never says "Affect" or "anxiety." Judge the content, not the vocabulary. Credit an ABCD element only if the summary conveys the person's actual internal state (what they feel, think, want, or how they act toward themselves/others) — not merely the external events or circumstances of the post; describing what happened without attributing a psychological state does not count.

Answer the following questions:

How would you rate the overall quality of the LLM-generated summary, in comparison to the gold standard summary?
Rate 1-5: 1) Poor, 2) Below Average, 3) Average, 4) Good, 5) Excellent

How would you rate the statement "the LLM-generated summary contains no contradictory information when compared with the gold standard summary"?
Rate 1-5: 1) Strongly disagree, 2) Disagree, 3) Neutral, 4) Agree, 5) Strongly agree

How would you rate the statement "the LLM-generated summary covers all the same information that is in the gold standard summary"?
Rate 1-5: 1) Strongly disagree, 2) Disagree, 3) Neutral, 4) Agree, 5) Strongly agree

Define quality: Quality considers both contradiction and coverage in relation to the gold standard in assessing the overall usefulness of the summary as a clinical description of the author's self-state(s). Good-quality summaries should not only effectively reference the information in the post, but should also clearly reference and define which self-states are present.
A rating of 5 means that the summary contains no contradictions and describes the same self-states as the gold standard.
A rating of 4 means that the summary contains no contradictions and covers most of the same information as the gold standard.
A rating of 3 means that the summary may contain mild contradictions but generally discusses the same self-states as the gold standard.
A rating of 2 means that there may be notable contradictions and/or the summary does not reference the proper self-states.
A rating of 1 means that there are severe contradictions and/or contains no informative or correct reference to self-states.

Define contradiction: A contradiction is a statement in a summary that directly contradicts a statement from the gold standard.
A rating of 5 means there is no information in the generated summary that contradicts the gold standard.
A rating of 4 means there is almost no information in the generated summary that contradicts the gold standard, and if there is, it is a detail that does not impede the overall description of self-states.
A rating of 3 means that there is information in the generated summary that is contradictory with the gold standard, but the overall summary is still consistent and clinically useful.
A rating of 2 means that there is a notable contradiction present hindering the usefulness of the summary.
A rating of 1 means there is a severe contradiction present.

Define covering same information: A generated summary covers the same information as the gold standard summary if the same self-states are identified, with the same adaptive/maladaptive label, and the same evidence or examples from the post for the self-states are identified.
A rating of 5 means almost exactly the same self-states and evidence were found.
A rating of 4 means there are some small differences in the self-states and evidence found.
A rating of 3 means some of the self-states or evidence overlapped but others were different.
A rating of 2 means most of the self-states or evidence was different, or limited reference to self-states was given.
A rating of 1 means entirely different self-states and evidence were given, or no references to self-states were made.

Q3a — Dominant self-state. Does the summary identify the same dominant self-state as the gold?
2 = match — names/conveys the same dominant state (adaptive / maladaptive / mixed)
1 = partial — conveys the right general direction but muddled, or calls a mixed state single (or vice versa)
0 = wrong — asserts the opposite dominant state
NS = not stated — never commits to a dominant state at all
Record "wrong" and "not stated" separately — do not merge them.

Q3b — Element recall (coverage of what's actually there). Look at each ABCD element present in the gold Evidence. For how many does the summary convey that element's content, in any wording?
Report as a fraction: (gold elements the summary conveys) / (total gold elements).
Count an element as conveyed if the summary expresses its meaning, even implicitly and even without clinical terms.
Do NOT give credit for elements the summary mentions that are NOT in the gold, those are handled in Q3c.

Q3c — Added / unsupported elements (precision, the over-generation check). How many ABCD elements does the summary assert that are NOT supported by the gold Evidence for this post? Count each distinct unsupported element once.
0 = none — every element it references is genuinely in the post
1, 2, 3+ = the count of invented / unsupported elements

Q3d — Polarity correctness (conditional on Q3b). Among the elements the summary DID correctly convey (the Q3b numerator), for how many did it get the polarity (adaptive vs maladaptive) right?
Report as a fraction: (correct-polarity among conveyed) / (conveyed elements).
Only judge polarity for elements the summary actually referenced."""


def build_prompt(post: str, gold: str, evidence: str, candidate: str) -> str:
    return f"""{RUBRIC}

Gold Evidence Annotations (the ABCD elements genuinely present in this post):
{evidence}

Gold standard summary:
<gold>
{gold}
</gold>

LLM-generated summary to evaluate:
<LLM-generated>
{candidate}
</LLM-generated>

Post (for context only — check if you need more information about an evidence span):
<post>
{post}
</post>

Using the rubric above, answer all seven questions about the LLM-generated summary.
For q3a, use the scale 0/1/2/NS (NOT 1-5).
For q3b and q3d, report as a fraction string like "3/5" (NOT a decimal).
For q3c, report as an integer count.

Respond with ONLY a JSON object with exactly these keys and value types:
{{
  "q1_overall_quality": <int 1-5>,
  "q2_no_contradiction": <int 1-5>,
  "q3_coverage": <int 1-5>,
  "q3a_dominant_self_state": <int 0-2, or "NS">,
  "q3b_element_recall": <string "X/Y">,
  "q3c_unsupported_elements": <int >= 0>,
  "q3d_polarity_correctness": <string "X/Y">
}}
"""


def _coerce(parsed: dict) -> dict:
    """Coerce each field to the right type to match human data format.
    q3b and q3d are kept as "X/Y" fraction strings (matching human files).
    q3a is 0-2 integer or NaN for NS. q3c is integer. q1/q2/q3 are 1-5 int."""
    out = {}
    for c in JUDGE_COLS:
        v = parsed.get(c)
        try:
            if c == "q3a_dominant_self_state":
                vstr = str(v).strip().upper()
                if vstr in ("NS", "NOT STATED", "N/A", ""):
                    out[c] = float("nan")
                else:
                    out[c] = float(int(round(float(v))))
            elif c == "q3b_element_recall" or c == "q3d_polarity_correctness":
                # Keep as fraction string "X/Y" to match human data
                vstr = str(v).strip()
                if "/" in vstr:
                    out[c] = vstr
                else:
                    # If the model returned a decimal, convert to fraction
                    try:
                        fval = float(vstr)
                        out[c] = vstr  # keep as-is; coercion downstream handles it
                    except ValueError:
                        out[c] = vstr
            elif c == "q3c_unsupported_elements":
                out[c] = float(int(round(float(v))))
            else:
                out[c] = float(int(round(float(v))))
        except (TypeError, ValueError):
            out[c] = float("nan")
    return out


def judge_one(client, model: str, prompt: str, temperature: float,
              num_predict: int = 2000, think: bool = True) -> dict:
    """Call the local LLM and return the coerced judgment dict.

    Models differ in their thinking behavior:
    - gpt-oss:20b NEEDS thinking enabled (think=True) — without it, the response
      is garbled or empty. The thinking trace is internal; the final JSON answer
      appears in message.content after the trace completes. Needs a higher
      num_predict (~2000) to leave room for both thinking and the JSON answer.
    - qwen3.5/3.6 NEED thinking disabled (think=False) — their reasoning trace
      consumes the entire token budget and returns empty content. With think=False
      they return clean JSON in ~100 tokens (num_predict=600 is sufficient).

    The JSON is extracted from content regardless of whether thinking produced
    surrounding text — we search for the last {...} block.
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    # think parameter: ollama supports think=True/False for thinking models.
    # For non-thinking models, think is ignored.
    kwargs["think"] = think
    r = client.chat(**kwargs)
    content = r.message.content or ""
    # Tolerate models that wrap JSON in prose / thinking tags.
    if "```" in content:
        content = content.split("```")[-2] if content.count("```") >= 2 else content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Find the last {...} block in the content (thinking trace may contain
        # intermediate JSON-like text; the answer is last).
        start = content.rfind("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(content[start:end + 1])
        else:
            raise
    return _coerce(parsed)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for CLPsych Task B summaries.")
    parser.add_argument("--system", action="append", default=None,
                        help="'label=path' to an official-format submission JSON (repeatable). "
                             "Defaults to the 4 human-eval systems.")
    parser.add_argument("--post_data", default="post_data.csv")
    parser.add_argument("--model", default="gpt-oss:20b",
                        help="ollama model name (default gpt-oss:20b; distinct from all "
                             "judged systems' base models. Also tested: qwen3.5:latest "
                             "with --no_think).")
    parser.add_argument("--think", action="store_true", default=True,
                        help="Enable thinking mode (default for gpt-oss:20b, which needs "
                             "it to produce coherent JSON). Use --no_think for Qwen3.x "
                             "models whose reasoning trace consumes the token budget.")
    parser.add_argument("--no_think", action="store_true",
                        help="Disable thinking (for Qwen3.x models). Overrides --think.")
    parser.add_argument("--max_posts", type=int, default=0,
                        help="Max posts to judge PER SYSTEM (0 = all annotated posts). Use a "
                             "small N to validate quickly before the full run.")
    parser.add_argument("--post_ids", default=None,
                        help="Comma-separated list of post_ids to restrict judging to "
                             "(e.g. the 30 human-eval posts). If omitted, all annotated "
                             "posts with a gold summary are judged.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output_dir", default="./outputs/llm_judge")
    args = parser.parse_args()

    print(f"Rubric: combined q1-q3 (prompt_judges.txt) + q3a-q3d (ABCD Fidelity docx), "
          f"with full ABCD example lists ({len(RUBRIC)} chars).")

    posts = load_post_data(args.post_data)
    print(f"Loaded {len(posts)} posts with post text, gold summary, and gold evidence.")

    # Restrict to the 30 human-eval posts (same posts the raters scored).
    # Auto-detect from the N_*.xlsx files; can override with --post_ids.
    post_id_filter = None
    if args.post_ids:
        post_id_filter = set(p.strip() for p in args.post_ids.split(","))
        print(f"Restricting to {len(post_id_filter)} specified post_ids.")
    else:
        # Auto-detect from the human eval files
        import glob as _glob
        n_files = (_glob.glob("rHuman_Evals_main/N_*.xlsx") +
                   _glob.glob("rHuman_evals/N_*.csv"))
        if n_files:
            try:
                if n_files[0].endswith((".xlsx", ".xls")):
                    _ref = pd.read_excel(n_files[0])
                else:
                    _ref = pd.read_csv(n_files[0], encoding="utf-8", errors="replace")
                post_id_filter = set(_ref["post_id"].astype(str))
                print(f"Auto-detected {len(post_id_filter)} human-eval posts from "
                      f"{os.path.basename(n_files[0])}.")
            except Exception:
                pass

    specs = []
    if args.system:
        for spec in args.system:
            label, _, path = spec.partition("=")
            specs.append((label.strip(), path.strip()))
    else:
        specs = DEFAULT_SYSTEMS

    # Build the work list: (system, post_id, summary) for posts with gold + evidence.
    work = []
    for label, path in specs:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {label}.")
            continue
        rows = load_summaries(path, label)
        kept = [r for r in rows if r["post_id"] in posts]
        if post_id_filter:
            kept = [r for r in kept if r["post_id"] in post_id_filter]
        if args.max_posts > 0:
            kept = kept[:args.max_posts]
        print(f"  {label}: {len(rows)} non-empty summaries, {len(kept)} to judge "
              f"(after gold/evidence join"
              f"{f' + post_id filter ({len(post_id_filter)} posts)' if post_id_filter else ''}"
              f"{f', max {args.max_posts}' if args.max_posts else ''}).")
        work.extend(kept)
    if not work:
        raise SystemExit("No (system, post) pairs to judge.")
    print(f"\nJudging {len(work)} (system, post) pairs with model {args.model}...")

    import ollama
    client = ollama.Client()

    think = not args.no_think if args.no_think else args.think
    num_predict = 600 if not think else 3000
    print(f"Think mode: {think} (num_predict={num_predict})")

    out_rows = []
    t0 = time.time()
    for i, item in enumerate(work, 1):
        info = posts[item["post_id"]]
        prompt = build_prompt(info["post"], info["gold"], info["evidence"], item["summary"])
        scores = None
        for attempt in range(2):  # retry once on JSON parse failure
            try:
                scores = judge_one(client, args.model, prompt, args.temperature,
                                  num_predict=num_predict, think=think)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"  [{i}/{len(work)}] {item['system']}/{item['post_id']}: "
                          f"retrying (attempt 1 failed: {e})")
                else:
                    print(f"  [{i}/{len(work)}] {item['system']}/{item['post_id']}: "
                          f"FAILED ({e}); NaN row.")
                    scores = None
        if scores is None:
            scores = {c: float("nan") for c in JUDGE_COLS}
        row = {"system": item["system"], "post_id": item["post_id"], **scores}
        out_rows.append(row)
        if i % 10 == 0 or i == len(work):
            dt = time.time() - t0
            print(f"  [{i}/{len(work)}] {item['system']}/{item['post_id']} done "
                  f"({dt:.0f}s elapsed, {dt/i:.1f}s/judgment)")

    out_df = pd.DataFrame(out_rows, columns=["system", "post_id"] + JUDGE_COLS)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "llm_judge_per_summary.csv")
    # Don't round string columns (q3b/q3d are "X/Y" fraction strings)
    numeric_cols = [c for c in JUDGE_COLS if c not in ("q3b_element_recall", "q3d_polarity_correctness")]
    out_df[numeric_cols] = out_df[numeric_cols].apply(pd.to_numeric, errors="coerce").round(6)
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(f"LLM-JUDGE BY SYSTEM  (model={args.model}, n={len(out_df)} judgments)")
    print("=" * 70)
    # For the summary, convert fraction strings to proportions for display
    display_df = out_df.copy()
    for c in ("q3b_element_recall", "q3d_polarity_correctness"):
        display_df[c] = display_df[c].apply(
            lambda v: float(v.split("/")[0]) / float(v.split("/")[1])
            if isinstance(v, str) and "/" in v and float(v.split("/")[1]) != 0
            else float("nan")
        )
    print(display_df.groupby("system")[JUDGE_COLS].mean().round(3).to_string())
    print(f"\nWrote {out_path}")
    print("Feed to judge_agreement.py (with the human N_*.csv) for B5: "
          "python judge_agreement.py --judge <this> --human <N_*.csv ...>")


if __name__ == "__main__":
    main()
