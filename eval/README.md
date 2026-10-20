# eval/

Scoring and analysis: everything that *scores* or *compares* model outputs
(official metrics, our alternative metrics, significance tests, agreement
analyses, ranking correlations). Generation lives in `../pipelines/`.

## Official-format machinery

| Script | Purpose |
|---|---|
| `evaluation.py` | Repo-side scorer for `pipeline.py` outputs: Task A.3 (accuracy/macro-F1), Task B (Consistency/Contradiction/Evidence Alignment via NLI), Task C (CS, CT). |
| `edit_output_for_official_eval.py` | Converts `task_b_results` / Task C JSONs (and BLUE-format CSVs) into official submission JSONs for `clpsych2025/evaluation/run.py`. |
| `analyze_official_results.py` | Distills a `results_dev_*.csv` into readable per-task tables. |

(The official shared-task scorer itself lives in
`../clpsych2025/evaluation/run.py` — run it from inside that directory.)

## CLPsych 2026 Task 3.1-style evaluation

| Script | Purpose |
|---|---|
| `task31_eval.py` | Recreates the CLPsych 2026 Task 3.1 evaluation (CS, CT, ROUGE-L recall, BERTScore recall, 350-word truncation) over our Task B submission JSONs, then ranks systems by the **average of the four per-metric ranks** (plus the official `t3_rank = CS` for reference). Gold comes from `clpsych2025/evaluation/data/dev.json`. Supports gold/constant/empty pseudo-systems and a `--mock` mode for pipeline tests. |

## Our alternative metric & significance tests

| Script | Purpose |
|---|---|
| `specificity_analysis.py` | Specificity/informativeness panel (TTR, MATTR, content-word ratio, mean IDF, hedges, ABCD density, boilerplate) + correlation with consistency. |
| `wilcoxon_label_test.py` | Wilcoxon on the official CS across label sources (older script; has a known mixing bug — see PROJECT_CONTEXT §3.7). |
| `wilcoxon_fidelity_test.py` | Wilcoxon on ABCD-fidelity F1 across label sources / retrieval types (correct). |

## Human/judge comparison machinery

| Script | Purpose |
|---|---|
| `rank_correlation.py` | MSLR-style Spearman analysis: metric rankings vs human rankings (system- and instance-level). |
| `judge_agreement.py` | LLM-judge vs human-consensus agreement per question (ICC variants with CIs + Spearman/Pearson). |
| `icc_human_raters.py` | Inter-annotator ICC across the human raters. |
| `judge_means.py` | Aggregates raw judge output into per-question means. |
| `borda_bootstrap.py` | Borda count + bootstrapping over pairwise preference sheets. |
| `build_pref_sheets.py` | Builds the pairwise preference sheets for human annotation. |
