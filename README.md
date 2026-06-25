# Advanced NLP Final Project: CLPsych Shared Task 2025

Team members: Sophie Brown, Alexandra Salem, Tamoghna Chakraborty, & Daniel Weiner

This repository contains our team's CLPsych 2025 shared task project code.

It was built with Python 3.13. Install requirements with `pip install -r requirements.txt`.

It contains the following scripts:
* `prepare_data.py` — saves the source data to CSVs.
* `pipeline.py` — runs the structural, semantic, hybrid, and concatenation retrieval-augmented summarization pipeline. Generates predictions only; no metrics.
    1. Task A.3 — ABCD classification of gold evidence spans
    2. Structural profile construction (12-dim + wellbeing + ratio)
    3. Cosine similarity retrieval for in-context example selection
    4. Task B — Post-level summarization (zero-shot and one-shot)
* `evaluation.py` — scores the predictions written by `pipeline.py`. Implements Task A.3 (accuracy + macro-F1), Task B (Consistency, Contradiction, Evidence Alignment via NLI), and Task C (CS, CT). Tasks A.1 and A.2 are not evaluated — the pipeline uses gold evidence spans and gold well-being scores as inputs.
* `eaionflux_model.py` — reproduction of the 2nd-place CLPsych 2025 team **EAIonFlux** ([Antony & Schoene, 2025](https://aclanthology.org/2025.clpsych-1.23.pdf)) for Tasks B and C: in-context learning with dual post-level / timeline-level vector databases, top-k retrieved demonstrations (k=5 / k=10), and their Appendix A.2 / A.3 prompts. Generation runs locally via Ollama; the embedding and generation models are configurable (paper defaults selectable on the CLI).
* `rank_correlation.py` — MSLR-style ([Wang et al., 2023](https://arxiv.org/pdf/2305.13693)) Spearman rank-correlation analysis between automated-metric rankings and human rankings. Correlates system-level rankings from each NLI metric against a human ranking you supply (`human_rankings_template.csv`), and adds an instance-level mode that correlates per-summary metric values with per-summary human ratings over all rated summaries (`human_rankings_instance_template.csv`).
* `specificity_analysis.py` — measures lexical specificity / informativeness of generated summaries (TTR, MATTR, content-word ratio, mean corpus IDF, hedge density, ABCD-term density, bigram boilerplate) and optionally correlates each signal with consistency, quantifying whether high-consistency summaries are systematically more generic. Pure computation on model outputs — no annotators.
* `judge_agreement.py` — quantifies LLM-as-a-judge vs human agreement per evaluation question with ICC (two-way, absolute-agreement and consistency, with 95% CIs — validated against the Shrout & Fleiss 1979 benchmark) plus Spearman and Pearson. Takes two per-summary rating files (human, judge).

## Installation

Pipeline (LLM inference) requirements:

```
pip install llama-cpp-python numpy scikit-learn huggingface-hub --break-system-packages
```

Evaluation requirements (NLI metrics for Task B / C):

```
pip install torch transformers nltk scikit-learn --break-system-packages
```

### GPU (CUDA) install for `llama-cpp-python`

The default `pip install llama-cpp-python` installs a CPU-only wheel — passing `n_gpu_layers=-1` will silently do nothing. To run the LLM on an NVIDIA GPU, install the prebuilt CUDA wheel from abetlen's index:

```powershell
pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir
```

Use `--index-url`, **not** `--extra-index-url` — with `--extra-index-url`, pip will silently fall back to the PyPI CPU wheel if it can't find a matching version on the CUDA index. Match `cuXXX` to your CUDA version from `nvidia-smi` (wheels are forward-compatible, so `cu124` works fine on a CUDA 13.x driver).

Verify the installed wheel has CUDA support:

```
python -c "from llama_cpp import llama_cpp as L; print([s for s in dir(L) if 'cuda' in s.lower()])"
```

A CUDA build prints many `ggml_backend_cuda_*` symbols. A CPU-only build prints only `['llama_supports_gpu_offload']`.

## Usage

The workflow is two-step: `pipeline.py` produces predictions, `evaluation.py` scores them.

### Step 1 — run the pipeline

```
python pipeline.py --data_dir /path/to/json/files --output_dir ./outputs --retrieval_mode hybrid/structural/semantic/hybrid_concat --alpha 0.5
```

This writes:
- `outputs/task_a3_predictions{timestamp}.json` — predicted vs. gold ABCD label for every evidence span.
- `outputs/task_b_results{timestamp}.json` — zero-shot and one-shot summaries for every annotated post, plus the retrieved example and structural similarity score.

Useful flags:
- `--skip_a3` — skip ABCD classification and use gold ABCD labels for retrieval instead.
- `--cross_validate` — run leave-one-timeline-out cross-validation; writes `task_b_cv_results.json`.
- `--model_path PATH` — point at a local `.gguf` model instead of downloading from HuggingFace.
- `--n_gpu_layers N` — layers to offload to GPU (`-1` = all, the default; needs the CUDA wheel above).

### Step 2 — score the predictions

Task A.3 (ABCD classification accuracy + macro-F1):

```
python evaluation.py --task a3 --results outputs/task_a3_predictions.json
```

Task B (Consistency, Contradiction, Evidence Alignment) — compare both prompting modes:

```
python evaluation.py --task b --results outputs/task_b_results.json --mode both
```

Or score a single mode:

```
python evaluation.py --task b --results outputs/task_b_results.json --mode zero_shot
python evaluation.py --task b --results outputs/task_b_results.json --mode one_shot
```

Each evaluation run writes its scores to the same directory as the predictions file, named `task_a3_evaluation_<timestamp>.json` or `task_b_evaluation_<timestamp>.json` — the timestamp is parsed from the predictions filename so evaluations match up with the run that produced them. Force CPU evaluation (instead of auto-detect) with `--device cpu`.

The first Task B run will download the NLI model (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, ~1.5 GB) and cache it under `~/.cache/huggingface/`.

### EAIonFlux reproduction (Task B & C)

Reproduces the 2nd-place EAIonFlux system: retrieval-augmented in-context learning
with a post-level and a timeline-level vector database, top-k demonstrations, and
the paper's Task B / Task C prompt templates. Generation is served locally by
[Ollama](https://ollama.com/); like the rest of this repo it uses gold evidence
spans and gold well-being scores as the "Task A outputs" feeding Tasks B and C.

Install Ollama and pull a model first, e.g. `ollama pull llama3.1:8b`. Then:

```
python eaionflux_model.py --data_dir /path/to/json/files --task both --k 5
```

Useful flags:
- `--k N` — number of in-context examples to retrieve (the paper ran `5` and `10`).
- `--model NAME` — Ollama generation model. Default `llama3.1:8b` (runs on a 16 GB GPU); pass `llama3.3:70b` to match the paper.
- `--embedding_model ID` — any `sentence-transformers` id. Default `sentence-transformers/all-mpnet-base-v2`; pass `Linq-AI-Research/Linq-Embed-Mistral` to match the paper.
- `--task {B,C,both}` — with `both`, generated Task B summaries are cascaded into the Task C prompt (as in the paper). For `C` alone, post summaries are omitted unless `--use_gold_post_summaries` is set.

Outputs use the project's existing `task_b_results` / `task_c` schema (see `EXAMPLE_DATA.json`),
so they are drop-in for the existing tooling. Task B (`task_b_eaionflux_k<K>_<ts>.json`) writes
`one_shot_summary` (the k-shot ICL summary) and `zero_shot_summary` (a no-retrieval ablation, or
null with `--skip_zero_shot`); convert with `--task B --model_name one_shot_summary`. Task C
(`task_c_eaionflux_k<K>_<input_type>_<ts>.json`) writes the `timeline_id, n_posts, input_type,
prompt_style, gold_summary, predicted_summary, generation_time_sec` records; convert with
`--task C`. After converting into `submissions_dev/`, score them with the official `run.py`
exactly like the other systems.

### Rank-correlation analysis (metrics vs human)

Following the MSLR precedent, `rank_correlation.py` ranks systems by each automated
metric (averaging per-instance values across instances) and computes the Spearman ρ
between every metric ranking and a human ranking, plus the full metric-vs-metric
matrix. It writes a ranks table (MSLR Table 1 analogue), a ρ matrix (Figure 1
analogue), a p-value matrix, and an optional heatmap.

Human rankings are not part of this repo — supply them in a file
(`human_rankings_template.csv` shows the schema: a `system` id column matching the
ranked systems, plus one or more human judgment columns). Use `--human_kind rank`
if those columns are ranks (1 = best) rather than scores.

```
# Rank the repo's own conditions (incl. EAIonFlux + baselines) from local eval output
python rank_correlation.py --level conditions --task B \
    --results clpsych2025/evaluation/results/results_dev_<ts>.csv \
    --human my_human_rankings.csv --heatmap

# Rank the official shared-task teams from a published system-level metrics table
python rank_correlation.py --level teams \
    --metrics_csv official_team_metrics.csv --human official_team_human.csv

# Instance-level (powered): per-summary CS vs human quality across ALL rated summaries
python rank_correlation.py --level instances --task B --instance_key post_id \
    --results clpsych2025/evaluation/results/results_dev_<ts>.csv \
    --human my_per_summary_human.csv
```

`--level instances` is the powered companion to the system-level result (which, with
only a handful of systems, is descriptive at best). It joins each summary's
per-instance metric value to that same summary's human rating and reports one
Spearman ρ — with a real p-value over many summaries — per (metric, human judgment).
It needs the long results CSV plus a **per-summary** human file (`system`, a
per-summary id column, and judgment columns); see `human_rankings_instance_template.csv`.
Use `--instance_key timeline_id` for Task C.

Metrics whose name contains `contradiction` or `mse` are treated as lower-is-better
(rank 1 = lowest); everything else as higher-is-better. Override with
`--lower_is_better` / `--higher_is_better`.

**System-id matching (important).** The `system` values in the human file must
exactly equal the ids the script derives from the eval output. For
`--level conditions`/`instances` that is `team_name_submission_id` by default
(`--system_key`), where `team_name`/`submission_id` come from the submission
filename (`run.py`'s `parse_filename`: `BLUE.json` → `BLUE_BLUE`;
`random_hybrid-a0.50.json` → `random_hybrid-a0.50`). If a join misses, the script
prints the unmatched ids and the ranks table shows the exact ids it expects — align
the human file's `system` column to those.

### Specificity / informativeness analysis

Turns the qualitative "high-consistency summaries are vague" observation into a
measured property. For each generated summary it computes a panel of specificity
signals over the pooled summaries (type-token ratio, moving-average TTR,
content-word ratio, **mean corpus IDF** — the cleanest specificity signal —
hedge density, ABCD-term density, and bigram boilerplate), aggregates per system,
and optionally correlates each signal with consistency.

Point it at any generated-summary files (`name=path#field`, repeatable; omit
`#field` to auto-detect). Use `--include_constant` to anchor the generic end with
the `baseline_constant` sentence, and `--cs_by_system` to get the headline
correlation:

```
python specificity_analysis.py \
    --source "eaionflux_k5=outputs/task_b_eaionflux_k5_<ts>.json#one_shot_summary" \
    --source "oneshot=outputs/task_b_results_gold_hybrid-a0.50_<ts>.json#one_shot_summary" \
    --include_constant \
    --cs_by_system cs_by_system.csv
```

A negative Spearman ρ between a "higher = more specific" metric (e.g. `mean_idf`)
and consistency is direct evidence that the metric rewards vaguer summaries. The
`specificity_by_system.csv` it writes can be fed straight into `rank_correlation.py`
(`--level teams --metrics_csv`) alongside consistency to put specificity and the
NLI metrics in one Spearman matrix.

### LLM-judge vs human agreement (ICC)

Quantifies how well the LLM-as-a-judge tracks human ratings, the same rigorous way
inter-rater reliability is reported. It treats the human rating and the judge rating
as two raters of the same summaries and reports, per question, ICC(2,1) (absolute
agreement) and ICC(3,1) (consistency) with 95% CIs, ICC(2,k)/ICC(3,k), and
Spearman/Pearson. The ICC point estimates and CIs are validated against the
Shrout & Fleiss (1979) benchmark.

Supply two per-summary rating files (one human, one judge), each with join id
column(s) and one numeric column per question. Questions are matched by identical
column names, or mapped with `--pairs`. Multiple rows per item (e.g. several human
raters) are averaged to the per-item consensus first.

```
# Both files share question column names (human_quality, human_consistency, ...)
python judge_agreement.py --human human_per_summary.csv --judge judge_per_summary.csv

# Judge file uses different names
python judge_agreement.py --human human_per_summary.csv --judge judge_per_summary.csv \
    --id_cols system,post_id --pairs human_quality=quality,human_consistency=coverage
```

This is also how to get the judge into `rank_correlation.py`: aggregate the judge's
per-summary scores to system means and add them as extra columns to the human file
(or a `--metrics_csv`), and the judge-vs-human ρ appears in the same Spearman matrix.

> **IMPORTANT:** Run locally. Do not use cloud APIs or cloud-based AI assistants with this script or the data. The data sharing agreement prohibits sending data to third-party LLM providers.

