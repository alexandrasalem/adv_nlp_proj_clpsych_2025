# CLPsych 2025 Critique Paper

Team members: Sophie Brown, Alexandra Salem, Tamoghna Chakraborty, & Daniel Weiner

This repository contains our CLPsych 2025 shared task system code and the
follow-up **critique paper** work: an evaluation of the shared task's official
Consistency Score (CS), degenerate-baseline and specificity analyses of CS, our
structure-aware alternative metric (ABCD-fidelity), LLM-judge and human
agreement analyses, and reimplementations of the shared task's participating
systems.

It was built with Python 3.13. Install requirements with `pip install -r requirements.txt`.

## Repository layout

| Directory | Contents |
|---|---|
| `data_prep/` | One-time conversion of the shared-task source JSONs into `post_data.csv` / `timeline_data.csv`. |
| `pipelines/` | Generation: our pipeline, the reimplemented shared-task systems, baselines, LLM judge. See `pipelines/README.md`. |
| `eval/` | Scoring & analysis: official-format tooling, the CLPsych 2026 Task 3.1-style evaluation, fidelity/Wilcoxon/rank-correlation/agreement scripts. See `eval/README.md`. |
| `figure_generators/` | Paper figures. See `figure_generators/README.md`. |
| `clpsych2025/` | The official shared-task code (untracked): `evaluation/run.py` scorer, submissions, results. |
| `outputs/` | Generated results (untracked). |
| `PROJECT_CONTEXT.md` | Project state, key findings, and rules — **read this first**. |

## Our pipeline

* `data_prep/prepare_data.py` — saves the source data to CSVs.
* `pipelines/pipeline.py` — runs the structural, semantic, hybrid, and concatenation retrieval-augmented summarization pipeline. Generates predictions only; no metrics.
    1. Task A.3 — ABCD classification of gold evidence spans
    2. Structural profile construction (12-dim + wellbeing + ratio)
    3. Cosine similarity retrieval for in-context example selection
    4. Task B — Post-level summarization (zero-shot and one-shot)

```
python pipelines/pipeline.py --data_dir /path/to/json/files --output_dir ./outputs --retrieval_mode hybrid/structural/semantic/hybrid_concat --alpha 0.5
```

Useful flags: `--skip_a3` (use gold ABCD labels), `--cross_validate`, `--model_path` (local `.gguf`), `--n_gpu_layers`.

* `eval/evaluation.py` — scores the predictions written by `pipeline.py`. Implements Task A.3 (accuracy + macro-F1), Task B (Consistency, Contradiction, Evidence Alignment via NLI), and Task C (CS, CT).

```
python eval/evaluation.py --task b --results outputs/task_b_results.json --mode both
```

The first Task B run downloads the NLI model (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, ~1.5 GB).

## Reimplemented CLPsych 2025 systems

`pipelines/` contains reproductions of the shared task's systems, all scored
under the repo convention: **gold evidence spans + gold well-being scores stand
in for the Task A outputs**, and **leave-one-timeline-out** is enforced for
every retrieval or sampling step (no in-context example ever comes from the
target post's own timeline, for Task B and Task C alike). uOttawa is the one
faithful exception on inputs — its prompt uses the raw post text only, exactly
as in the paper.

| Script | System | Method | Original model |
|---|---|---|---|
| `pipelines/eaionflux_model.py` | EAIonFlux (2nd place) | k-shot ICL (paper Appendix A.2/A.3 prompts), post/timeline vector DBs | llama3.3:70b |
| `pipelines/bulusi_model.py` | BULUSI | 5-shot; contriever top-50 re-ranked by ms-marco-MiniLM-L-6-v2; team's actual prompts (their GitHub) | Qwen2.5-72B |
| `pipelines/aquarius_model.py` | Aquarius (Wang et al.) | one-shot; single most BGE-large-en-v1.5-similar post; paper Fig. 3 prompt | Qwen2.5-32B |
| `pipelines/ism_model.py` | ISM | joint template (temp 0.1); 10 candidate summaries/item; Gaussian KDE (Scott's rule) selection over MiniLM embeddings | Meta-Llama-3-8B-Instruct |
| `pipelines/uottawa_model.py` | uOttawa (5th place Task B) | prompt engineering only, variants S1/S2/S3 (paper Table 7 template); no evidence input | Llama-3.3-70B |

Every system script:
* implements **both Task B and Task C** (`--task b`, `--task c`, or `--task bc` for one model load, Task B first);
* supports `--backend vllm` (cluster: local weights, `--tensor_parallel_size`, optional `--quantization`), `--backend ollama` (local single-GPU), and `--backend mock` (functional tests — stand-in embedders too, so nothing downloads);
* is **resumable** — per-post results append to a JSONL checkpoint and are skipped on restart;
* writes an **official submission-format JSON** (the same structure as
  `dev.json` / the gold data, with gold evidence and well-being filled in) that
  plugs straight into `eval/task31_eval.py` and `clpsych2025/evaluation/run.py`;
* writes a `*_settings.md` **decoding log** recording every generation/retrieval
  setting and whether it came from the paper or is our documented default (the
  source for the paper's reimplementation section), including any prompts
  shortened by the automatic context-window guard.

### Cluster run (all systems, original paper models)

`pipelines/run_all_systems.sh` runs BULUSI (72B), Aquarius (32B), ISM (8B),
uOttawa S2 (70B), and the EAIonFlux rerun (70B) — **Task B only** — sequentially
through vLLM with tensor parallelism across 4 GPUs, resumably and fully locally
(local weights, `HF_HUB_OFFLINE=1`). Task C blocks for every system are
included but commented out for later use (uOttawa's Task C cascades its own
Task B summaries). Configure `DATA_DIR` and the `*_PATH`
model directories, then:

```bash
bash pipelines/run_all_systems.sh          # rerun to resume after any crash/preemption
```

Copy the `outputs/cluster_run/<system>/*_submission.json` files back and score
them locally (see next section). The commented blocks in the driver also run
uOttawa's S1/S3 variants.

## CLPsych 2026 Task 3.1-style evaluation (on our Task B results)

`eval/task31_eval.py` recreates the CLPsych 2026 shared task Task 3.1
evaluation — **CS** (1 − mean contradiction), **CT** (mean per-sentence max
contradiction), **ROUGE-L Recall** (rouge-score, Porter stemmer), and
**BERTScore Recall** (per-gold-sentence max F1, `microsoft/deberta-xlarge-mnli`
with baseline rescaling), all after truncating predictions to 350 words — but
adapted to our Task B submission JSONs and gold summaries from
`clpsych2025/evaluation/data/dev.json`. Systems are ranked on each metric
(ties → average rank) and the four **ranks** are averaged into the final
ranking; the official 2026 key (`t3_rank = t3_cs`) is reported alongside for
reference. Gold / constant / empty pseudo-systems are included by default
(flags `--no_gold` / `--no_constant` / `--no_empty` to drop them).

```
# every submission in submissions_dev/ (plus gold/constant/empty anchors)
python eval/task31_eval.py

# the cluster system outputs
python eval/task31_eval.py --submissions_dir outputs/cluster_run --pattern "*_submission"
```

Outputs: per-post and per-system CSVs under `outputs/task31_eval/`.
`--mock` runs the whole pipeline without loading any model; `--limit N`
scores only the first N gold posts; `--no_bertscore` skips the large
BERTScore model.

## Rank-correlation analysis (metrics vs human)

Following the MSLR precedent, `eval/rank_correlation.py` ranks systems by each
automated metric (averaging per-instance values across instances) and computes
the Spearman ρ between every metric ranking and a human ranking, plus the full
metric-vs-metric matrix. Multiple annotators are averaged to the per-item
consensus first. Supply human rankings in the schema of
`human_rankings_template.csv` (or per-summary `human_rankings_instance_template.csv`).

```
python eval/rank_correlation.py --level conditions --task B \
    --results clpsych2025/evaluation/results/results_dev_<ts>.csv \
    --human my_human_rankings.csv --heatmap

python eval/rank_correlation.py --level instances --task B --instance_key post_id \
    --results clpsych2025/evaluation/results/results_dev_<ts>.csv \
    --human my_per_summary_human.csv
```

`--restrict_to_human` inner-joins metrics to the human-rated summaries so
metric means and human ratings cover exactly the same posts, and additionally
prints the pooled instance-level ρ. Metrics containing `contradiction` or `mse`
are treated as lower-is-better. **System-id matching:** the `system` values in
the human file must equal `team_name_submission_id` as derived from submission
filenames (`run.py`'s `parse_filename`); unmatched ids are printed.

## Specificity / informativeness analysis

`eval/specificity_analysis.py` computes a panel of specificity signals over
generated summaries (TTR, MATTR, content-word ratio, **mean corpus IDF**, hedge
density, ABCD-term density, bigram boilerplate), aggregates per system, and
optionally correlates each signal with consistency (`--cs_by_system`) — a
negative ρ for "higher = more specific" metrics is direct evidence that the
metric rewards vaguer summaries.

```
python eval/specificity_analysis.py \
    --source "eaionflux_k5=outputs/task_b_eaionflux_k5_<ts>.json#one_shot_summary" \
    --source "oneshot=outputs/task_b_results_gold_hybrid-a0.50_<ts>.json#one_shot_summary" \
    --include_constant \
    --cs_by_system cs_by_system.csv
```

## LLM-judge vs human agreement (ICC)

`eval/judge_agreement.py` treats the human consensus and the LLM judge as two
raters of the same summaries and reports, per question, ICC(2,1)/ICC(3,1) with
95% CIs, ICC(2,k)/ICC(3,k), and Spearman/Pearson — validated against the
Shrout & Fleiss (1979) benchmark.

```
python eval/judge_agreement.py --human human_per_summary.csv --judge judge_per_summary.csv
```

`eval/icc_human_raters.py` does the same across the human raters themselves;
`eval/borda_bootstrap.py` + `eval/build_pref_sheets.py` handle the pairwise
preference sheets (Borda count + bootstrapping).

## Installation

```
pip install -r requirements.txt
```

LLM inference runs locally (ollama / llama.cpp / vLLM on the cluster); NLI and
BERTScore models download from HuggingFace on first use.

### GPU (CUDA) install for `llama-cpp-python`

The default `pip install llama-cpp-python` installs a CPU-only wheel. To run
the LLM on an NVIDIA GPU, install the prebuilt CUDA wheel:

```powershell
pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir
```

Use `--index-url`, **not** `--extra-index-url** (pip silently falls back to the
CPU wheel otherwise). Match `cuXXX` to your CUDA version (wheels are
forward-compatible).

Verify the wheel has CUDA support:

```
python -c "from llama_cpp import llama_cpp as L; print([s for s in dir(L) if 'cuda' in s.lower()])"
```

A CUDA build prints many `ggml_backend_cuda_*` symbols; a CPU-only build prints
only `['llama_supports_gpu_offload']`.

> **IMPORTANT:** Run everything locally. Do not use cloud APIs or cloud-based
> AI assistants with this data. The data sharing agreement prohibits sending
> data to third-party LLM providers. This applies to the cluster runs as well
> (local weights only, `HF_HUB_OFFLINE=1`).
