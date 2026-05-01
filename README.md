# Advanced NLP Final Project: CLPsych Shared Task 2025

Team members: Sophie Brown, Alexandra Salem, Tamoghna Chakraborty, & Daniel Weiner

This repository contains our team's CLPsych 2025 shared task project code.

It was built with Python 3.13. Install requirements with `pip install -r requirements.txt`.

It contains the following scripts:
* `prepare_data.py` — saves the source data to CSVs.
* `pipeline.py` — runs the structural retrieval-augmented summarization pipeline. Generates predictions only; no metrics.
    1. Task A.3 — ABCD classification of gold evidence spans
    2. Structural profile construction (12-dim + wellbeing + ratio)
    3. Cosine similarity retrieval for in-context example selection
    4. Task B — Post-level summarization (zero-shot and one-shot)
* `evaluation.py` — scores the predictions written by `pipeline.py`. Implements Task A.3 (accuracy + macro-F1), Task B (Consistency, Contradiction, Evidence Alignment via NLI), and Task C (CS, CT). Tasks A.1 and A.2 are not evaluated — the pipeline uses gold evidence spans and gold well-being scores as inputs.

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
python pipeline.py --data_dir /path/to/json/files --output_dir ./outputs
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

> **IMPORTANT:** Run locally. Do not use cloud APIs or cloud-based AI assistants with this script or the data. The data sharing agreement prohibits sending data to third-party LLM providers.

