# Advanced NLP Final Project: CLPsych Shared Task 2025

Team members: Sophie Brown, Alexandra Salem, Tamoghna Chakraborty, & Daniel Weiner

This repository contains our team's CLPsych 2025 shared task project code.

It was built with Python 3.13. Install requirements with `pip install -r requirements.txt`.

It contains the following scripts:
* Data is saved to csvs using `prepare_data.py`.
* `pipeline.py` runs the structural retrieval-augmented summarization pipeline:
    1. Task A.3 — ABCD classification of gold evidence spans
    2. Structural profile construction (12-dim + wellbeing + ratio)
    3. Cosine similarity retrieval for in-context example selection
    4. Task B — Post-level summarization (zero-shot and one-shot)

## Installation

Base requirements:

```
pip install llama-cpp-python numpy scikit-learn huggingface-hub --break-system-packages
```

### GPU (CUDA) install for `llama-cpp-python`

The default `pip install llama-cpp-python` installs a CPU-only wheel — passing `n_gpu_layers=-1` will silently do nothing. To run the LLM on an NVIDIA GPU, install the prebuilt CUDA wheel from abetlen's index:

```powershell
pip uninstall llama-cpp-python -y
pip cache remove llama_cpp_python
pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir
```

Use `--index-url`, **not** `--extra-index-url` — with `--extra-index-url`, pip will silently fall back to the PyPI CPU wheel if it can't find a matching version on the CUDA index. Match `cuXXX` to your CUDA version from `nvidia-smi` (wheels are forward-compatible, so `cu124` works fine on a CUDA 13.x driver).

Verify the installed wheel has CUDA support:

```
python -c "from llama_cpp import llama_cpp as L; print([s for s in dir(L) if 'cuda' in s.lower()])"
```

A CUDA build prints many `ggml_backend_cuda_*` symbols. A CPU-only build prints only `['llama_supports_gpu_offload']`.

## Usage

```
python pipeline.py --data_dir /path/to/json/files --output_dir ./outputs
```

> **IMPORTANT:** Run locally. Do not use cloud APIs or cloud-based AI assistants with this script or the data. The data sharing agreement prohibits sending data to third-party LLM providers.

