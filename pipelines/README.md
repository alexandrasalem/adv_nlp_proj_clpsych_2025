# pipelines/

Generation pipelines: everything that *produces* model outputs (Task A.3
classification, Task B/C summaries, baselines, judges) as opposed to scoring
or analyzing them. Run scripts from the repo root, e.g.
`python pipelines/pipeline.py ...`.

## Our system

| Script | Purpose |
|---|---|
| `pipeline.py` | Main pipeline (Task A.3 ABCD classification → structural profile → retrieval-augmented one-shot/zero-shot Task B summarization; sweep mode). |
| `ABCD_classifier.py` | Standalone Task A.3 span classifier. |
| `run_zeroshot_label_sources.py` | Zero-shot-only Task B generation per label source (gold/predicted/random). |
| `run_random_retrieval.py` | Random-retrieval condition per label source (seed-42 retrieval, labels vary). |
| `running_task_b_baselines.py` / `running_task_c_baselines.py` | Task B / Task C baseline generation runs (ollama). |
| `make_degenerate_baselines.py` | Degenerate-baseline (constant/empty/gold-summary) submissions in official format. |
| `make_degenerate_task_b_results.py` | Same baselines in the repo's `task_b_results` format. |
| `task_c_eval_sweep.ps1` | PowerShell driver that scores Task C submissions across the official eval sweep. |

## Reimplemented CLPsych 2025 shared-task systems

All reimplementations follow the repo convention: gold evidence spans + gold
well-being scores stand in for the Task A outputs, and leave-one-timeline-out
is enforced for every retrieval/sampling step. Each writes a resumable JSONL
checkpoint, an official submission-format JSON, and a `*_settings.md` decoding
log (the source for the paper's reimplementation section).

| Script | System | Original model |
|---|---|---|
| `eaionflux_model.py` | EAIonFlux (2nd place) ICL reproduction, Tasks B + C | llama3.3:70b (paper original; ollama or vLLM backend) |
| `bulusi_model.py` | BULUSI — 5-shot, contriever top-50 → ms-marco-MiniLM-L-6-v2 rerank, team's actual prompt | Qwen2.5-72B |
| `aquarius_model.py` | Aquarius — one-shot, BGE-large-en-v1.5 argmax retrieval, paper Fig. 3 prompt | Qwen2.5-32B |
| `ism_model.py` | ISM — joint template, temp 0.1, 10 candidates/post, Gaussian-KDE (Scott) selection over MiniLM summary embeddings | Meta-Llama-3-8B |
| `uottawa_model.py` | uOttawa — prompt-only on raw post text, variants S1 (zero-shot) / S2 (one-shot) / S3 (few-shot 7) | Llama-3.3-70B |

`run_all_systems.sh` is the cluster driver: it runs BULUSI, Aquarius, ISM,
uOttawa (S2) and the EAIonFlux B+C rerun sequentially through vLLM with tensor
parallelism across 4 GPUs, on local weights, resumably (rerunning it only
generates what is missing). `system_utils.py` holds the shared machinery
(data loading, LOTO, vLLM/ollama/mock backends, checkpoints, submission
writer, settings logger).

## Metrics & judges that generate model scores

| Script | Purpose |
|---|---|
| `abcd_fidelity.py` | ABCD-fidelity metric (NLI recall/precision/F1 vs the gold ABCD profile). |
| `llm_judge.py` | LLM-as-a-judge (gpt-oss:20b) with the q1–q3d instrument from `prompt_judges.txt`. |

## Cluster notes

On the cluster: `bash pipelines/run_all_systems.sh` with `DATA_DIR`,
`*_PATH` (local model-weight directories) and optionally `TP` / `QUANTIZATION`
set. Everything is local-only (vLLM loads from disk; `HF_HUB_OFFLINE=1`).
