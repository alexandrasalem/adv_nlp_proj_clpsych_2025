# figure_generators/

Paper figure generation for the critique paper. All scripts read the metric /
human-rating CSVs under `outputs/` and `outputs/final_analysis/` (never the
raw post data).

| Script | Figures |
|---|---|
| `figure_generator.py` | All paper figures in one pass (then fine-tune individually). |
| `make_paper_heatmaps.py` | The 3 paper-quality heatmaps: system-level 13×13 (metrics × human), instance-level metric × human questions, plus the third from the paper's set. |
| `generate_task_c.py` | Task C generation for EAIonFlux + the 6 new-prompt sweep systems via the shared Qwen2.5-14B (ollama) — grouped here because its output feeds the Task C figures. |
