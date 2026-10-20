# data_prep/

One-time data preparation: converting the raw shared-task distribution into
the CSVs the rest of the repo works from.

| Script | Purpose |
|---|---|
| `prepare_data.py` | Reads the per-timeline source JSONs (`--data_files_location`, default `train-clpsych2025-v2/`) and writes `timeline_data.csv` (timeline ids + gold timeline summaries) and `post_data.csv` (post ids, texts, gold summaries, well-being scores, evidence). Note: `post_data.csv` contains the actual post/summary content and is off-limits to agents (data-sharing agreement). |
| `explore_data.py` | Early exploratory look at the source JSONs (structure/sanity checks). |

`post_data.csv` at the repo root is the output of `prepare_data.py` and also
serves as the timeline/post index used by `eval/edit_output_for_official_eval.py`.
