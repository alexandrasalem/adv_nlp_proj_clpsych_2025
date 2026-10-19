import ollama
import json
import os
import argparse

import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse eaionflux_model.py's data layer verbatim so baseline loading/parsing
# matches the rest of the pipeline: timeline JSON files -> Timeline/Post
# dataclasses via load_timelines(). Do not re-implement that parsing here.
from pipelines.eaionflux_model import load_timelines, Post, Timeline

parser = argparse.ArgumentParser(description="Arguments for setting up the baseline runs.")
parser.add_argument("--data_dir", required=True,
                    help="Directory of timeline JSON files (or a single .json), "
                         "same format expected by eaionflux_model.py.")
parser.add_argument("--output_dir", default="./outputs",
                    help="Where to write the baseline CSV outputs.")

args = parser.parse_args()

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 300  # per paper Appendix C.3 — do not raise


def _post_row(post: Post, summary: str, truncated: bool | None = None) -> dict:
    """Flatten a Post into the post_data.csv row schema that evaluation.py reads."""
    row = {
        "timeline_id": post.timeline_id,
        "post_index": post.post_index,
        "post_id": post.post_id,
        "post_date": post.date,
        "post": post.text,
        # evaluation.py filters on post_summary != "NONE", so keep that sentinel.
        "post_summary": post.summary or "NONE",
        "well_being_score": "" if post.wellbeing is None else post.wellbeing,
        "adaptive_evidence": " | ".join(t for _, _, t in post.evidence["adaptive"]),
        "maladaptive_evidence": " | ".join(t for _, _, t in post.evidence["maladaptive"]),
    }
    if summary is not None:
        row["task_b_baseline_blue_team_summaries"] = summary
    if truncated is not None:
        row["task_b_baseline_truncated"] = truncated
    return row


def blue_baseline_one_post(post_text):
      prompt = f"""
      Given the following Reddit post, summarize the interplay between adaptive and maladaptive self-states.

      Post:
      \"{post_text}\"

      Response format:
      {{ "summary": "<post-level summary>" }}
      """

      res = ollama.generate(model='llama3.2:3b', prompt=prompt, format="json", stream=False)
      res_dict = json.loads(res.response)
      summary = res_dict["summary"]
      return summary

def run_task_b_blue_baseline(timelines: list[Timeline], output_dir: str):
    """Run the blue-team baseline over every post in the loaded timelines; write CSV."""

    all_posts = [p for tl in timelines for p in tl.posts]
    rows = []
    for post in tqdm.tqdm(all_posts):
        summary = blue_baseline_one_post(post.text)
        rows.append(_post_row(post, summary=summary, truncated=None))

    os.makedirs(output_dir, exist_ok=True)
    data = pd.DataFrame(rows)
    new_csv_name = "post_data_task_b_blue_baseline.csv"
    out_path = os.path.join(output_dir, new_csv_name)
    data.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return data

def single_official_baseline_post_summary(post_text, model, tokenizer, device):
    prompt = f"""
Analyze the following social media post and identify the dominant self-state (adaptive or maladaptive).
Begin by determining which self-state is more dominant and describe it first.
For each self-state, highlight the central organizing aspect - A (Affect), B (Behavior), C (Cognition), or D (Desire / Need) that drives the state.
Describe how this central aspect influences the other aspects, focusing on the potential causal relationships between them.
If the self-state is maladaptive, explain how negative emotions, behaviors, or thoughts hinder psychological needs, and if adaptive, explain how positive aspects support psychological needs.
If both adaptive and maladaptive states are present, describe each in turn.
If only one self-state is evident, focus solely on that.
You must not make anything up.
Keep the description concise and only describe observations if they are fully supported by the text.
Post Content : { post_text }
Summary :
"""
    messages = [{"role": "user", "content": prompt}]
    # apply_chat_template may return either a raw tensor (older transformers)
    # or a BatchEncoding (newer versions / return_dict=True). Normalize to a
    # plain input_ids tensor so torch.ones_like works below.
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return either a raw tensor (older transformers)
    # or a BatchEncoding (newer versions). BatchEncoding subclasses UserDict
    # (not dict), so check for the input_ids attribute instead of isinstance.
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"]
    elif isinstance(encoded, (dict,)) or hasattr(encoded, "keys"):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    input_ids = input_ids.to(device)

    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt by token count rather than string-splitting on
    # "Summary :" — exact, and won't IndexError if the marker is absent.
    generated_ids = outputs[0][prompt_len:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Truncated if we consumed the full budget without emitting EOS.
    hit_cap = len(generated_ids) >= MAX_NEW_TOKENS
    emitted_eos = tokenizer.eos_token_id in generated_ids.tolist()
    was_truncated = hit_cap and not emitted_eos

    return summary, was_truncated


def run_task_b_official_baseline(timelines: list[Timeline], output_dir: str):
    """Run the official baseline over every post in the loaded timelines; write CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 — fp32 would be ~32GB and will not fit on a 16GB card.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    all_posts = [p for tl in timelines for p in tl.posts]
    rows, truncated_flags = [], []

    for post in tqdm.tqdm(all_posts, total=len(all_posts)):
        summary, was_truncated = single_official_baseline_post_summary(
            post.text, model, tokenizer, device
        )
        rows.append(_post_row(post, summary=None, truncated=was_truncated))
        # Rename the summary into the official column expected by evaluation.py.
        rows[-1]["task_b_baseline_official_summaries"] = summary
        truncated_flags.append(was_truncated)

    os.makedirs(output_dir, exist_ok=True)
    data = pd.DataFrame(rows)
    n_trunc = sum(truncated_flags)
    print(f"\nTruncated at the {MAX_NEW_TOKENS}-token cap: "
          f"{n_trunc}/{len(truncated_flags)} ({100*n_trunc/len(truncated_flags):.1f}%)")

    new_csv_name = "post_data_task_b_official_baseline.csv"
    out_path = os.path.join(output_dir, new_csv_name)
    data.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return data


if __name__ == "__main__":
    timelines = load_timelines(args.data_dir)
    run_task_b_official_baseline(timelines, args.output_dir)
    # run_task_b_blue_baseline(timelines, args.output_dir)
