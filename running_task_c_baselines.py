import ollama
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import argparse

parser = argparse.ArgumentParser(description="Arguments for setting up the baseline runs.")
parser.add_argument("--post_csv_location", default = "post_data.csv", help="full path to the data folder")

args = parser.parse_args()

post_csv_location = args.post_csv_location


def blue_baseline_timeline(timeline_text):
      prompt = f"""
      Given the following series of Reddit posts from one user, generate a timeline-level summary.
      Begin by determining which self-state is dominant (adaptive/maladaptive) and describe it first and focus on the interplay between adaptive and maladaptive self-states over time.

      Timeline:
      \"{timeline_text}\"

      Response format:
      {{ "summary": "<timeline-level summary>" }}
      """

      res = ollama.generate(model='llama3.2:3b', prompt=prompt, format="json", stream=False)
      res_dict = json.loads(res.response)
      summary = res_dict["summary"]
      return summary

def run_task_c_blue_baseline(post_csv_location):
    data = pd.read_csv(post_csv_location)
    timelines = {}
    for index, row in data.iterrows():
        if row['timeline_id'] not in timelines.keys():
            timelines[row['timeline_id']] = [row['post']]
        else:
            timelines[row['timeline_id']].append(row['post'])

    summaries = []
    for timeline in tqdm.tqdm(timelines.items()):
        joined_timeline = "\n\n".join(timeline)
        summary = blue_baseline_timeline(joined_timeline)
        summaries.append(summary)

    data['task_c_baseline_blue_team_summaries'] = summaries
    new_csv_name = f'{post_csv_location[:-4]}_task_c_blue_baseline.csv'
    data.to_csv(new_csv_name, index=False)

def single_official_baseline_timeline_summary(all_posts_concatenated, model, tokenizer, device):
    prompt = f"""
Generate a timeline-based summary analyzing the evolution of self-states across all posts in chronological order. Emphasize the interplay between adaptive and maladaptive self-states, focusing on temporal dynamics such as flexibility, rigidity, improvement, and deterioration. Describe how the dominance of self-states shifts over time, highlighting key emotional, cognitive, and behavioral changes that contribute to these transitions. You must not make anything up. Keep the description concise and only describe observations if they are fully supported by the text.
All Posts Content : { all_posts_concatenated }
Timeline Summary :
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(device)


    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature = 0.1,
        top_p = 0.9,
        max_new_tokens=300,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_full = decoded_full.split("Timeline Summary :\n")[1]
    return decoded_full

def run_task_c_official_baseline(post_csv_location):
    # prepping model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8b-Instruct",
    )

    model.to(device)

    data = pd.read_csv(post_csv_location)
    timelines = {}
    for index, row in data.iterrows():
        if row['timeline_id'] not in timelines.keys():
            timelines[row['timeline_id']] = row['post']
        else:
            timelines[row['timeline_id']] = f'{timelines[row['timeline_id']]}\n\n{row['post']}'

    summaries = []
    for timeline in tqdm.tqdm(timelines.items()):
        summary = single_official_baseline_timeline_summary(timeline, model, tokenizer, device)
        summaries.append(summary)

    data['task_c_baseline_official_summaries'] = summaries
    new_csv_name = f'{post_csv_location[:-4]}_task_c_official_baseline.csv'
    data.to_csv(new_csv_name, index=False)

if __name__ == "__main__":
    run_task_c_official_baseline(post_csv_location)
    run_task_b_blue_baseline(post_csv_location)