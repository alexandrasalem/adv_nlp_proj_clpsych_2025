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

def run_task_b_blue_baseline(post_csv_location):

    data = pd.read_csv(post_csv_location)
    summaries = []
    for index, row in tqdm.tqdm(data.iterrows()):
        post_text = row['post']
        summary = blue_baseline_one_post(post_text)
        summaries.append(summary)

    data['task_b_baseline_blue_team_summaries'] = summaries
    new_csv_name = f'{post_csv_location[:-4]}_task_b_blue_baseline.csv'
    data.to_csv(new_csv_name, index=False)

def single_official_baseline_post_summary(post_text, model, tokenizer, device):
    prompt = f"""
Analyze the following social media post and identify the dominant self-state (adaptive or maladaptive). Begin by determining which self-state is more dominant and describe it first. For each self-state, highlight the central organizing aspect - A (Affect), B (Behavior), C (Cognition), or D (Desire / Need) that drives the state. Describe how this central aspect influences the other aspects, focusing on the potential causal relationships between them. If the self-state is maladaptive, explain how negative emotions, behaviors, or thoughts hinder psychological needs, and if adaptive, explain how positive aspects support psychological needs. If both adaptive and maladaptive states are present, describe each in turn. If only one self-state is evident, focus solely on that. You must not make anything up. Keep the description concise and only describe observations if they are fully supported by the text. 
Post Content : { post_text } 
Summary :
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
    decoded_full = decoded_full.split("Summary :\n")[1]
    return decoded_full

def run_task_b_official_baseline(post_csv_location):
    # prepping model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8b-Instruct",
    )

    model.to(device)

    data = pd.read_csv(post_csv_location)
    summaries = []
    for index, row in tqdm.tqdm(data.iterrows()):
        post_text = row['post']
        summary = single_official_baseline_post_summary(post_text, model, tokenizer, device)
        summaries.append(summary)

    data['task_b_baseline_official_summaries'] = summaries
    new_csv_name = f'{post_csv_location[:-4]}_task_b_official_baseline.csv'
    data.to_csv(new_csv_name, index=False)

if __name__ == "__main__":
    run_task_b_official_baseline(post_csv_location)
    run_task_b_blue_baseline(post_csv_location)