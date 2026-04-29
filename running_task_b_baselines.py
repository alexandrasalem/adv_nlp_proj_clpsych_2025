import pandas as pd
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description="Arguments for setting up the baseline runs.")
parser.add_argument("--post_csv_location", default = "post_data.csv", help="full path to the data folder")

args = parser.parse_args()

post_csv_location = args.post_csv_location


# Baseline provided by the Shared Task
# add here

# Baseline from BLUE team
# def single_blue_team_post_summary(post_text, model, tokenizer, device):
#     prompt = f"""
#     Given the following Reddit post, summarize the interplay between adaptive and maladaptive self-states.
#
#     Post:
#     \"{post_text}\"
#
#     Response format:
#     {{ "summary": "<post-level summary>" }}
#     """
#
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
#     outputs = model.generate(**inputs,
#                              temperature=0.7,
#                              top_p=0.9,
#                              do_sample=True,
#                              max_new_tokens=128)
#     generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
#     decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#     print("here")
#     print(decoded_output)
#     print("leaving")
#     return decoded_output


def single_blue_team_post_summary(post_text, model, tokenizer, device):
    prompt = f"Summarize this:\n{post_text}"

    inputs = tokenizer(prompt, return_tensors="pt")

    print("INPUT SHAPE:", inputs["input_ids"].shape)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # turn OFF sampling for debugging
        pad_token_id=tokenizer.eos_token_id,
    )

    print("OUTPUT SHAPE:", outputs.shape)

    decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("FULL DECODE:", decoded_full)

    return decoded_full

def main():
    # pulling sample post (for now)
    data = pd.read_csv(post_csv_location)
    for index, row in data.iterrows():
        post_test = row['post']
        break

    # prepping model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
    )

    model.to(device)

    # running on post
    output = single_blue_team_post_summary(post_text = post_test, model = model, tokenizer = tokenizer, device = device)
    print(output)

if __name__ == '__main__':
    main()