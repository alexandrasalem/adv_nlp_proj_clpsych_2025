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
def single_blue_team_post_summary(post_text, model, tokenizer, device):
    prompt = f"""
    Given the following Reddit post, summarize the interplay between adaptive and maladaptive self-states.

    Post:
    \"{post_text}\"

    Response format:
    {{ "summary": "<post-level summary>" }}
    """

    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**input_ids, max_new_tokens=128)
    decoded_output = tokenizer.decode(outputs[0])
    return decoded_output


def main():
    # pulling sample post (for now)
    data = pd.read_csv(post_csv_location)
    for index, row in data.iterrows():
        post_test = row['post']
        break

    # prepping model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        device_map="auto",
    )
    model.to(device)

    # running on post
    output = single_blue_team_post_summary(post_text = post_test, model = model, tokenizer = tokenizer, device = device)
    print(output)

if __name__ == '__main__':
    main()