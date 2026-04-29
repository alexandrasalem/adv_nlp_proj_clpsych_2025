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
    prompt = f"Given the following Reddit post, summarize the interplay between adaptive and maladaptive self-states."
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": post_text},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = model.generate(**inputs,
                             temperature=0.7,
                             top_p=0.9,
                             do_sample=True,
                             max_new_tokens=128)

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("here")
    print(decoded_output)
    print("leaving")
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
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        device_map="auto",
    )

    # running on post
    output = single_blue_team_post_summary(post_text = post_test, model = model, tokenizer = tokenizer, device = device)
    print(output)

if __name__ == '__main__':
    main()