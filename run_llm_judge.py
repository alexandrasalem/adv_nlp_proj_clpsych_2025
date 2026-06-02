"""
LLM-as-a-judge evaluation pipeline for the CLPSych 2025 Shared Task 
that runs Llama-3.1-8B-Instruct locally via Ollama. 

Compares candidate summaries against gold standard summaries for Task
B and scores them according to the devised annotation guidelines.

Reads: 
- a JSON judge config
- a JSONL evaluation dataset

Output:
- a JSONL of raw judgments
- a CSV of judgments

Usage:
    python run_llm_judge.py

Install:
    pip install ollama

Run Ollama:
    ollama pull llama3.1:8b
"""

import json
import csv
from pathlib import Path

import ollama


# Config
MODEL = "llama3.1:8b"

INPUT_FILE = "data/test.jsonl"
JUDGE_CONFIG_FILE = "prompts/judge_config.json"

RAW_OUTPUT_FILE = "outputs/raw_judgments.jsonl"
CSV_OUTPUT_FILE = "outputs/judgments.csv"


def load_judge_config(path):
    with open(path, "r", encoding="utf-8") as sink:
        return json.load(sink)


def build_few_shot(config):
    examples = config.get("few_shot_examples", [])

    if not examples:
        return ""
    
    blocks = []

    for i, ex in enumerate(examples, start=1):
        blocks.append(f"""
EXAMPLE{i}

[POST]
{ex["post"]}
[/POST]

[GOLD SUMMARY]
{ex["gold_summary"]}
[/GOLD SUMMARY]

[CANDIDATE SUMMARY]
{ex["candidate_summary"]}
[/CANDIDATE SUMMARY]

[CORRECT JUDGMENT]
{json.dumps(ex["judgment"], indent=2)}
[/CORRECT JUDGMENT]
""")

    return "\n".join(blocks)


def build_system_prompt(config):
    few_shot_section = build_few_shot(config)

    return f"""
You are acting as an LLM-as-a-judge for the following evaluation task.

Use the rubric and examples below to judge the candidate summary.

TASK
{json.dumps(config["task_description"], indent=2)}

CLINICAL FRAMEWORK
{json.dumps(config["clinical_framework"], indent=2)}

SELF-STATE DEFINITIONS
{json.dumps(config["self_state_definitions"], indent=2)}

EVALUATION DIMENSIONS
{json.dumps(config["evaluation_dimensions"], indent=2)}

EVALUATION RULES
{json.dumps(config["evaluation_rules"], indent=2)}

REQUIRED OUTPUT SCHEMA
{json.dumps(config["required_output_schema"], indent=2)}

FEW-SHOT EXAMPLES
{few_shot_section}

Now evaluate the candidate summary.

Return valid JSON only.
"""


def build_user_message(item):
    return {
        "role": "user",
        "content": f"""
Evaluate the candidate summary below.

[POST]
{item["post"]}
[/POST]

[GOLD SUMMARY]
{item["gold_summary"]}
[/GOLD SUMMARY]

[CANDIDATE SUMMARY]
{item["candidate_summary"]}
[/CANDIDATE SUMMARY]
"""
    }


def run_judge(system_prompt, item):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            build_user_message(item)
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    return json.loads(response["message"]["content"])


def main():
    Path("outputs").mkdir(exist_ok=True)

    config = load_judge_config(JUDGE_CONFIG_FILE)
    system_prompt = build_system_prompt(config)

    raw_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            item = json.loads(line)

            item_id = item["item_id"]
            system_id = item.get("system", "unknown")

            judgment = run_judge(system_prompt, item)

            row = {
                "item_id": item_id,
                "system": system_id,
                "model": MODEL,
                "judge_config_name": config.get("name", "unknown"),
                "judge_config_version": config.get("version", "unknown"),
                "overall_quality": judgment["overall_quality"],
                "no_contradiction": judgment["no_contradiction"],
                "coverage": judgment["coverage"],
                "rationale": judgment.get("rationale", "")
            }

            raw_rows.append(row)

            print(f"Finished {item_id} | {system_id}")

    with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for row in raw_rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    print("Done.")
    print(f"Raw judgments saved to: {RAW_OUTPUT_FILE}")
    print(f"CSV judgments saved to: {CSV_OUTPUT_FILE}")


if __name__ == "__main__":
    main()