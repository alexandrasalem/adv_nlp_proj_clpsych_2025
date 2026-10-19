import json
import ast
import pandas as pd
import argparse


def _parse_evidence_cell(value):
    """Coerce a CSV cell into a list of evidence spans.

    A cell may hold a JSON/Python list literal ("['span1', 'span2']") or a
    single plain-text span. Empty/NaN cells become an empty list.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Try to interpret a serialized list literal first.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except (ValueError, json.JSONDecodeError):
            pass
        # Otherwise treat the whole cell as one span.
        return [text]
    return [str(value)]

parser = argparse.ArgumentParser()

parser.add_argument("--task", default = "B", choices = ["B", "C"])
parser.add_argument("--alex_format", default = "no", choices = ["yes", "no"])
parser.add_argument("--model_name", choices = ["zero_shot_summary", "one_shot_summary", "task_b_baseline_official_summaries", "task_b_baseline_blue_team_summaries"])
parser.add_argument("--input_json_filepath")
parser.add_argument("--output_json_filepath")

args = parser.parse_args()

post_index_to_id = {}
post_data = pd.read_csv("post_data.csv")
for _, row in post_data.iterrows():
    if row['timeline_id'] not in post_index_to_id.keys():
        post_index_to_id[row['timeline_id']] = {row['post_index']: row['post_id']}
    else:
        post_index_to_id[row['timeline_id']][row['post_index']] = row['post_id']


def convert_dan_task_b_res_to_official_eval(results_loc, model_name, output_loc):
    new_output = {}

    for timeline in post_index_to_id.keys():
        timeline_post_ids = post_index_to_id[timeline].values()
        added_post_dict = {post_index: False for post_index in timeline_post_ids}

        new_output[timeline] = {"timeline_level": {"summary": ""}, "post_level": {}}
        with open(results_loc, "r", encoding="utf-8") as f:
            data = json.load(f)
            for post in data:
                timeline_id = post['timeline_id']
                if timeline_id != timeline:
                    continue
                post_index = post['post_index']
                post_id = post_index_to_id[timeline_id][post_index]

                post_output_template = {'adaptive_evidence': [], 'maladaptive_evidence': [], 'summary': '',
                                        'wellbeing_score': None}
                post_output_template['summary'] = post[model_name]
                # Populate evidence so run.py computes the Evidence Alignment metric.
                # run.py concatenates adaptive + maladaptive for Task B, so putting all
                # spans in adaptive_evidence is equivalent for Task B scoring.
                post_output_template['adaptive_evidence'] = post.get('evidence_texts', [])
                new_output[timeline_id]['post_level'][post_id] = post_output_template
                added_post_dict[post_id] = True
            for post_id, added in added_post_dict.items():
                if not added:
                    post_output_template = {'adaptive_evidence': [], 'maladaptive_evidence': [], 'summary': '',
                                            'wellbeing_score': None}
                    new_output[timeline]['post_level'][post_id] = post_output_template

    with open(output_loc, "w") as f:
        json.dump(new_output, f)


def convert_alex_task_b_res_to_official_eval(results_loc, model_name, output_loc):
    new_output = {}

    for timeline in post_index_to_id.keys():
        timeline_post_ids = post_index_to_id[timeline].values()

        new_output[timeline] = {"timeline_level": {"summary": ""}, "post_level": {}}
        blue_data = pd.read_csv(results_loc)
        has_adaptive = "adaptive_evidence" in blue_data.columns
        has_maladaptive = "maladaptive_evidence" in blue_data.columns
        for _, row in blue_data.iterrows():
            posts_for_this_timeline = post_index_to_id[timeline].values()
            post_id = row["post_id"]
            if post_id not in posts_for_this_timeline:
                continue
            post_output_template = {'adaptive_evidence': [], 'maladaptive_evidence': [], 'summary': '',
                                    'wellbeing_score': None}
            post_output_template['summary'] = row[model_name]
            # Populate evidence so run.py computes the Evidence Alignment metric.
            # Columns are optional: missing -> empty list. See _parse_evidence_cell
            # for how list-literal vs. plain-text cells are handled.
            if has_adaptive:
                post_output_template['adaptive_evidence'] = _parse_evidence_cell(row.get("adaptive_evidence"))
            if has_maladaptive:
                post_output_template['maladaptive_evidence'] = _parse_evidence_cell(row.get("maladaptive_evidence"))
            new_output[timeline]['post_level'][post_id] = post_output_template

    with open(output_loc, "w") as f:
        json.dump(new_output, f)


def convert_tamoghna_task_c_res_to_official_eval(results_loc, output_loc):
    new_output = {}
    with open(results_loc, "r", encoding="utf-8") as f:
        data = json.load(f)
    for predicted_timeline in data:
        timeline_id = predicted_timeline['timeline_id']
        new_output[timeline_id] = {"timeline_level": {"summary": predicted_timeline["predicted_summary"]}, "post_level": {}}

    for timeline in post_index_to_id.keys():
        posts_for_this_timeline = post_index_to_id[timeline].values()
        for post_id in posts_for_this_timeline:
            post_output_template = {'adaptive_evidence': [], 'maladaptive_evidence': [], 'summary': '',
                                    'wellbeing_score': None}
            new_output[timeline]['post_level'][post_id] = post_output_template

    with open(output_loc, "w") as f:
        json.dump(new_output, f)

def main():
    if args.alex_format == "no":
        if args.task == "B":
            convert_dan_task_b_res_to_official_eval(results_loc=args.input_json_filepath,
                                                    model_name=args.model_name,
                                                    output_loc=args.output_json_filepath)
        elif args.task == "C":
            convert_tamoghna_task_c_res_to_official_eval(results_loc=args.input_json_filepath,
                                                    output_loc=args.output_json_filepath)
        else:
            raise ValueError("Invalid input for --task")
    elif args.alex_format == "yes":
        if args.task == "B":
            convert_alex_task_b_res_to_official_eval(results_loc=args.input_json_filepath,
                                                    model_name=args.model_name,
                                                    output_loc=args.output_json_filepath)
        elif args.task == "C":
            raise NotImplementedError
        else:
            raise ValueError("Invalid input for --task")
    else:
        raise ValueError("Invalid input for --alex_format")

if __name__ == "__main__":
    main()

