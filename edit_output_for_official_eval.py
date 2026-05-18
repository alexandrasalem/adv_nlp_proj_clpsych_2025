import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--task", default = "B", choices = ["B", "C"])
parser.add_argument("--alex_format", default = "no", choices = ["yes", "no"])
parser.add_argument("--model_name", choices = ["zero_shot_summary", "one_shot_summary", "task_b_baseline_official_summaries", "task_b_blue_official_summaries"])
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
        with open(results_loc, "r") as f:
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
        for _, row in blue_data.iterrows():
            posts_for_this_timeline = post_index_to_id[timeline].values()
            post_id = row["post_id"]
            if post_id not in posts_for_this_timeline:
                continue
            post_output_template = {'adaptive_evidence': [], 'maladaptive_evidence': [], 'summary': '',
                                    'wellbeing_score': None}
            post_output_template['summary'] = row[model_name]
            new_output[timeline]['post_level'][post_id] = post_output_template

    with open(output_loc, "w") as f:
        json.dump(new_output, f)


def convert_tamoghna_task_c_res_to_official_eval(results_loc, output_loc):
    new_output = {}
    with open(results_loc, "r") as f:
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

