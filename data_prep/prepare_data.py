import os
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Arguments for setting up the data.")
parser.add_argument("--data_files_location", default = "train-clpsych2025-v2/", help="full path to the data folder")
parser.add_argument("--timeline_data_output_filename", default = "timeline_data.csv", help="output filename for the timeline data")
parser.add_argument("--post_data_output_filename", default = "post_data.csv", help="output filename for the post data")

args = parser.parse_args()

data_files_location = args.data_files_location
timeline_data_output_filename = args.timeline_data_output_filename
post_data_output_filename = args.post_data_output_filename

def prepare_timeline_data(files_location = data_files_location, output_filename = timeline_data_output_filename):
    files = os.listdir(files_location)
    timeline_ids = []
    timeline_summaries = []
    for file in files:
        file_loc = f'{files_location}{file}'
        with open(file_loc, "r") as f:
            file_json = json.load(f)
            timeline_id = file_json['timeline_id']
            timeline_summary = file_json['timeline_summary']
            timeline_ids.append(timeline_id)
            timeline_summaries.append(timeline_summary)

    data = pd.DataFrame({'timeline_id': timeline_ids,
                         'timeline_summary': timeline_summaries})
    data.to_csv(output_filename, index=False)
    return data

def prepare_post_data(files_location = data_files_location, output_filename = post_data_output_filename):
    files = os.listdir(files_location)
    post_indices = []
    post_ids = []
    post_dates = []
    posts = []
    post_summaries = []
    well_being_scores = []
    maladaptive_evidences = []
    adaptive_evidences = []
    for file in files:
        file_loc = f'{files_location}{file}'
        with open(file_loc, "r") as f:
            file_json = json.load(f)
            for post in file_json['posts']:
                post_idx = post['post_index']
                post_id = post['post_id']
                post_text = post['post']
                post_date = post['date']
                if post['Post Summary'] is None:
                    summary = "NONE"
                else:
                    summary = post['Post Summary']
                if post['Well-being'] is None:
                    well_being = "NONE"
                else:
                    well_being = post['Well-being']
                if post['evidence'] is None:
                    maladaptive_evidence = "NONE"
                    adaptive_evidence = "NONE"
                else:
                    maladaptive_evidence = post['evidence']['maladaptive-state']
                    adaptive_evidence = post['evidence']['adaptive-state']
                post_indices.append(post_idx)
                post_ids.append(post_id)
                post_dates.append(post_date)
                posts.append(post_text)
                post_summaries.append(summary)
                well_being_scores.append(well_being)
                maladaptive_evidences.append(maladaptive_evidence)
                adaptive_evidences.append(adaptive_evidence)

    data = pd.DataFrame({'post_index': post_indices,
                         'post_id': post_ids,
                         'post_date': post_dates,
                         'post': posts,
                         'post_summary': post_summaries,
                         'well_being_score': well_being_scores,
                         'maladaptive_evidence': maladaptive_evidences,
                         'adaptive_evidence': adaptive_evidences})
    data.to_csv(output_filename, index=False)
    return data

if __name__ == '__main__':
    prepare_timeline_data()
    prepare_post_data()
