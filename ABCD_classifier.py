import os
import json
import datetime
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline import (
    load_timelines, extract_posts,
    ABCD_KEYS, ABCD_DEFINITIONS, STATE_TYPES,
    PostProfile
)

def run_cross_validation(
        posts: list[PostProfile],
        output_dir: str = "./outputs",
):
    """
    Run leave-one-timeline-out cross-validation for logistic regression.
    
    for each timeline use all other timelines as the training data
    and test on the one timeline left.

    if TF-IDF works well then categories are largely keyword driven
    if embeddings are better that means semantic understanding of the spans matter more

    NOTE: potentially concatenate the post embedding to the end of the span embedding before feeding
    into logistic regression (as another baseline if TF-IDF works well) to attempt to supply
    TF-IDF method with some context
    """
    print("\n" + "=" * 60)
    print("LEAVE ONE TIMELINE OUT CROSS-VALIDATION")
    print("=" * 60)

    timeline_ids = sorted(set(p.timeline_id for p in posts))
    print(f"Running over {len(timeline_ids)} timelines")

    all_results = []

    for tl_id in timeline_ids:
        # embedded evidence spans
        # targets
        target_spans = [vec for p in posts for vec in p.span_semantic_vectors if p.timeline_id == tl_id]
        # training pool
        pool_spans = [vec for p in posts for vec in p.span_semantic_vectors if p.timeline_id != tl_id]
        # training labels
        pool_labels = [span.gold_abcd_key for p in posts for span in p.evidence_spans if p.timeline_id != tl_id]

        tfidf = TfidfVectorizer()
        # TF-IDF representation of spans (baseline)
        X_train = tfidf.fit_transform([s.text for p in posts for s in p.evidence_spans if p.timeline_id != tl_id])
        X_test = tfidf.transform([s.text for p in posts for s in p.evidence_spans if p.timeline_id == tl_id])

        if not target_spans or not pool_spans or not pool_labels:
            continue

        print(f"\n  Timeline {tl_id}: {len(target_spans)} target spans, "
              f"{len(pool_spans)} pool posts")
        
        lr = LogisticRegression()
        lr.fit(np.array(pool_spans), pool_labels)
        
        lr_tf = LogisticRegression()
        lr_tf.fit(X_train)


def main():
    parser = argparse.ArgumentParser(
        description="ABCD Classification"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with timeline JSON files")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model to use for embedding the post text")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    output_dir = args.output_dir
    timestamp = datetime.datetime.now().strftime("%Y%M%d-%H%M%S")
    data_out_path = os.path.join(output_dir, f"abcd_classifications_{timestamp}.json")
    # DO YOU NEED TO SAVE ANY OTHER RAW DATA FOR DEBUGGING?

    timelines = load_timelines(args.data_dir)
    posts = extract_posts(timelines=timelines, context_window=0)
    # WE NOW HAVE POSTS WITH ACCESSABLE EMBEDDINGS AND SPAN EMBEDDINGS

    run_cross_validation(posts)