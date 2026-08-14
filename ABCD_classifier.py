import os
import json
import datetime
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from pipeline import (
    load_timelines, extract_posts,
    ABCD_KEYS, ABCD_DEFINITIONS, STATE_TYPES,
    PostProfile
)

def build_definition_embeddings(model: SentenceTransformer) -> dict:
    """
    Embed each ABCD definition text, producing 12 vectors:
    6 ABCD keys x 2 polarities (adaptive, maladaptive).

    Returns dict mapping (abcd_key, polarity) -> normalized embedding vector.
    """
    texts = {}
    for key in ABCD_KEYS:
        defn = ABCD_DEFINITIONS[key]
        for polarity in ["adaptive", "maladaptive"]:
            example_key = f"{polarity}_examples"
            text = (
                f"{defn['name']} ({polarity}): {defn['description']} "
                f"Examples: {defn[example_key]}"
            )
            texts[(key, polarity)] = text

    # Embed all 12 definition texts at once
    keys_ordered = list(texts.keys())
    texts_ordered = [texts[k] for k in keys_ordered]
    vecs = model.encode(texts_ordered, normalize_embeddings=True)

    return {k: v for k, v in zip(keys_ordered, vecs)}


def compute_definition_similarity_features(
    posts: list[PostProfile],
    def_embeddings: dict,
) -> list[np.ndarray]:
    """
    For each evidence span, compute cosine similarity to all 12 definition
    embeddings (6 ABCD keys x 2 polarities), producing a 12-dim feature vector.

    Also computes a 6-dim polarity-matched version: for each span, only use
    the similarities to definitions matching that span's polarity.

    Returns list of feature dicts, one per span, in the same flattened order
    as iterating (post, span) over posts.
    """
    # Ordered keys for consistent indexing
    all_def_keys = [(k, p) for p in ["adaptive", "maladaptive"] for k in ABCD_KEYS]
    def_matrix = np.array([def_embeddings[k] for k in all_def_keys])  # (12, 768)

    matched_def_keys = {
        "adaptive": [(k, "adaptive") for k in ABCD_KEYS],
        "maladaptive": [(k, "maladaptive") for k in ABCD_KEYS],
    }

    features_12 = []  # 12-dim: similarity to all definitions
    features_6 = []   # 6-dim: similarity to polarity-matched definitions only

    for p in posts:
        for span_vec, span in zip(p.span_semantic_vectors, p.evidence_spans):
            # 12-dim: cosine similarity to all 12 definitions
            # Vectors are already normalized, so dot product = cosine similarity
            sims_12 = span_vec @ def_matrix.T
            features_12.append(sims_12)

            # 6-dim: only the 6 definitions matching this span's polarity
            matched_keys = matched_def_keys[span.polarity]
            matched_matrix = np.array([def_embeddings[k] for k in matched_keys])
            sims_6 = span_vec @ matched_matrix.T
            features_6.append(sims_6)

    return features_12, features_6


def run_cross_validation(
        posts: list[PostProfile],
        output_dir: str = "./outputs",
):
    """
    Run leave-one-timeline-out cross-validation for logistic regression.
    
    Compares four methods on the same folds:
        1. Log Regression on dense embeddings
        2. SVM (RBF kernel) on dense span embeddings
        3. Log Regression on TF-IDF features
        4. SVM (RBF kernel) on TF-IDF features

    if TF-IDF works well then categories are largely keyword driven
    if embeddings are better that means semantic understanding of the spans matter more

    NOTE: potentially concatenate the post embedding to the end of the span embedding before feeding
    into logistic regression (as another baseline if TF-IDF works well) to attempt to supply
    TF-IDF method with some context
    """
    print("\n" + "=" * 60)
    print("LEAVE ONE TIMELINE OUT CROSS-VALIDATION")
    print("=" * 60)

    posts_with_spans = [p for p in posts if p.evidence_spans]
    timeline_ids = sorted(set(p.timeline_id for p in posts_with_spans))
    print(f"Running over {len(timeline_ids)} timelines, "
          f"{sum(len(p.evidence_spans) for p in posts_with_spans)} total spans")
    
    # Pre-compute definition similarity features (once, outside the loop) 
    print("Embedding ABCD definitions...")
    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    def_embeddings = build_definition_embeddings(st_model)
    del st_model  # free memory

    features_12, features_6 = compute_definition_similarity_features(
        posts_with_spans, def_embeddings
    )

    # Build a flat index so we can slice per-fold
    # Map (timeline_id, span_position_in_flat_list) -> feature vectors
    span_tl_ids = [
        p.timeline_id
        for p in posts_with_spans
        for _ in p.evidence_spans
    ]
    features_12 = np.array(features_12)  # (n_spans, 12)
    features_6 = np.array(features_6)    # (n_spans, 6)

    # Zero-shot baseline: argmax of polarity-matched 6-dim similarities
    zs_preds = [ABCD_KEYS[np.argmax(f)] for f in features_6]
    zs_labels = [s.gold_abcd_key for p in posts_with_spans for s in p.evidence_spans]
    zs_acc = accuracy_score(zs_labels, zs_preds)
    zs_f1 = f1_score(zs_labels, zs_preds, labels=ABCD_KEYS, average="macro", zero_division=0)
    print(f"\nZero-shot (argmax definition similarity): "
          f"Accuracy={zs_acc:.4f}, Macro F1={zs_f1:.4f}")
    
    methods = [
        "lr_embed", "svm_embed",
        "lr_tfidf", "svm_tfidf",
        "lr_span_post", "svm_span_post",
        "lr_enriched", "svm_enriched",
        "lr_defsim6", "svm_defsim6",
        "lr_defsim12", "svm_defsim12",
        "lr_embed_defsim6", "svm_embed_defsim6",
    ]
    all_gold = {m: [] for m in methods}
    all_pred = {m: [] for m in methods}
    fold_results = []

    for tl_id in timeline_ids:
        # split posts by timeline fold
        train_posts = [p for p in posts_with_spans if p.timeline_id != tl_id]
        test_posts = [p for p in posts_with_spans if p.timeline_id == tl_id]

        # get the train, test, and labels
        train_vecs = [vec for p in train_posts for vec in p.span_semantic_vectors]
        test_vecs = [vec for p in test_posts for vec in p.span_semantic_vectors]
        train_labels = [s.gold_abcd_key for p in train_posts for s in p.evidence_spans]
        test_labels = [s.gold_abcd_key for p in test_posts for s in p.evidence_spans]
        train_texts = [s.text for p in train_posts for s in p.evidence_spans]
        test_texts = [s.text for p in test_posts for s in p.evidence_spans]


        # Span + post concatenation 
        # Pair each span embedding with its parent post's semantic_vector
        train_span_post = [
            np.concatenate([span_vec, p.semantic_vector])
            for p in train_posts
            for span_vec in p.span_semantic_vectors
        ]
        test_span_post = [
            np.concatenate([span_vec, p.semantic_vector])
            for p in test_posts
            for span_vec in p.span_semantic_vectors
        ]

        # Enriched features: span+post embedding + handcrafted signals 
        def build_enriched(posts_subset):
            rows = []
            for p in posts_subset:
                post_len = len(p.post_text.split())
                for span_vec, span in zip(p.span_semantic_vectors, p.evidence_spans):
                    span_len = len(span.text.split())
                    handcrafted = np.array([
                        1.0 if span.polarity == "adaptive" else 0.0,   # polarity one-hot
                        0.0 if span.polarity == "adaptive" else 1.0,
                        span_len / max(post_len, 1),                   # span/post length ratio
                        min(span_len / 20.0, 1.0),                     # normalized span length
                    ], dtype=np.float32)
                    rows.append(np.concatenate([span_vec, p.semantic_vector, handcrafted]))
            return rows

        train_enriched = build_enriched(train_posts)
        test_enriched = build_enriched(test_posts)

        if not test_vecs or not train_vecs:
            continue

        X_train_embed = np.array(train_vecs)
        X_test_embed = np.array(test_vecs)
        X_train_sp = np.array(train_span_post)
        X_test_sp = np.array(test_span_post)
        X_train_enr = np.array(train_enriched)
        X_test_enr = np.array(test_enriched)

        # TF-IDF fit on training spans
        tfidf = TfidfVectorizer()
        X_train_tfidf = tfidf.fit_transform(train_texts)
        X_test_tfidf = tfidf.transform(test_texts)

        # train and predict with each method
        fold_predictions = {}

        # Logistic regression on embeddings
        lr_embed = LogisticRegression(max_iter=1000, C=1.0)
        lr_embed.fit(X_train_embed, train_labels)
        fold_predictions["lr_embed"] = lr_embed.predict(X_test_embed).tolist()

        # SVM on embeddings
        svm_embed = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_embed.fit(X_train_embed, train_labels)
        fold_predictions["svm_embed"] = svm_embed.predict(X_test_embed).tolist()

        # Logistic regression on TF-IDF
        lr_tfidf = LogisticRegression(max_iter=1000, C=1.0)
        lr_tfidf.fit(X_train_tfidf, train_labels)
        fold_predictions["lr_tfidf"] = lr_tfidf.predict(X_test_tfidf).tolist()

        # SVM on TF-IDF
        svm_tfidf = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_tfidf.fit(X_train_tfidf, train_labels)
        fold_predictions["svm_tfidf"] = svm_tfidf.predict(X_test_tfidf).tolist()

        # Logistic Regression on span + post concatenation
        lr_sp = LogisticRegression(max_iter=1000, C=1.0)
        lr_sp.fit(X_train_sp, train_labels)
        fold_predictions["lr_span_post"] = lr_sp.predict(X_test_sp).tolist()

        # SVM (RBF) on span + post concatenation
        svm_sp = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_sp.fit(X_train_sp, train_labels)
        fold_predictions["svm_span_post"] = svm_sp.predict(X_test_sp).tolist()

        # Logistic Regression on enriched (span + post + handcrafted)
        lr_enr = LogisticRegression(max_iter=1000, C=1.0)
        lr_enr.fit(X_train_enr, train_labels)
        fold_predictions["lr_enriched"] = lr_enr.predict(X_test_enr).tolist()

        # SVM (RBF) on enriched (span + post + handcrafted)
        svm_enr = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_enr.fit(X_train_enr, train_labels)
        fold_predictions["svm_enriched"] = svm_enr.predict(X_test_enr).tolist()

        # ── Definition similarity features: slice from precomputed arrays ──
        train_mask = np.array([tid != tl_id for tid in span_tl_ids])
        test_mask = np.array([tid == tl_id for tid in span_tl_ids])

        X_train_ds6 = features_6[train_mask]    # (n_train, 6)
        X_test_ds6 = features_6[test_mask]      # (n_test, 6)
        X_train_ds12 = features_12[train_mask]  # (n_train, 12)
        X_test_ds12 = features_12[test_mask]    # (n_test, 12)

        # Combine: span embedding + 6-dim definition similarity = 774-dim
        # Much smaller than span+post (1536) but adds targeted category signals
        X_train_eds6 = np.hstack([X_train_embed, X_train_ds6])
        X_test_eds6 = np.hstack([X_test_embed, X_test_ds6])

        # LR on 6-dim polarity-matched definition similarity
        lr_ds6 = LogisticRegression(max_iter=1000, C=1.0)
        lr_ds6.fit(X_train_ds6, train_labels)
        fold_predictions["lr_defsim6"] = lr_ds6.predict(X_test_ds6).tolist()

        # SVM on 6-dim polarity-matched definition similarity
        svm_ds6 = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_ds6.fit(X_train_ds6, train_labels)
        fold_predictions["svm_defsim6"] = svm_ds6.predict(X_test_ds6).tolist()

        # LR on 12-dim full definition similarity
        lr_ds12 = LogisticRegression(max_iter=1000, C=1.0)
        lr_ds12.fit(X_train_ds12, train_labels)
        fold_predictions["lr_defsim12"] = lr_ds12.predict(X_test_ds12).tolist()

        # SVM on 12-dim full definition similarity
        svm_ds12 = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_ds12.fit(X_train_ds12, train_labels)
        fold_predictions["svm_defsim12"] = svm_ds12.predict(X_test_ds12).tolist()

        # LR on span embedding + 6-dim definition similarity (774-dim)
        lr_eds6 = LogisticRegression(max_iter=1000, C=1.0)
        lr_eds6.fit(X_train_eds6, train_labels)
        fold_predictions["lr_embed_defsim6"] = lr_eds6.predict(X_test_eds6).tolist()

        # SVM on span embedding + 6-dim definition similarity (774-dim)
        svm_eds6 = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm_eds6.fit(X_train_eds6, train_labels)
        fold_predictions["svm_embed_defsim6"] = svm_eds6.predict(X_test_eds6).tolist()


        fold_entry = {
            "timeline_id": tl_id,
            "n_test_spans": len(test_labels), 
            "n_train_spans": len(train_labels),
        }

        for m in methods:
            preds = fold_predictions[m]
            all_gold[m].extend(test_labels)
            all_pred[m].extend(preds)
            fold_acc = accuracy_score(test_labels, preds)
            fold_entry[f"{m}_accuracy"] = round(fold_acc, 4)

        fold_results.append(fold_entry)
        print(f"  Timeline {tl_id}: {len(test_labels)} spans | "
              f"LR-Emb={fold_entry['lr_embed_accuracy']:.3f} "
              f"SVM-Emb={fold_entry['svm_embed_accuracy']:.3f} "
              f"LR-TF={fold_entry['lr_tfidf_accuracy']:.3f} "
              f"SVM-TF={fold_entry['svm_tfidf_accuracy']:.3f} "
              f"LR-SP={fold_entry['lr_span_post_accuracy']:.3f} "
              f"SVM-SP={fold_entry['svm_span_post_accuracy']:.3f} "
              f"LR-ENR={fold_entry['lr_enriched_accuracy']:.3f} "
              f"SVM-ENR={fold_entry['svm_enriched_accuracy']:.3f} "
              f"DS6={fold_entry['lr_defsim6_accuracy']:.3f} "
              f"SDS6={fold_entry['svm_defsim6_accuracy']:.3f} "
              f"DS12={fold_entry['lr_defsim12_accuracy']:.3f} "
              f"SDS12={fold_entry['svm_defsim12_accuracy']:.3f} "
              f"E+DS6={fold_entry['lr_embed_defsim6_accuracy']:.3f} "
              f"E+SDS6={fold_entry['svm_embed_defsim6_accuracy']:.3f}")

    # ── Global metrics across all folds ──
    print("\n" + "=" * 60)
    print("GLOBAL RESULTS (all folds aggregated)")
    print("=" * 60)

    abcd_names = [ABCD_DEFINITIONS[k]["name"] for k in ABCD_KEYS]
    global_metrics = {}

    for m in methods:
        acc = accuracy_score(all_gold[m], all_pred[m])
        macro_f1 = f1_score(all_gold[m], all_pred[m], labels=ABCD_KEYS,
                            average="macro", zero_division=0)
        weighted_f1 = f1_score(all_gold[m], all_pred[m], labels=ABCD_KEYS,
                               average="weighted", zero_division=0)

        global_metrics[m] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        }

    # Print comparison table
    # Add zero-shot to metrics for display
    global_metrics["zero_shot"] = {
        "accuracy": round(zs_acc, 4),
        "macro_f1": round(zs_f1, 4),
        "weighted_f1": round(float(f1_score(zs_labels, zs_preds, labels=ABCD_KEYS,
                                             average="weighted", zero_division=0)), 4),
    }
    all_methods_display = ["zero_shot"] + methods
    print(f"\n{'Method':<20} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 52)
    for m in all_methods_display:
        g = global_metrics[m]
        print(f"  {m:<18} {g['accuracy']:>10.4f} {g['macro_f1']:>10.4f} {g['weighted_f1']:>12.4f}")

    # Print full classification report for the best method
    best_method = max(methods, key=lambda m: global_metrics[m]["macro_f1"])
    print(f"\nBest method by macro F1: {best_method}")
    print(f"\nClassification report ({best_method}):")
    print(classification_report(
        all_gold[best_method], all_pred[best_method],
        labels=ABCD_KEYS, target_names=abcd_names, zero_division=0,
    ))

    # ── Save results ──
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_timelines": len(timeline_ids),
        "n_total_spans": len(all_gold[methods[0]]),
        "global_metrics": global_metrics,
        "best_method": best_method,
        "fold_results": fold_results,
        "per_span_predictions": {
            m: {
                # "gold": all_gold[m],
                "pred": all_pred[m],
            }
            for m in methods
        },
    }

    cm = confusion_matrix(all_gold[best_method], all_pred[best_method], labels=ABCD_KEYS)
    disp = ConfusionMatrixDisplay(cm, display_labels=ABCD_KEYS)
    disp.plot()
    plt.savefig("abcd_confusion.png")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(output_dir, f"abcd_classifier_cv_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output



def main():
    parser = argparse.ArgumentParser(
        description="ABCD Classification — Logistic Regression & SVM baselines"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with timeline JSON files")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    timelines = load_timelines(args.data_dir)
    # context_window=0 bc we only need span embeddings, not timeline context
    posts = extract_posts(timelines=timelines, context_window=0)

    run_cross_validation(posts, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
