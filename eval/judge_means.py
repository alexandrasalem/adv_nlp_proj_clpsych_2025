import pandas as pd
import os

df = pd.read_csv("outputs/llm_judge_gptoss_v2/llm_judge.csv")
cols = ["q1_overall_quality","q2_no_contradiction","q3_coverage",
        "q3a_dominant_self_state","q3b_element_recall",
        "q3c_unsupported_elements","q3d_polarity_correctness"]

# Coerce fraction strings to proportions for q3b and q3d
for c in ["q3b_element_recall", "q3d_polarity_correctness"]:
    df[c] = df[c].apply(
        lambda v: float(str(v).split("/")[0]) / float(str(v).split("/")[1])
        if isinstance(v, str) and "/" in v and float(str(v).split("/")[1]) != 0
        else float("nan")
    )

means = df.groupby("system")[cols].mean()
means = means.reindex(["Ours", "EAIonFlux", "Official_baseline", "BLUE"])
print("=== Mistral judge system-level means (n=120) ===")
print(means.round(3).to_string())
mean_output_path = os.path.join("outputs", "final_analysis", "mistral_judge", "judge_means.csv")
full_output_path = os.path.join("outputs", "final_analysis", "mistral_judge", "full_judge_scores.csv")
means.to_csv(mean_output_path)
df.to_csv(full_output_path)