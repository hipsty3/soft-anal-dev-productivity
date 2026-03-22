import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm  
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("processed/3.1_analysis_bins_data.csv")

# -------------------------------
# 1. Descriptive statistics
# -------------------------------
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Summary statistics:\n", df.describe())

print(f"AI Group value count: {df['ai_group'].value_counts()}")

# Aggregate by language
df["ai_group"] = df["ai_group"].astype(str)

language_summary = df.groupby("language").agg(
    n_months=("commit_author", "count"),
    mean_commits=("commits", "mean"),
    median_commits=("commits", "median"),
    mean_total_changes=("total_changes", "mean"),
    median_total_changes=("total_changes", "median"),
    ai_none=("ai_group", lambda x: (x == "none").sum()),
    ai_low=("ai_group", lambda x: (x == "low").sum()),
    ai_medium=("ai_group", lambda x: (x == "medium").sum()),
    ai_high=("ai_group", lambda x: (x == "high").sum()),
    ai_very_high=("ai_group", lambda x: (x == "very_high").sum())
).reset_index()

# Optional: sort by number of months
language_summary = language_summary.sort_values(by="n_months", ascending=False)

print(language_summary)


sns.boxplot(data=df, x="ai_group", y="log_commits")
plt.show()

sns.boxplot(data=df, x="ai_group", y="log_total_changes")
plt.show()

df["ai_group"].value_counts()