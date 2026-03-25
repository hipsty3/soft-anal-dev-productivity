import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.utils import (
    interpret_coef,
    save_show,
    paired_boxplot,
    paired_boxplot_with_means,
    paired_regplot,
    paired_lineplot_from_summary,
)

# -------------------------------
# Setup
# -------------------------------
os.makedirs("plots", exist_ok=True)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("processed/3_analysis_ready_data.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Summary statistics:\n", df.describe())

print("\nAI Group value count:")
print(df["ai_group"].value_counts())

print("\nUsed AI value count:")
print(df["used_ai"].value_counts())

# -------------------------------
# 1. Language summary
# -------------------------------
language_summary = (
    df.groupby("language")
    .agg(
        n_months=("commit_author", "count"),
        mean_commits=("commits", "mean"),
        median_commits=("commits", "median"),
        mean_total_changes=("total_changes", "mean"),
        median_total_changes=("total_changes", "median"),
        mean_ai_usage=("ai_usage", "mean"),
        ai_users=("used_ai", "sum"),
        pct_ai_users=("used_ai", "mean"),
    )
    .reset_index()
    .sort_values(by="n_months", ascending=False)
)

print("\nLanguage summary:")
print(language_summary)

# -------------------------------
# 2. Fast fixed-effects preparation
# -------------------------------
# Developer fixed effects are implemented by demeaning variables
# within each developer instead of creating thousands of dummies.
for col in ["log_commits", "log_total_changes", "used_ai", "ai_usage"]:
    df[f"{col}_dm"] = df[col] - df.groupby("commit_author")[col].transform("mean")

# -------------------------------
# 3. Adoption effect
# -------------------------------
print("\n==============================")
print("ADOPTION EFFECT (used_ai)")
print("==============================")

model1 = smf.ols(
    "log_commits_dm ~ used_ai_dm + C(year) + C(month)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["commit_author"]})

print(model1.summary())
coef1 = model1.params["used_ai_dm"]
print(f"Interpretation: Using AI is associated with {interpret_coef(coef1) * 100:.2f}% change in commits\n")

model2 = smf.ols(
    "log_total_changes_dm ~ used_ai_dm + C(year) + C(month)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["commit_author"]})

print(model2.summary())
coef2 = model2.params["used_ai_dm"]
print(f"Interpretation: Using AI is associated with {interpret_coef(coef2) * 100:.2f}% change in total changes\n")

# -------------------------------
# 4. Intensity effect (among AI users)
# -------------------------------
# Filter out near-zero AI usage so this model captures intensity,
# not just the difference between zero and non-zero users.
ai_df = df[df["ai_usage"] > 0.01].copy()

# Recompute demeaned variables within filtered sample
for col in ["log_commits", "log_total_changes", "ai_usage"]:
    ai_df[f"{col}_dm"] = ai_df[col] - ai_df.groupby("commit_author")[col].transform("mean")

print("\n==============================")
print("INTENSITY EFFECT (ai_usage)")
print("==============================")

model3 = smf.ols(
    "log_commits_dm ~ ai_usage_dm + C(year) + C(month)",
    data=ai_df
).fit(cov_type="cluster", cov_kwds={"groups": ai_df["commit_author"]})

print(model3.summary())
coef3 = model3.params["ai_usage_dm"]
print(f"Interpretation: A 10% increase in AI usage is associated with {interpret_coef(coef3 * 0.1) * 100:.2f}% change in commits\n")

model4 = smf.ols(
    "log_total_changes_dm ~ ai_usage_dm + C(year) + C(month)",
    data=ai_df
).fit(cov_type="cluster", cov_kwds={"groups": ai_df["commit_author"]})

print(model4.summary())
coef4 = model4.params["ai_usage_dm"]
print(f"Interpretation: A 10% increase in AI usage is associated with {interpret_coef(coef4 * 0.1) * 100:.2f}% change in total changes\n")

# -------------------------------
# 5. Plots
# -------------------------------

# 5.1 Productivity by AI usage group
paired_boxplot(
    data=df,
    x="ai_group",
    y1="log_commits",
    y2="log_total_changes",
    title1="Log Commits by AI Usage Group",
    title2="Log Total Changes by AI Usage Group",
    xlabel="AI Usage Group",
    ylabel1="Log(Commits)",
    ylabel2="Log(Total Changes)",
    save_path="plots/productivity_by_ai_group.png",
)

# 5.2 Distribution of AI usage
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["ai_usage"], bins=50, ax=ax)
ax.set_title("Distribution of AI Usage")
ax.set_xlabel("AI Usage")
ax.set_ylabel("Count")
save_show(fig, "plots/ai_usage_distribution.png")

# 5.3 Productivity by AI adoption
paired_boxplot(
    data=df,
    x="used_ai",
    y1="log_commits",
    y2="log_total_changes",
    title1="Commits by AI Adoption",
    title2="Total Changes by AI Adoption",
    xlabel="Used AI (0 = No, 1 = Yes)",
    ylabel1="Log(Commits)",
    ylabel2="Log(Total Changes)",
    save_path="plots/productivity_by_ai_adoption.png",
)

# 5.4 AI usage vs productivity
paired_regplot(
    data=ai_df,
    x="ai_usage",
    y1="log_commits",
    y2="log_total_changes",
    title1="AI Usage vs Commits",
    title2="AI Usage vs Total Changes",
    xlabel="AI Usage",
    ylabel1="Log(Commits)",
    ylabel2="Log(Total Changes)",
    save_path="plots/ai_usage_vs_productivity.png",
)

# 5.5 Non-linear AI usage effect
x_vals = np.linspace(ai_df["ai_usage"].min(), ai_df["ai_usage"].max(), 100)

# 5.6 AI adoption over time
time_trend = df.groupby(["year", "month"])["used_ai"].mean().reset_index()
time_trend["date"] = time_trend["year"].astype(str) + "-" + time_trend["month"].astype(str)

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=time_trend, x="date", y="used_ai", ax=ax)
ax.set_title("AI Adoption Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Proportion Using AI")
plt.xticks(rotation=45)
save_show(fig, "plots/ai_adoption_over_time.png")

# 5.7 Productivity by AI group with means
paired_boxplot_with_means(
    data=df,
    x="ai_group",
    y1="log_commits",
    y2="log_total_changes",
    title1="Commits by AI Usage Group (with Means)",
    title2="Total Changes by AI Usage Group (with Means)",
    xlabel="AI Usage Group",
    ylabel1="Log(Commits)",
    ylabel2="Log(Total Changes)",
    save_path="plots/productivity_by_ai_group_with_means.png",
)

# 5.9 AI group usage vs productivity
bin_summary = (
    ai_df.groupby("ai_group", observed=False)
    .agg(
        mean_log_commits=("log_commits", "mean"),
        mean_log_total_changes=("log_total_changes", "mean"),
    )
    .reset_index()
)

paired_lineplot_from_summary(
    x_vals=range(len(bin_summary)),
    y1_vals=bin_summary["mean_log_commits"],
    y2_vals=bin_summary["mean_log_total_changes"],
    title1="Average Commits by AI Usage Bins",
    title2="Average Total Changes by AI Usage Bins",
    xlabel="AI Usage Bin Index",
    ylabel1="Mean Log(Commits)",
    ylabel2="Mean Log(Total Changes)",
    save_path="plots/binned_ai_usage_productivity.png",
)

# 5.10 Adoption effect with raw data points (single plot)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="used_ai", y="log_total_changes", ax=ax)
sns.stripplot(data=df, x="used_ai", y="log_total_changes", alpha=0.2, color="black", ax=ax)
ax.set_title("Adoption Effect with Raw Data")
ax.set_xlabel("Used AI (0 = No, 1 = Yes)")
ax.set_ylabel("Log(Total Changes)")
save_show(fig, "plots/adoption_with_points.png")

# 5.11 Top languages by AI adoption
lang_plot = language_summary.sort_values("pct_ai_users", ascending=False).head(10)
lang_plot_usage = language_summary.sort_values("mean_ai_usage", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=lang_plot, x="pct_ai_users", y="language", ax=ax)
ax.set_title("Top Languages by AI Adoption Rate")
ax.set_xlabel("Proportion of AI Usage")
ax.set_ylabel("Language")
save_show(fig, "plots/top_languages_by_ai_adoption.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=lang_plot_usage, x="mean_ai_usage", y="language", ax=ax)
ax.set_title("Top Languages by Mean AI Usage Percentage")
ax.set_xlabel("Mean AI Usage Percentage")
ax.set_ylabel("Language")
save_show(fig, "plots/top_languages_by_mean_ai_usage.png")

# 5.12 All languages by AI adoption (appendix)
fig, ax = plt.subplots(figsize=(8, 10))
sns.barplot(data=language_summary, x="pct_ai_users", y="language", ax=ax)
ax.set_title("AI Adoption Rate by Language")
ax.set_xlabel("Proportion of AI Usage")
ax.set_ylabel("Language")
save_show(fig, "plots/all_languages_by_ai_adoption.png")

fig, ax = plt.subplots(figsize=(8, 10))
sns.barplot(data=language_summary, x="mean_ai_usage", y="language", ax=ax)
ax.set_title("Mean AI Usage percentage by Language")
ax.set_xlabel("Mean AI Usage percentage")
ax.set_ylabel("Language")
save_show(fig, "plots/all_languages_by_mean_ai_usage.png")

# # 5️⃣ Optional summary table
language_summary_sorted = language_summary.sort_values("mean_ai_usage", ascending=False)
print("Top languages by mean AI usage:\n", language_summary_sorted.head(10))

