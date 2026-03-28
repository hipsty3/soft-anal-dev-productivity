import os
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scripts.utils import (
    interpret_coef,
    save_show,
    paired_boxplot,
    paired_boxplot_with_means,
    percent_axis_x,
    percent_axis_y,
)

# -------------------------------
# Setup
# -------------------------------
os.makedirs("plots", exist_ok=True)

# Clean default style WITH gridlines
sns.set_theme(style="whitegrid")

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
df_lang = df[df["language"] != "Unknown"]

language_summary = (
    df_lang.groupby("language")
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

# 5.1 Productivity by AI group with means

# log scale with means
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

# 5.2 Distribution of AI usage
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["ai_usage"], bins=50, ax=ax)
ax.set_title("Distribution of AI Usage")
ax.set_xlabel("AI Usage")
ax.set_ylabel("Count")
percent_axis_x(ax)
save_show(fig, "plots/ai_usage_distribution.png")

# Ai users only
fig, ax = plt.subplots(figsize=(8, 4))
data = df[df["used_ai"] == 1]["ai_usage"]
sns.histplot(data, bins=50, ax=ax)
ax.set_title("Distribution of AI Usage (Among AI Users)")
ax.set_xlabel("AI Usage")
ax.set_ylabel("Count")
percent_axis_x(ax)
save_show(fig, "plots/ai_usage_distribution_nonzero.png")

# 5.3 Productivity by AI adoption

# log scale
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

# 5.4 AI adoption over time
time_trend = df.groupby(["year", "month"])["used_ai"].mean().reset_index()
time_trend["date"] = time_trend["year"].astype(str) + "-" + time_trend["month"].astype(str)

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=time_trend, x="date", y="used_ai", ax=ax)
ax.set_title("AI Adoption Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Proportion Using AI")
percent_axis_y(ax)
plt.xticks(rotation=45)
save_show(fig, "plots/ai_adoption_over_time.png")


###################
# LANGUAGE ANALYSIS
###################

# Top 10 languages by frequency
top_langs = (
    df_lang["language"]
    .value_counts()
    .head(10)
    .index
)

df_lang_top = df_lang[df_lang["language"].isin(top_langs)]

language_summary_top = language_summary[
    language_summary["language"].isin(top_langs)
]

# 1. Mean AI usage by language
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=language_summary,
    x="mean_ai_usage",
    y="language",
    ax=ax,
    order=language_summary.sort_values("mean_ai_usage", ascending=False)["language"]
)
ax.set_title("Mean AI Usage Percentage by Language")
ax.set_xlabel("Mean AI Usage Percentage")
ax.set_ylabel("Language")
percent_axis_x(ax)
save_show(fig, "plots/mean_ai_usage_by_language.png")

# Mean AI Usage top language
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=language_summary_top,
    x="mean_ai_usage",
    y="language",
    ax=ax,
    order=language_summary_top.sort_values("mean_ai_usage", ascending=False)["language"]
)
ax.set_title("Mean AI Usage Percentage by Language (Top 10)")
ax.set_xlabel("Mean AI Usage Percentage")
ax.set_ylabel("Language")
percent_axis_x(ax)
save_show(fig, "plots/top10_mean_ai_usage_by_language_top10.png")

# 2. AI group distribution within each language
lang_ai_dist = (
    df_lang.groupby(["language", "ai_group"])
    .size()
    .reset_index(name="count")
)

lang_ai_dist["pct"] = lang_ai_dist.groupby("language")["count"].transform(lambda x: x / x.sum())

pivot_df = lang_ai_dist.pivot(index="language", columns="ai_group", values="pct").fillna(0)
pivot_df["ai_total"] = pivot_df.get("low", 0) + pivot_df.get("high", 0)
pivot_df = pivot_df.sort_values("ai_total", ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
pivot_df.drop(columns="ai_total").plot(kind="barh", stacked=True, ax=ax)

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("AI Usage Distribution by Language")
ax.set_xlabel("Proportion")
ax.set_ylabel("Language")

save_show(fig, "plots/ai_group_distribution_by_language.png")

# AI group distribution within each language (top 10)
lang_ai_dist = (
    df_lang_top.groupby(["language", "ai_group"])
    .size()
    .reset_index(name="count")
)

lang_ai_dist["pct"] = lang_ai_dist.groupby("language")["count"].transform(lambda x: x / x.sum())

pivot_df = lang_ai_dist.pivot(index="language", columns="ai_group", values="pct").fillna(0)
pivot_df["ai_total"] = pivot_df.get("low", 0) + pivot_df.get("high", 0)
pivot_df = pivot_df.sort_values("ai_total", ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
pivot_df.drop(columns="ai_total").plot(kind="barh", stacked=True, ax=ax)

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("AI Usage Distribution by Language (Top 10)")
ax.set_xlabel("Proportion")
ax.set_ylabel("Language")

save_show(fig, "plots/ai_group_distribution_by_language_top10.png")

# 3. Language distribution within each AI group
group_lang_dist = (
    df_lang.groupby(["ai_group", "language"])
    .size()
    .reset_index(name="count")
)

group_lang_dist["pct"] = group_lang_dist.groupby("ai_group")["count"].transform(lambda x: x / x.sum())

g = sns.catplot(
    data=group_lang_dist,
    x="pct",
    y="language",
    col="ai_group",
    kind="bar",
    sharex=False,
    height=5,
    aspect=0.8
)

for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Language")

g.fig.suptitle("Language Distribution Within Each AI Usage Group", y=1.02)

plt.tight_layout()
plt.savefig("plots/language_distribution_by_ai_group.png", dpi=300)
plt.show()

# top 10
group_lang_dist_top = (
    df_lang_top.groupby(["ai_group", "language"])
    .size()
    .reset_index(name="count")
)
group_lang_dist_top["pct"] = group_lang_dist_top.groupby("ai_group")["count"].transform(lambda x: x / x.sum())
g = sns.catplot(
    data=group_lang_dist_top,
    x="pct",
    y="language",
    col="ai_group",
    kind="bar",
    sharex=False,
    height=5,
    aspect=0.8
)
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Language")
g.fig.suptitle("Language Distribution Within Each AI Usage Group (Top 10)", y=1.02)
plt.tight_layout()
plt.savefig("plots/language_distribution_by_ai_group_top10.png", dpi=300)
plt.show()