import os
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

print("Step 3: Analyzing data and generating plots...")

# -------------------------------
# Setup
# -------------------------------
os.makedirs("plots", exist_ok=True)

# Load data
df = pd.read_csv("processed/3_analysis_ready_data.csv")

# -------------------------------
# Summary
# -------------------------------
print("Columns:", df.columns.tolist())
print("Summary statistics:\n", df.describe())

print("\nAI Group value count (developer-month level):")
print(df["ai_group"].value_counts())

print("\nUsed AI value count (developer-month level):")
print(df["used_ai"].value_counts())

# -------------------------------
# Developer-level summary
# -------------------------------
df_dev = (
    df.groupby("commit_author")
    .agg(
        median_monthly_commits=("commits", "median"),
        median_monthly_total_changes=("total_changes", "median"),
        mean_monthly_commits=("commits", "mean"),
        mean_monthly_total_changes=("total_changes", "mean"),
        mean_ai_usage=("ai_usage", "mean"),
        language=("language", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    )
    .reset_index()
)

df_dev["dev_ai_group"] = pd.cut(
    df_dev["mean_ai_usage"],
    bins=[-0.001, 0, 0.2, 1],
    labels=["none", "low", "high"],
    include_lowest=True
)

# -------------------------------
# Developer-level language summary
# -------------------------------
language_summary_dev = (
    df_dev[df_dev["language"] != "Unknown"]
    .groupby("language")
    .agg(
        n_developers=("commit_author", "count"),
        median_commits=("median_monthly_commits", "median"),
        median_total_changes=("median_monthly_total_changes", "median"),
        mean_commits=("mean_monthly_commits", "mean"),
        mean_total_changes=("mean_monthly_total_changes", "mean"),
        mean_ai_usage=("mean_ai_usage", "mean"),
    )
    .reset_index()
    .sort_values(by="n_developers", ascending=False)
)

print("\nDeveloper-level Language summary:")
print(language_summary_dev)

group_counts = df_dev["dev_ai_group"].value_counts().reindex(["none", "low", "high"])
group_props = (group_counts / len(df_dev)).round(4)

print("\nAI usage group counts (developer level):")
print(group_counts)

print("\nAI usage group proportions (developer level):")
print(group_props)

direction_median = (
    df_dev.groupby("dev_ai_group")[[
        "median_monthly_commits",
        "median_monthly_total_changes"
    ]]
    .median()
    .reindex(["none", "low", "high"])
)

print("\nMedian productivity metrics by AI usage group:")
print(direction_median)

direction_mean = (
    df_dev.groupby("dev_ai_group")[[
        "mean_monthly_commits",
        "mean_monthly_total_changes"
    ]]
    .mean()
    .reindex(["none", "low", "high"])
)

print("\nMean productivity metrics by AI usage group:")
print(direction_mean)

# -------------------------------
# Statistical tests
# -------------------------------
print("\n==============================")
print("KRUSKAL-WALLIS TEST")
print("==============================")

groups_commits = [
    df_dev[df_dev["dev_ai_group"] == "none"]["median_monthly_commits"],
    df_dev[df_dev["dev_ai_group"] == "low"]["median_monthly_commits"],
    df_dev[df_dev["dev_ai_group"] == "high"]["median_monthly_commits"],
]

stat_c, p_c = kruskal(*groups_commits)
print(f"Commits → H-stat: {stat_c:.4f}, p-value: {p_c:.6g}")

groups_changes = [
    df_dev[df_dev["dev_ai_group"] == "none"]["median_monthly_total_changes"],
    df_dev[df_dev["dev_ai_group"] == "low"]["median_monthly_total_changes"],
    df_dev[df_dev["dev_ai_group"] == "high"]["median_monthly_total_changes"],
]

stat_t, p_t = kruskal(*groups_changes)
print(f"Total Changes → H-stat: {stat_t:.4f}, p-value: {p_t:.6g}")

print("\n==============================")
print("PAIRWISE MANN-WHITNEY TESTS")
print("==============================")

def pairwise_test(data, col):
    pairs = [("none", "low"), ("none", "high"), ("low", "high")]

    for g1, g2 in pairs:
        group1 = data[data["dev_ai_group"] == g1][col]
        group2 = data[data["dev_ai_group"] == g2][col]
        stat, p = mannwhitneyu(group1, group2, alternative="two-sided")
        print(f"{g1} vs {g2} ({col}) → p-value: {p:.6g}")

pairwise_test(df_dev, "median_monthly_commits")
pairwise_test(df_dev, "median_monthly_total_changes")

# -------------------------------
# Plots
# -------------------------------

sns.set_theme(style="whitegrid")

# 1. Productivity by AI usage group (main figure)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(
    data=df_dev,
    x="dev_ai_group",
    y="median_monthly_commits",
    order=["none", "low", "high"],
    showfliers=False,
    ax=axes[0]
)
axes[0].set_title("Median Monthly Commits by AI Usage Group")
axes[0].set_xlabel("AI Usage Group")
axes[0].set_ylabel("Median Monthly Commits")

sns.boxplot(
    data=df_dev,
    x="dev_ai_group",
    y="median_monthly_total_changes",
    order=["none", "low", "high"],
    showfliers=False,
    ax=axes[1]
)
axes[1].set_title("Median Monthly Total Changes by AI Usage Group")
axes[1].set_xlabel("AI Usage Group")
axes[1].set_ylabel("Median Monthly Total Changes")

plt.tight_layout()
plt.savefig("plots/productivity_by_ai_group_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()


# 2. Group size distribution
fig, ax = plt.subplots(figsize=(7, 5))

group_counts.plot(kind="bar", ax=ax)
ax.set_title("Developer Counts by AI Usage Group")
ax.set_xlabel("AI Usage Group")
ax.set_ylabel("Number of Developers")
ax.set_xticklabels(["none", "low", "high"], rotation=0)

plt.tight_layout()
plt.savefig("plots/ai_group_counts.png", dpi=300, bbox_inches="tight")
plt.show()


# 3. Group proportion distribution
fig, ax = plt.subplots(figsize=(7, 5))

group_props.plot(kind="bar", ax=ax)
ax.set_title("Developer Proportions by AI Usage Group")
ax.set_xlabel("AI Usage Group")
ax.set_ylabel("Proportion of Developers")
ax.set_xticklabels(["none", "low", "high"], rotation=0)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig("plots/ai_group_proportions.png", dpi=300, bbox_inches="tight")
plt.show()


# 4. Top 10 languages by number of developers
top_langs = language_summary_dev.head(10)

fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(
    data=top_langs,
    x="n_developers",
    y="language",
    ax=ax
)
ax.set_title("Top 10 Languages by Number of Developers")
ax.set_xlabel("Number of Developers")
ax.set_ylabel("Language")

plt.tight_layout()
plt.savefig("plots/top10_languages_by_developers.png", dpi=300, bbox_inches="tight")
plt.show()


# 5. Mean AI usage by top 10 languages
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(
    data=top_langs.sort_values("mean_ai_usage", ascending=False),
    x="mean_ai_usage",
    y="language",
    ax=ax
)
ax.set_title("Mean AI Usage by Top 10 Languages")
ax.set_xlabel("Mean AI Usage")
ax.set_ylabel("Language")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout()
plt.savefig("plots/top10_languages_mean_ai_usage.png", dpi=300, bbox_inches="tight")
plt.show()