import pandas as pd
import numpy as np

df = pd.read_csv("processed/1_aggregated_data.csv")

# Ensure numeric columns first
numeric_columns = ["commits", "ai_commits", "lines_added", "lines_removed", "total_changes", "ai_usage"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remove missing critical values
critical_columns = ["commit_author", "year", "month", "total_changes", "commits", "ai_usage", "ai_commits"]
df = df.dropna(subset=critical_columns)

# Remove empty authors
df["commit_author"] = df["commit_author"].astype(str).str.strip()
df = df[df["commit_author"] != ""]

# Recalculate ai_usage to ensure consistency
df["ai_usage"] = df["ai_commits"] / df["commits"]

# Remove invalid rows
df = df[df["total_changes"] > 0]
df = df[df["commits"] > 0]
df = df[(df["ai_usage"] >= 0) & (df["ai_usage"] <= 1)]

# Remove negative values
for col in numeric_columns:
    df = df[df[col] >= 0]

# Fix language
df["language"] = df["language"].fillna("Unknown").astype(str).str.strip()
df.loc[df["language"] == "", "language"] = "Unknown"

# Log transforms
df["log_commits"] = np.log1p(df["commits"])
df["log_lines_added"] = np.log1p(df["lines_added"])
df["log_lines_removed"] = np.log1p(df["lines_removed"])
df["log_total_changes"] = np.log1p(df["total_changes"])

# AI usage categories
df["ai_group"] = pd.cut(
    df["ai_usage"],
    bins=[0, 0.01, 0.50, 1],
    labels=["none", "medium", "high",],
    include_lowest=True  # <- include 0 in first bin
)

# AI usage flag
df["used_ai"] = (df["ai_usage"] > 0).astype(int)

print(f"AI Group value count: {df['ai_group'].value_counts()}")
print(f"Used AI value count: {df['used_ai'].value_counts()}")

df["dev_month_id"] = (
    df["commit_author"] + "_" +
    df["year"].astype(str) + "_" +
    df["month"].astype(str)
)

# Save
df.to_csv("processed/2_cleaned_data.csv", index=False)

print("----------------------------------------\n")
print("Final rows:", len(df))
print(df.describe())

# Prepare analysis-ready data
analysis_df = df[[
    "commit_author", "year", "month", "language",
    "commits", "ai_usage", "ai_group", "used_ai", "total_changes",
    "log_commits", "log_total_changes"
]]

analysis_df.to_csv("processed/3_analysis_ready_data.csv", index=False)

print("----------------------------------------")
print("Analysis final rows:", len(analysis_df))
print(analysis_df.describe())

# Save data for bins
bins_df = df[["commit_author", "year", "month", "language", "ai_group", "commits", "total_changes", "log_commits", "log_total_changes"]]
bins_df.to_csv("processed/3.1_analysis_bins_data.csv", index=False)
print("----------------------------------------")
print("Bins final rows:", len(bins_df))
print(bins_df.describe())   