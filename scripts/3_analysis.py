import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm  
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("processed/3_analysis_ready_data.csv")

# -------------------------------
# 1. Descriptive statistics
# -------------------------------
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Summary statistics:\n", df.describe())

print(f"AI Group value count: {df['ai_group'].value_counts()}")
print(f"Used AI value count: {df['used_ai'].value_counts()}")

# Aggregate by language
language_summary = df.groupby("language").agg(
    n_months=("commit_author", "count"),
    mean_commits=("commits", "mean"),
    median_commits=("commits", "median"),
    mean_total_changes=("total_changes", "mean"),
    median_total_changes=("total_changes", "median"),
    mean_ai_usage=("ai_usage", "mean"),
    ai_users=("used_ai", "sum")
).reset_index()

# Optional: sort by number of months
language_summary = language_summary.sort_values(by="n_months", ascending=False)

print(language_summary)

# -------------------------------
# 2. Correlation
# -------------------------------

# Correlation matrix
print(df[["ai_usage", "commits", "total_changes"]].corr())

# Pearson correlation
corr, p_value = pearsonr(df["ai_usage"], df["commits"])
corr_changes, p_value_changes = pearsonr(df["ai_usage"], df["total_changes"])
print(f"Pearson correlation between AI usage and commits: {corr:.4f} (p-value: {p_value:.4e})") 
print(f"Pearson correlation between AI usage and total changes: {corr_changes:.4f} (p-value: {p_value_changes:.4e})")

# -------------------------------
# 3. Regression analysis with log transforms
# -------------------------------

# Regression with used_ai percentage
model_ai_use_commits = smf.ols("log_commits ~ used_ai", data=df).fit()
print(model_ai_use_commits.summary())

model_ai_use_volume = smf.ols("log_total_changes ~ used_ai", data=df).fit()  
print(model_ai_use_volume.summary())  

model_ai_percentage_commits = smf.ols("log_commits ~ ai_usage", data=df).fit()
print(model_ai_percentage_commits.summary())  

model_ai_percentage_volume = smf.ols("log_total_changes ~ ai_usage", data=df).fit()
print(model_ai_percentage_volume.summary())

# # Seperate regression by language (all languages)
# results_ai_use_commits = {}
# results_ai_use_volume = {}
# results_ai_percentage_commits = {}
# results_ai_percentage_volume = {}
# for lang, subdf in df.groupby("language"):
#     # if len(subdf) < 50:  # skip very small samples
#     #     continue
#     print(f"--- {lang} ---")
#     lang_model_ai_use_commits = smf.ols("log_commits ~ used_ai", data=subdf).fit()
#     results_ai_use_commits[lang] = lang_model_ai_use_commits
#     print(lang_model_ai_use_commits.summary())

#     lang_model_ai_use_volume = smf.ols("log_total_changes ~ used_ai", data=subdf).fit()
#     results_ai_use_volume[lang] = lang_model_ai_use_volume
#     print(lang_model_ai_use_volume.summary())
    
#     lang_model_ai_percentage_commits = smf.ols("log_commits ~ ai_usage", data=subdf).fit()
#     results_ai_percentage_commits[lang] = lang_model_ai_percentage_commits
#     print(lang_model_ai_percentage_commits.summary())
    
#     lang_model_ai_percentage_volume = smf.ols("log_total_changes ~ ai_usage", data=subdf).fit()
#     results_ai_percentage_volume[lang] = lang_model_ai_percentage_volume
#     print(lang_model_ai_percentage_volume.summary())
    
# # -------------------------------
# # 4. Visualization and summary
# # -------------------------------

# # 1️⃣ Distribution of AI usage
# plt.figure(figsize=(8,4))
# sns.histplot(df["ai_usage"], bins=50, kde=False)
# plt.title("Distribution of AI Usage")
# plt.xlabel("AI Usage (%)")
# plt.ylabel("Count of Dev-Months")
# plt.tight_layout()
# plt.show()

# # 2️⃣ Productivity by AI group
# plt.figure(figsize=(8,4))
# sns.boxplot(x="ai_group", y="log_total_changes", data=df, order=["none","low","medium","high","very_high"])
# plt.title("Productivity (log total changes) by AI Usage Group")
# plt.xlabel("AI Usage Group")
# plt.ylabel("Log(Total Changes)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,4))
# sns.boxplot(x="ai_group", y="log_commits", data=df, order=["none","low","medium","high","very_high"])
# plt.title("Productivity (log commits) by AI Usage Group")
# plt.xlabel("AI Usage Group")
# plt.ylabel("Log(Commits)")
# plt.tight_layout()
# plt.show()

# # 3️⃣ Correlation heatmap
# plt.figure(figsize=(6,4))
# corr_matrix = df[["ai_usage","commits","total_changes"]].corr()
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.tight_layout()
# plt.show()

# # 4️⃣ AI effect per language (regression coefficient)
# coef_df = pd.DataFrame({
#     "language": list(results_ai_use_volume.keys()),
#     "coef": [res.params["used_ai"] for res in results_ai_use_volume.values()]
# })
# coef_df = coef_df.sort_values("coef", ascending=False)

# plt.figure(figsize=(10,6))
# sns.barplot(x="coef", y="language", data=coef_df, palette="viridis")
# plt.title("Estimated Effect of AI Usage on Productivity (log total changes) by Language")
# plt.xlabel("Regression Coefficient (used_ai)")
# plt.ylabel("Language")
# plt.tight_layout()
# plt.show()

# # 5️⃣ Optional summary table
# language_summary_sorted = language_summary.sort_values("mean_ai_usage", ascending=False)
# print("Top languages by mean AI usage:\n", language_summary_sorted.head(10))

