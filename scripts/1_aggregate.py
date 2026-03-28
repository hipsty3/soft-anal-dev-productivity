import sqlite3
import os
import pandas as pd

print("Step 1: Aggregating data from SQLite database...")

conn = sqlite3.connect("dataset/ai_commit_research_8.db")

query = """
SELECT
    commit_author,
    MAX(language) AS language,
    year,
    month,
    COUNT(*) AS commits,
    SUM(ai_flag) AS ai_commits,
    SUM(additions) AS lines_added,
    SUM(deletions) AS lines_removed,
    SUM(total_changes) AS total_changes,
    CAST(SUM(ai_flag) AS FLOAT) / COUNT(*) AS ai_usage
FROM commits
WHERE total_changes > 0
GROUP BY commit_author, year, month
"""

df = pd.read_sql_query(query, conn)

df.to_csv("processed/1_aggregated_data.csv", index=False)

print(df.head())