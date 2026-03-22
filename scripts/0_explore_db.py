import sqlite3

conn = sqlite3.connect("dataset/ai_commit_research_8.db")

cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# See table schema
cursor.execute("PRAGMA table_info('commits');")
print(cursor.fetchall())


# See table data (5 rows)
cursor.execute("SELECT * FROM commits LIMIT 5;")
print(cursor.fetchall())

# Count total number of rows in the commits table
cursor.execute("SELECT COUNT(*) FROM commits;")
print(cursor.fetchone())

# Nicer output
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

rows = cursor.execute("SELECT * FROM commits LIMIT 5").fetchall()
for row in rows:
    print(dict(row))