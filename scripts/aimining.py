import requests
import sqlite3
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

repo_counter_lock = Lock()
token_lock = Lock()

VALID_REPO_TARGET = 700
valid_repo_count = 0


TOKENS = [
    # add tokens here
]

MAX_REPOS = 6000
MIN_PRS = 100
MIN_AI_PRS = 5

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 3, 8)
DB_NAME = "ai_commit_research.db"


token_index = 0

def get_headers():
    global token_index
    with token_lock:
        return {
            "Authorization": f"Bearer {TOKENS[token_index]}",
            "Content-Type": "application/json"
        }

def rotate_token():
    global token_index
    with token_lock:
        token_index = (token_index + 1) % len(TOKENS)
        print(f"Rotated to token {token_index}")


conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS commits (
    repo TEXT,
    language TEXT,
    pr_number INTEGER,
    commit_author TEXT,
    ai_flag INTEGER,
    year INTEGER,
    month INTEGER,
    additions INTEGER,
    deletions INTEGER,
    total_changes INTEGER,
    is_org INTEGER
)
""")
conn.commit()


def get_popular_repos():
    repos = []
    
    star_ranges = [
        (3000,4000),
        (4001, 6000),
        (6001, 10000),
        (10001, 20000),
        (20001, 50000),
        (50001, 999999)
    ]

    for min_star, max_star in star_ranges:
        page = 1

        while len(repos) < MAX_REPOS:
            r = requests.get(
                "https://api.github.com/search/repositories",
                headers={"Authorization": f"Bearer {TOKENS[0]}"},
                params={
                    "q": f"stars:{min_star}..{max_star} archived:false",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page
                }
            )

            data = r.json()
            if "items" not in data or not data["items"]:
                break

            repos.extend(data["items"])
            page += 1

            if page > 10: 
                break

            time.sleep(0.3)

        if len(repos) >= MAX_REPOS:
            break

    return repos[:MAX_REPOS]

query = """
query($owner: String!, $name: String!, $cursor: String) {
  rateLimit {
    cost
    remaining
    resetAt
  }
  repository(owner: $owner, name: $name) {
    primaryLanguage {
      name
    }
    pullRequests(
      first: 50
      after: $cursor
      states: MERGED
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        createdAt
        author {
          __typename
          ... on User {
            login
          }
        }
        commits(first: 20) {
          nodes {
            commit {
              message
              additions
              deletions
              author {
                user {
                  login
                }
                name
              }
            }
          }
        }
      }
    }
  }
}
"""

def safe_graphql_request(variables, max_retries=5):
    global token_index

    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": variables},
                headers=get_headers(),
                timeout=20
            )

            if r.status_code in [401, 403]:
                print(f"Token {token_index} exhausted or forbidden.")
                rotate_token()
                continue

            if r.status_code >= 500:
                print("GitHub server error. Retrying...")
                time.sleep(3)
                continue

            if not r.text.strip():
                print("Empty response. Rotating token.")
                rotate_token()
                continue

            data = r.json()

            if "errors" in data:
                print("GraphQL error:", data["errors"])
                rotate_token()
                continue

            time.sleep(0.1)    
            return data

        except requests.exceptions.JSONDecodeError:
            print("JSON decode failed. Rotating token.")
            rotate_token()

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}. Retrying...")
            time.sleep(3)

    print("Max retries hit. Skipping this repo.")
    return None

def mine_repo(owner, repo, is_org):
    global valid_repo_count

    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()

    full_name = f"{owner}/{repo}"
 
    cursor_val = None
    pr_count = 0
    ai_pr_count = 0
    commit_rows = []

    while True:

        variables = {
            "owner": owner,
            "name": repo,
            "cursor": cursor_val
        }

        data = safe_graphql_request(variables)

        if data is None:
            conn.close()
            return
        
        repo_data = data["data"]["repository"]
        pr_block = repo_data["pullRequests"]
        cursor_val = pr_block["pageInfo"]["endCursor"]

        primary_lang = repo_data["primaryLanguage"]
        language = primary_lang["name"] if primary_lang else "Unknown"

        for pr in pr_block["nodes"]:

            created = datetime.fromisoformat(
                pr["createdAt"].replace("Z", "+00:00")
            )

            
            if created.timestamp() > END_DATE.timestamp():
                continue

            if created.timestamp() < START_DATE.timestamp():
                break

            if pr["author"] is None or pr["author"]["__typename"] != "User":
                continue

            pr_ai_flag = 0

            for c in pr["commits"]["nodes"]:
                commit_data = c["commit"]
                msg = commit_data["message"].lower()

                additions = commit_data.get("additions") or 0
                deletions = commit_data.get("deletions") or 0
                total_changes = additions + deletions

                author_info = commit_data.get("author")
                if author_info and author_info.get("user"):
                    commit_author = author_info["user"]["login"]
                else:
                    commit_author = author_info.get("name") if author_info else "unknown"

                commit_ai_flag = 0
                if "co-authored-by:" in msg and (
                    "copilot" in msg or "claude" in msg
                ):
                    commit_ai_flag = 1
                    pr_ai_flag = 1

                commit_rows.append((
                    full_name,
                    language,
                    pr["number"],
                    commit_author,
                    commit_ai_flag,
                    created.year,
                    created.month,
                    additions,
                    deletions,
                    total_changes,
                    is_org
                ))

            if pr_ai_flag:
                ai_pr_count += 1

            pr_count += 1

        if not pr_block["pageInfo"]["hasNextPage"]:
            break

    if (
        pr_count >= MIN_PRS and
        ai_pr_count >= MIN_AI_PRS
    ):
        cursor.executemany("""
            INSERT INTO commits VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, commit_rows)

        conn.commit()

        with repo_counter_lock:
            valid_repo_count += 1
            print(f"Kept {owner}/{repo} | PRs: {pr_count} | AI: {ai_pr_count}")

    conn.close()

def process_repo(repo):
    owner = repo["owner"]["login"]
    name = repo["name"]
    is_org = 1 if repo["owner"].get("type") == "Organization" else 0
    mine_repo(owner, name, is_org)

def main():
    repos = get_popular_repos()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_repo, repo) for repo in repos]

        for _ in tqdm(as_completed(futures), total=len(futures)):
            if valid_repo_count >= VALID_REPO_TARGET:
                break

    print("done")

if __name__ == "__main__":
    main()
