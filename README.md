# README
This repository provides the data mining and data processing code and resources for a software analytics research project on the effect of AI usage on Developer Productivity. This project is part of an assessment for a Masters in Computing Science degree subject on Software Analytics in Rijkuniversiteit Gronigen.


## Developer Productivity Metrics
We defined developer productivity with two metrics:
1. **Commit Frequency:** number of commits by a developer per month.
2. **Commit Volume:** total number of lines added and deleted in a commit by a developer within a month.

## Data mining script
Data mining script can be found in `scripts > aimining.py`. The dataset mined must fulfill these requirements:
- Must be open-sourced projects
- Repository must be popular (minimum 100 Pull requests)
- Commit data must be mined starting from 1 January 2025, which we believe AI adoption has started
- Developers of the commits must be active contributors (minimum made 100 commits)

## Dataset

**Rows:** represent a single commit by a developer, with relevant information for processing as columns.

**Columns:**  
- `repo`: Name of the repo where commit is found
- `language`: Programming language of the commit
- `pr_number`: Pull request number
- `commit_author`: Git username of commit author
- `ai_flag`: AI usage flag (0=no AI, 1=uses AI)
- `year`: Year commit was made
- `month`: Month commit was made
- `additions`: Number of lines added in a commit
- `deletions`: Number of lines deleted in a commit
- `total_changes`: Total number of change
- `is_org`: Indicates whether the repo is personal or part of an organisation. Redundant and unused for this project.

## Data processing instructions
In the `Makefile`, several commands were created for each step of the process. Write `make` before each command in the command line on the root folder (e.g. `make load_data`). 

Alternatively, to run all the commands and complete all the data processing pipeline at once, run `make file`.

1. `load_data`: Unzips and unloads the dataset zip file and saves it in a created dataset folder.
2. `aggregate`: Aggregrates and transforms the singular commit data per row into developer-month data per row, summing the total number of commits of that developer per month to get **commit frequnecy**, as well as summing the total changes to get **commit volume** of that developer in a month. `ai_usage` per development month data is 
3. `clean_data`: Removes missing and invalid values. Prepares the data for analysis. Divides the ai usage data into 3 categories, `none` (0 AI Usage), `low` (less that 20% AI Usage) and `high` (more than 20% AI Usage).
4. `analysis`: Prints descriptive statistics of the data. Takes the median of commit frequency and commit volume for Kruskal-Wallis and Pairwise Mann-Whitney statistical tests the three different AI usage group. Plots are generated from the median data for data interpretation.

*Credits: Ábel Nátán Dankó, Alexandru Pennert, Raisa Amalia (2026)*
