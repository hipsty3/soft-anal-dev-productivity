file: load_dataset aggregate clean_data analysis analysis_bins

load_dataset:
	mkdir dataset
	unzip -q ai_commit_research_8.zip -d dataset_temp
	mv dataset_temp/ai_commit_research_8.db dataset/
	rm -rf dataset_temp

aggregate:
	mkdir processed
	uv run python -m scripts.1_aggregate

clean_data:
	uv run python -m scripts.2_clean_data

analysis:
	uv run python -m scripts.3_analysis

analysis_bins:
	uv run python -m scripts.3_analysis_bins