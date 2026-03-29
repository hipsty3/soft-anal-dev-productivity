file: load_dataset aggregate clean_data analysis

load_dataset:
	mkdir -p dataset
	unzip -q ai_commit_research_8.zip -d dataset_temp
	mv dataset_temp/ai_commit_research_8.db dataset/
	rm -rf dataset_temp

aggregate:
	mkdir -p processed
	uv run python -m scripts.1_aggregate

clean_data:
	uv run python -m scripts.2_clean_data

clean_data_100:
	uv run python -m scripts.2_clean_data_100_dev

analysis:
	mkdir -p plots
	uv run python -m scripts.3_analysis

analysis_100:
	mkdir -p plots_100_dev
	uv run python -m scripts.3_analysis_100_dev