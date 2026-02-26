PYTHON ?= python

.PHONY: setup ingest normalize index query eval

setup:
	uv sync

ingest:
	$(PYTHON) -m src.cli ingest arxiv --query "retrieval augmented generation" --max 5
	$(PYTHON) -m src.cli ingest rss --feeds sources/rss_feeds.txt
	$(PYTHON) -m src.cli ingest youtube --urls sources/youtube_urls.txt

normalize:
	$(PYTHON) -m src.cli normalize

index:
	$(PYTHON) -m src.cli index

query:
	$(PYTHON) -m src.cli query --q "$(Q)"

eval:
	$(PYTHON) -m src.cli eval --file eval/questions.json
