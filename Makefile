PYTHON ?= .venv/bin/python

.PHONY: setup ingest normalize index query eval test smoke clean

setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

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

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke_test.py

clean:
	rm -rf data/processed/*
	rm -rf data/index/*
