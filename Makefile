PYTHON ?= .venv/bin/python

.PHONY: setup ingest normalize index query eval test smoke smoke_paid smoke_answer smoke_offline clean clean_all

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
	SMOKE_EMBED=0 $(PYTHON) scripts/smoke_test.py

smoke_paid:
	SMOKE_EMBED=1 $(PYTHON) scripts/smoke_test.py

smoke_answer:
	SMOKE_EMBED=0 SMOKE_QUERY=1 $(PYTHON) scripts/smoke_test.py

smoke_offline:
	SMOKE_EMBED=0 SMOKE_QUERY=0 $(PYTHON) scripts/smoke_test.py

clean:
	rm -f data/processed/*.jsonl
	rm -f data/processed/*.json
	rm -f data/index/*.index

clean_all:
	rm -f data/processed/*.jsonl
	rm -f data/processed/*.json
	rm -f data/index/*.index
	rm -f data/index/*.npy
	rm -f data/index/*.json
	rm -f data/index/*.jsonl
	rm -f data/index/embeddings_cache*
