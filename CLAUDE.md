# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (uses uv if available, falls back to pip)
make setup
# or: uv sync --extra dev

# Ingestion
python -m src.cli ingest arxiv --query "retrieval augmented generation" --max 10
python -m src.cli ingest rss --feeds sources/rss_feeds.txt
python -m src.cli ingest youtube --urls sources/youtube_urls.txt

# Build pipeline
python -m src.cli normalize
python -m src.cli index

# Query
python -m src.cli query --q "What is RAG?" --k 5
make query Q="What is RAG?"

# Evaluation
python -m src.cli eval --file eval/questions.json

# Tests (offline, no API keys needed)
make test
python -m pytest -q
# Run a single test file:
.venv/bin/python -m pytest tests/test_bm25_tokenize.py -q

# Smoke tests
make smoke          # offline (SMOKE_EMBED=0)
make smoke_paid     # full end-to-end (incurs OpenAI cost)
SMOKE_QUERY=0 make smoke  # disable query step explicitly

# Linting
.venv/bin/python -m ruff check src/ tests/

# Clean artifacts
make clean      # removes processed/ and faiss index
make clean_all  # also removes embedding cache and BM25 index
```

## Architecture

The pipeline has five sequential stages:

1. **Ingestion** (`src/ingest/`) — Fetches raw content from sources and writes JSONL to `data/raw/`:
   - `arxiv.py`: arXiv API + PyMuPDF for PDF parsing
   - `rss.py`: feedparser + trafilatura for article extraction
   - `youtube.py`: yt-dlp for captions (falls back to Whisper if API key present)

2. **Normalization** (`src/ingest/normalize.py`) — Merges raw JSONL into a unified `data/processed/docs.jsonl` using a deterministic `doc_id` (URL hash). Append-only with duplicate protection.

3. **Indexing** (`src/index/`) — Reads `docs.jsonl`, chunks and embeds text, then writes two indexes:
   - `data/index/faiss.index` + `data/index/metadata.jsonl` (dense vector search)
   - `data/index/bm25.joblib` (BM25 lexical index)
   - Embedding is cached in `data/index/embeddings_cache.jsonl` to avoid re-embedding unchanged chunks

4. **Retrieval** (`src/rag/retrieve.py`) — `retrieve_chunks()` auto-selects mode:
   - **hybrid** (default when BM25 index exists): fetches expanded candidate sets from both dense and BM25, then fuses via Reciprocal Rank Fusion (RRF, k=60)
   - **dense**: FAISS L2 search only
   - Mode can be overridden with `mode=` argument

5. **Answer generation** (`src/rag/answer.py`, `src/rag/prompt.py`) — Passes retrieved chunks to `gpt-4o-mini` with a strict grounded-answer prompt. Citations in `[source_type:doc_id location]` format are embedded in answers; CLI output replaces internal IDs with source URLs.

## Key Configuration

All config lives in `src/config.py` as `AppConfig` (frozen dataclass). Reads from environment / `.env`:

| Env var | Default |
|---|---|
| `OPENAI_API_KEY` | required for indexing/querying |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` |
| `CHUNK_MAX_CHARS` | `1000` |
| `CHUNK_OVERLAP_CHARS` | `150` |
| `TOP_K` | `5` |

## Data Schemas

**`docs.jsonl`** — unified document with `doc_id`, `source_type` (`arxiv_pdf|youtube|rss_blog`), `title`, `source_uri`, `created_at`, and `segments[]` (each with `segment_id`, `text`, `metadata`).

**`chunks.jsonl`** — flat chunks with `chunk_id`, `doc_id`, `text`, and `metadata` (including `citation` in format `[arxiv:<doc_id> p.<page>]`, `[youtube:<doc_id> HH:MM:SS]`, or `[rss:<doc_id>]`).

## Embedder Details

`src/index/embed.py` implements retry logic with exponential backoff (up to 6 attempts, respecting server `Retry-After` hints) and a persistent embedding cache keyed by `chunk_id`. Cache is invalidated if the model or embedding dimension changes.

## Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`
- `ffmpeg` on PATH (required for YouTube audio extraction)
- OpenAI API key in `.env`
