# multimodal-rag-ingest

API-first multimodal RAG pipeline that ingests arXiv papers, YouTube talks, and RSS posts into a unified knowledge base for hybrid retrieval and grounded, citation-backed answers.

## Features

### Ingestion
- arXiv ingestion via API + PDF parsing
- YouTube ingestion via `yt-dlp` captions, with Whisper fallback transcription when captions are unavailable and `OPENAI_API_KEY` is set
- RSS ingestion via feed parsing + article extraction
- Deterministic `doc_id` generation from source URI hash
- Idempotent ingestion with duplicate protection in source JSONL files

### Retrieval
- OpenAI embeddings for dense retrieval
- FAISS dense vector index
- BM25 lexical index
- Hybrid retrieval (Dense FAISS + BM25) with dense fallback when BM25 artifacts are missing
- Grounded answers generated strictly from retrieved context with validated citations

### Evaluation
- Retrieval evaluation metrics: Recall@K and MRR

## Architecture

```text
Sources
  |
  v
Ingestion
  |
  v
Normalization (docs.jsonl)
  |
  v
Chunking + Metadata Enrichment (chunks.jsonl)
  |
  v
Embeddings + BM25 Artifacts
  |
  v
Indexes (FAISS / BM25)
  |
  v
Retrieval (Dense / BM25 / Hybrid)
  |
  v
Grounded Answer + Citations
```

## Quick Start

Prerequisites:
- Python 3.11+
- `uv` (recommended) or `pip`
- `ffmpeg` on `PATH` (required for YouTube audio extraction/transcription fallback)
- OpenAI API key for embedding/indexing and LLM answering

Setup:

```bash
make setup
cp .env.example .env
# set OPENAI_API_KEY=...
```

End-to-end pipeline:

```bash
python -m src.cli ingest arxiv --query "retrieval augmented generation" --max 5
python -m src.cli ingest rss --feeds sources/rss_feeds.txt
python -m src.cli ingest youtube --urls sources/youtube_urls.txt
python -m src.cli normalize
python -m src.cli index
python -m src.cli query --q "What is retrieval augmented generation?" --k 5
python -m src.cli eval --file eval/questions.json --k 5
```

Primary artifacts:
- `data/processed/docs.jsonl`
- `data/processed/chunks.jsonl`
- `data/index/faiss.index`
- `data/index/metadata.jsonl`
- `data/index/bm25.joblib`

## CLI Commands

### Ingest
```bash
python -m src.cli ingest arxiv --query "retrieval augmented generation" --max 10
python -m src.cli ingest rss --feeds sources/rss_feeds.txt
python -m src.cli ingest youtube --urls sources/youtube_urls.txt
```

### Process and Index

```bash
python -m src.cli normalize
python -m src.cli index
```

### Query and Evaluate
```bash
python -m src.cli query --q "What is RAG?" --k 5
python -m src.cli eval --file eval/questions.json --k 5
```

## Evaluation

Run:

```bash
python -m src.cli eval --file eval/questions.json --k 5
```

Outputs:
- mode-by-mode table for `dense`, `bm25`, and `hybrid` (when BM25 index exists)
- `Recall@1`, `Recall@5`, `Recall@10`, `MRR`
- detailed JSON report at `data/eval/results.json`

Example metrics output:

| mode   | Recall@1 | Recall@5 | Recall@10 | MRR  |
|--------|----------|----------|-----------|------|
| dense  | 0.40     | 0.72     | 0.84      | 0.56 |
| bm25   | 0.35     | 0.68     | 0.80      | 0.49 |
| hybrid | 0.46     | 0.79     | 0.88      | 0.61 |

## Testing

1. Unit tests
```bash
make test
```

2. Smoke test
```bash
make smoke
```

3. Full paid smoke test
```bash
make smoke_paid
```

Smoke behavior notes:
- `make smoke` always runs normalization and validates `data/processed/docs.jsonl`.
- By default, `make smoke` attempts query (`SMOKE_QUERY=1`) only if `data/index/faiss.index` already exists.
- Query execution can call OpenAI (embedding for retrieval and chat completion for answers), so smoke may call OpenAI depending on flags and existing artifacts.
- If index is missing, `make smoke` skips query to avoid costs.
- `make smoke_paid` forces indexing + query and will incur OpenAI usage.

## Data Schemas

### `docs.jsonl`

```json
{
  "doc_id": "string",
  "source_type": "arxiv_pdf|youtube|rss_blog",
  "title": "string",
  "source_uri": "string",
  "created_at": "ISO8601",
  "segments": [
    {
      "segment_id": "string",
      "text": "string",
      "metadata": {
        "page": 3,
        "timestamp_start": 12.34,
        "timestamp_end": 18.21,
        "url": "string",
        "published_at": "ISO8601"
      }
    }
  ]
}
```

### `chunks.jsonl`

```json
{
  "chunk_id": "string",
  "doc_id": "string",
  "text": "string",
  "metadata": {
    "source_type": "arxiv_pdf|youtube|rss_blog",
    "title": "string",
    "source_uri": "string",
    "citation": "[arxiv:<doc_id> p.<page>] | [youtube:<doc_id> HH:MM:SS] | [rss:<doc_id>]"
  }
}
```

### `eval/questions.json`

```json
[
  {
    "question": "What is retrieval augmented generation?",
    "relevant_doc_ids": ["abc123def4567890"]
  }
]
```

## Repository Layout

```text
multimodal-rag-ingest/
├── src/
│   ├── cli.py
│   ├── ingest/
│   ├── index/
│   ├── rag/
│   └── eval/
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
├── sources/
│   ├── rss_feeds.txt
│   └── youtube_urls.txt
├── eval/
│   └── questions.json
├── scripts/
│   └── smoke_test.py
├── tests/
├── Makefile
└── README.md
```

## Roadmap

- Cross-encoder reranking
- Incremental indexing
- Faithfulness scoring
- Web UI

## Why this project

This project explores production-style multimodal retrieval over heterogeneous sources (papers, videos, and blogs). It focuses on:
- reproducible ingestion and normalization
- hybrid retrieval with dense and lexical search
- grounded answers with explicit citations
- measurable retrieval quality via Recall@K and MRR

## License

MIT
