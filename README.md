# Multimodal RAG Ingest

API-first multimodal Retrieval-Augmented Generation (RAG) system that ingests:

- arXiv research papers (PDF)
- YouTube talks (video/audio)
- RSS blog posts (HTML)

and enables unified semantic search + grounded Q&A with citations.

---

## ✨ Features

- 📄 arXiv ingestion via API + PDF parsing
- 🎥 YouTube ingestion via yt-dlp + transcription
- 📰 RSS blog ingestion via feed parsing + content extraction
- 🧠 OpenAI embeddings
- 🔎 FAISS vector index
- 📚 Grounded Q&A with source citations
- 📊 Retrieval evaluation (Recall@K, MRR)

---

## 🏗 Architecture

Sources
  ↓
Ingestion (arXiv / YouTube / RSS)
  ↓
Normalization (docs.jsonl)
  ↓
Chunking (chunks.jsonl)
  ↓
Embeddings (OpenAI)
  ↓
FAISS Index
  ↓
Retrieval + RAG Answer
  ↓
Cited Output

---

## 🚀 Setup

### 1. Clone

```bash
git clone https://github.com/<your-username>/multimodal-rag-ingest.git
cd multimodal-rag-ingest
```

### 2. Install dependencies

Using uv:

```bash
make setup
```

### 3. Configure environment

```bash
cp .env.example .env
```

Add your OpenAI key:

```
OPENAI_API_KEY=your_key_here
```

Ensure `ffmpeg` is installed for YouTube/audio processing.

---

## 📥 Ingest Data

### arXiv

```bash
python -m src.cli ingest arxiv --query "retrieval augmented generation" --max 10
```

### RSS Blogs

Add feeds to `sources/rss_feeds.txt`, then:

```bash
python -m src.cli ingest rss --feeds sources/rss_feeds.txt
```

### YouTube

Add video URLs to `sources/youtube_urls.txt`, then:

```bash
python -m src.cli ingest youtube --urls sources/youtube_urls.txt
```

---

## 🔄 Normalize & Index

```bash
make normalize
make index
```

Artifacts:

- `data/processed/docs.jsonl`
- `data/processed/chunks.jsonl`
- `data/index/faiss.index`

---

## 🔍 Query

```bash
make query Q="What is retrieval augmented generation?"
```

Example output:

```
Answer:
RAG combines retrieval with generation by conditioning an LLM on retrieved context.

Sources:
[arxiv:paper123 p.4]
[youtube:talk45 00:12:10]
[rss:blog_post_abc]
```

---

## 📊 Evaluation

Add queries to `eval/questions.json`:

```json
[
  {
    "q": "What is dense retrieval?",
    "expected_doc_ids": ["doc_123"]
  }
]
```

Run:

```bash
make eval
```

Outputs:

- Recall@K
- MRR
- Retrieval latency

---

## 🧱 Data Schema

### docs.jsonl

Unified document representation across modalities.

### chunks.jsonl

Chunk-level representation used for embedding and indexing.

---

## 🔮 Roadmap

- Hybrid search (BM25 + dense)
- Reranking
- Personalization layer
- Slide deck ingestion (.pptx)
- Web UI with drag-and-drop
- Faithfulness scoring

---

## 📜 License

MIT
# multimodal-rag-ingest

API-first multimodal RAG pipeline for ingesting arXiv papers, YouTube talks, and RSS blog posts into a unified searchable knowledge base.

## Architecture

```text
                +------------------+
                |   Source APIs    |
                |------------------|
                | arXiv / YouTube  |
                | RSS + Articles   |
                +---------+--------+
                          |
                          v
                +------------------+
                | Ingestion Layer  |
                |------------------|
                | arxiv.py         |
                | youtube.py       |
                | rss.py           |
                +---------+--------+
                          |
                          v
                +------------------+
                | Normalization    |
                |------------------|
                | docs.jsonl       |
                +---------+--------+
                          |
                          v
                +------------------+
                | Chunk + Embed    |
                |------------------|
                | chunks.jsonl      |
                | OpenAI embeddings |
                +---------+--------+
                          |
                          v
                +------------------+
                | FAISS Vector DB  |
                |------------------|
                | faiss.index       |
                | metadata.jsonl    |
                +---------+--------+
                          |
                          v
                +------------------+
                | Retrieval + RAG  |
                |------------------|
                | grounded answers |
                | with citations   |
                +------------------+
```

## Features

- Deterministic `doc_id` from URL hash
- Idempotent ingestion (`*.jsonl` append-only with duplicate protection)
- Unified multimodal schema (`docs.jsonl`)
- Semantic chunking with metadata-preserving citations
- OpenAI embeddings + FAISS retrieval
- Grounded answer generation with strict citation format
- Retrieval evaluation with Recall@K and MRR

## Repository Layout

```text
multimodal-rag-ingest/
├── README.md
├── pyproject.toml
├── .env.example
├── Makefile
├── .gitignore
├── src/
├── data/
├── sources/
└── eval/
```

## Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`
- `ffmpeg` installed and available on PATH (required for YouTube audio extraction)
- OpenAI API key

## Setup

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env

# with uv
uv sync

# alternative
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI Commands

```bash
python -m src.cli ingest arxiv --query "retrieval augmented generation" --max 10
python -m src.cli ingest rss --feeds sources/rss_feeds.txt
python -m src.cli ingest youtube --urls sources/youtube_urls.txt
python -m src.cli normalize
python -m src.cli index
python -m src.cli query --q "What is RAG?"
python -m src.cli eval --file eval/questions.json
```

## Makefile Shortcuts

```bash
make setup
make ingest
make normalize
make index
make query Q="What is RAG?"
make eval
```

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
    "citation": "[arxiv:<doc_id> p.<page>] | [youtube:<doc_id> HH:MM:SS] | [rss:<doc_id>]"
  }
}
```

## Grounded Answer Behavior

- Uses retrieved chunks only
- Includes citations in required format
- If answer is missing from retrieved context, responds explicitly that it was not found

## Evaluation Format

`eval/questions.json` expects:

```json
[
  {
    "question": "What is retrieval augmented generation?",
    "relevant_doc_ids": ["abc123..."]
  }
]
```

The `eval` command prints:
- `Recall@K`
- `MRR`

## Notes

- YouTube ingestion first attempts captions/subtitles, then falls back to Whisper transcription when API key is available.
- Network/API failures are logged and skipped without crashing the whole run.
