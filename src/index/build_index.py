from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.index.chunk import chunk_documents
from src.index.embed import Embedder
from src.index.faiss_store import FaissStore
from src.ingest.normalize import read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


def build_index(
    docs_path: Path,
    chunks_path: Path,
    faiss_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    chunk_max_chars: int,
    chunk_overlap_chars: int,
) -> tuple[int, int]:
    docs = read_jsonl(docs_path)
    chunks = chunk_documents(docs, max_chars=chunk_max_chars, overlap_chars=chunk_overlap_chars)
    write_jsonl(chunks_path, chunks)
    if not chunks:
        raise ValueError("No chunks produced. Run ingestion and normalization first.")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embedding and indexing.")

    embedder = Embedder(api_key=openai_api_key, model=embed_model)
    vectors = embedder.embed_texts([chunk["text"] for chunk in chunks])
    store = FaissStore(dim=vectors.shape[1])
    store.add(vectors=vectors, metadata=chunks)
    store.save(index_path=faiss_path, metadata_path=metadata_path)
    LOGGER.info("Indexed %s chunks from %s documents", len(chunks), len(docs))
    return len(docs), len(chunks)
