from __future__ import annotations

import json

import numpy as np

from src.index.build_index import build_index
from src.index.embed import EmbeddingFailure
from src.ingest.normalize import write_jsonl


def test_build_index_skips_failed_embeddings(tmp_path, monkeypatch) -> None:
    docs_path = tmp_path / "docs.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    faiss_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.jsonl"
    bm25_path = tmp_path / "bm25.joblib"
    write_jsonl(
        docs_path,
        [
            {
                "doc_id": "doc-1",
                "source_type": "rss_blog",
                "title": "Doc 1",
                "source_uri": "https://example.com/1",
                "segments": [{"segment_id": "s1", "text": "alpha", "metadata": {}}],
            }
        ],
    )

    successful_chunks = [
        {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "text": "alpha",
            "metadata": {"citation": "[rss:doc-1]", "source_uri": "https://example.com/1"},
        }
    ]
    failures = [
        EmbeddingFailure(
            batch_index=7,
            chunk_id="chunk-2",
            doc_id="doc-1",
            text="beta",
            error="RuntimeError: failed",
        )
    ]

    monkeypatch.setattr(
        "src.index.build_index.chunk_documents",
        lambda docs, max_chars, overlap_chars: successful_chunks
        + [
            {
                "chunk_id": "chunk-2",
                "doc_id": "doc-1",
                "text": "beta",
                "metadata": {"citation": "[rss:doc-1]", "source_uri": "https://example.com/1"},
            }
        ],
    )
    monkeypatch.setattr(
        "src.index.build_index.Embedder.embed_texts_with_failures",
        lambda self, chunks: (np.asarray([[1.0, 0.0]], dtype=np.float32), successful_chunks, failures),
    )

    doc_count, chunk_count = build_index(
        docs_path=docs_path,
        chunks_path=chunks_path,
        faiss_path=faiss_path,
        metadata_path=metadata_path,
        bm25_path=bm25_path,
        openai_api_key="test-key",
        embed_model="text-embedding-3-small",
        chunk_max_chars=100,
        chunk_overlap_chars=10,
    )

    assert doc_count == 1
    assert chunk_count == 1
    assert faiss_path.exists()
    assert metadata_path.exists()
    assert bm25_path.exists()
    failed_log = tmp_path / "failed_embeddings.jsonl"
    assert failed_log.exists()
    rows = [json.loads(line) for line in failed_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["chunk_id"] == "chunk-2"
