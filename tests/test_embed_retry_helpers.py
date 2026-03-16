from __future__ import annotations

import pytest

from src.index.embed import BACKOFF_CAP_S, Embedder, EmbeddingRequest, compute_backoff_s, parse_retry_ms


def test_parse_retry_ms() -> None:
    message = "Please try again in 125ms"
    assert parse_retry_ms(message) == 125


def test_backoff_schedule_monotonic() -> None:
    values = [compute_backoff_s(attempt) for attempt in range(1, 10)]
    assert values == sorted(values)
    assert values[-1] <= BACKOFF_CAP_S
    assert values[-1] == BACKOFF_CAP_S


def test_embed_batch_with_fallback_splits_and_preserves_order(monkeypatch) -> None:
    embedder = Embedder(api_key="test-key", model="text-embedding-3-small")

    def fake_embed_batch(batch: list[str], batch_index: int) -> list[list[float]]:
        if len(batch) > 1:
            raise RuntimeError(f"batch {batch_index} failed")
        return [[float(ord(batch[0][-1]))]]

    monkeypatch.setattr(embedder, "_embed_batch", fake_embed_batch)

    vectors, failures = embedder._embed_batch_with_fallback(
        [
            (0, EmbeddingRequest(chunk_id="c1", doc_id="d1", text="doc-1")),
            (1, EmbeddingRequest(chunk_id="c2", doc_id="d2", text="doc-2")),
            (2, EmbeddingRequest(chunk_id="c3", doc_id="d3", text="doc-3")),
        ],
        batch_index=1,
    )

    assert failures == []
    assert vectors == [(0, [49.0]), (1, [50.0]), (2, [51.0])]


def test_embed_batch_with_fallback_raises_for_single_text(monkeypatch) -> None:
    embedder = Embedder(api_key="test-key", model="text-embedding-3-small")

    def fake_embed_batch(batch: list[str], batch_index: int) -> list[list[float]]:
        raise RuntimeError(f"batch {batch_index} failed")

    monkeypatch.setattr(embedder, "_embed_batch", fake_embed_batch)

    with pytest.raises(RuntimeError):
        embedder._embed_batch_with_fallback(
            [(0, EmbeddingRequest(chunk_id="c1", doc_id="d1", text="doc-1"))],
            batch_index=1,
        )


def test_embed_texts_with_failures_skips_permanent_single_chunk(monkeypatch) -> None:
    embedder = Embedder(api_key="test-key", model="text-embedding-3-small")

    def fake_embed_batch(batch: list[str], batch_index: int) -> list[list[float]]:
        if batch == ["bad chunk"]:
            raise RuntimeError(f"batch {batch_index} failed")
        if len(batch) > 1:
            raise RuntimeError(f"batch {batch_index} failed")
        return [[float(len(batch[0]))]]

    monkeypatch.setattr(embedder, "_embed_batch", fake_embed_batch)

    vectors, successful_rows, failures = embedder.embed_texts_with_failures(
        [
            {"chunk_id": "c1", "doc_id": "d1", "text": "good"},
            {"chunk_id": "c2", "doc_id": "d2", "text": "bad chunk"},
            {"chunk_id": "c3", "doc_id": "d3", "text": "fine"},
        ]
    )

    assert successful_rows == [
        {"chunk_id": "c1", "doc_id": "d1", "text": "good"},
        {"chunk_id": "c3", "doc_id": "d3", "text": "fine"},
    ]
    assert vectors.tolist() == [[4.0], [4.0]]
    assert len(failures) == 1
    assert failures[0].chunk_id == "c2"
