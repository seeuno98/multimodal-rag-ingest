from __future__ import annotations

import json

from src.index.embed import (
    append_embeddings_to_cache,
    load_embedding_cache,
    validate_cache_meta,
)


def test_load_embedding_cache_returns_expected_keys(tmp_path) -> None:
    cache_path = tmp_path / "embeddings_cache.jsonl"
    cache_path.write_text(
        "\n".join(
            [
                json.dumps({"chunk_id": "chunk-a", "embedding": [0.1, 0.2]}),
                json.dumps({"chunk_id": "chunk-b", "embedding": [0.3, 0.4]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cache = load_embedding_cache(cache_path)

    assert set(cache) == {"chunk-a", "chunk-b"}


def test_append_embeddings_to_cache_then_reload(tmp_path) -> None:
    cache_path = tmp_path / "embeddings_cache.jsonl"

    append_embeddings_to_cache(cache_path, [("chunk-new", [1.0, 2.0, 3.0])])
    cache = load_embedding_cache(cache_path)

    assert cache["chunk-new"] == [1.0, 2.0, 3.0]


def test_validate_cache_meta_match_and_mismatch() -> None:
    meta = {"model": "text-embedding-3-small", "dimension": 1536}

    assert validate_cache_meta(meta, "text-embedding-3-small", 1536) is True
    assert validate_cache_meta(meta, "text-embedding-3-large", 1536) is False
    assert validate_cache_meta(meta, "text-embedding-3-small", 3072) is False
