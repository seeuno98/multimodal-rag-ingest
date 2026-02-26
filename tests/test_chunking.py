from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_chunks_schema() -> None:
    chunks_path = Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        pytest.skip(f"Missing artifact: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as f:
        first_line = next((line.strip() for line in f if line.strip()), "")

    if not first_line:
        pytest.skip(f"No chunks in {chunks_path}")

    chunk = json.loads(first_line)
    assert chunk.get("chunk_id"), "chunk_id is required"
    assert chunk.get("doc_id"), "doc_id is required"
    text = chunk.get("text", "")
    assert isinstance(text, str) and len(text.strip()) > 20, "chunk text must be longer than 20 chars"
