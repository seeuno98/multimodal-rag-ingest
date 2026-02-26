from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_docs_schema() -> None:
    docs_path = Path("data/processed/docs.jsonl")
    if not docs_path.exists():
        pytest.skip(f"Missing artifact: {docs_path}")

    with docs_path.open("r", encoding="utf-8") as f:
        first_line = next((line.strip() for line in f if line.strip()), "")

    if not first_line:
        pytest.skip(f"No documents in {docs_path}")

    doc = json.loads(first_line)
    for key in ("doc_id", "source_type", "title", "segments"):
        assert key in doc, f"Missing key: {key}"

    segments = doc["segments"]
    assert isinstance(segments, list) and segments, "segments must be non-empty"

    segment = segments[0]
    text = segment.get("text", "")
    assert isinstance(text, str) and text.strip(), "segment.text must be non-empty"
