from __future__ import annotations

from src.index.bm25 import tokenize


def test_tokenize_simple_text() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]
