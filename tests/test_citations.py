from __future__ import annotations

from src.rag.answer import extract_citations, validate_citations


def test_extract_citations_regex() -> None:
    text = "Use [youtube:abc 00:00:19] and [rss:def] for support."
    assert extract_citations(text) == ["[youtube:abc 00:00:19]", "[rss:def]"]


def test_validate_citations() -> None:
    allowed = {"[youtube:a 00:00:19]", "[rss:b]"}
    answer = "text [1] and [rss:b]"
    valid, invalid = validate_citations(answer, allowed)
    assert valid == ["[rss:b]"]
    assert invalid == ["[1]"]
