from __future__ import annotations

from src.rag.answer import contains_top_citation


def test_contains_top_citation_true() -> None:
    top = ["[youtube:x 00:00:19]", "[rss:y]"]
    ans = "Hello [rss:y]"
    assert contains_top_citation(ans, top) is True


def test_contains_top_citation_false() -> None:
    top = ["[youtube:x 00:00:19]", "[rss:y]"]
    ans = "Hello [arxiv:z p.1]"
    assert contains_top_citation(ans, top) is False
