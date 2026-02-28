from __future__ import annotations

from src.rag.citations import (
    build_citation_url_map,
    parse_youtube_timestamp,
    replace_citations_with_urls,
)


def test_parse_youtube_timestamp() -> None:
    assert parse_youtube_timestamp("00:02:53") == 173
    assert parse_youtube_timestamp("2:53") == 173


def test_build_citation_url_map_and_replace() -> None:
    retrieved_chunks = [
        {"citation": "[rss:abc]", "source_uri": "https://example.com"},
        {
            "citation": "[youtube:xyz 00:02:53]",
            "source_uri": "https://www.youtube.com/watch?v=VID",
        },
    ]
    answer = "Text [rss:abc] and [youtube:xyz 00:02:53]"

    citation_url_map = build_citation_url_map(retrieved_chunks)
    output = replace_citations_with_urls(answer, citation_url_map)

    assert "(https://example.com)" in output
    assert "(https://www.youtube.com/watch?v=VID&t=173s)" in output


def test_replace_citations_already_parenthesized() -> None:
    citation_url_map = {"[rss:abc]": "https://example.com"}
    answer = "Hello ([rss:abc])."

    output = replace_citations_with_urls(answer, citation_url_map)

    assert output == "Hello (https://example.com)."


def test_replace_citations_not_parenthesized() -> None:
    citation_url_map = {"[rss:abc]": "https://example.com"}
    answer = "Hello [rss:abc]."

    output = replace_citations_with_urls(answer, citation_url_map)

    assert output == "Hello (https://example.com)."


def test_replace_citations_parenthesized_with_whitespace() -> None:
    citation_url_map = {"[rss:abc]": "https://example.com"}
    answer = "Hello ( [rss:abc] )."

    output = replace_citations_with_urls(answer, citation_url_map)

    assert output == "Hello (https://example.com)."
