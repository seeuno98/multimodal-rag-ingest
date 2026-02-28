from __future__ import annotations

import re

CITATION_PATTERN = re.compile(r"\[[^\]]+\]")


def extract_bracket_citations(text: str) -> list[str]:
    return CITATION_PATTERN.findall(text)


def parse_youtube_timestamp(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        if not (minutes.isdigit() and seconds.isdigit()):
            raise ValueError(f"Invalid YouTube timestamp: {ts}")
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        if not (hours.isdigit() and minutes.isdigit() and seconds.isdigit()):
            raise ValueError(f"Invalid YouTube timestamp: {ts}")
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    raise ValueError(f"Invalid YouTube timestamp: {ts}")


def build_citation_url_map(retrieved_chunks: list[dict]) -> dict[str, str]:
    citation_map: dict[str, str] = {}
    for chunk in retrieved_chunks:
        citation = chunk.get("citation") or chunk.get("metadata", {}).get("citation")
        source_uri = chunk.get("source_uri") or chunk.get("metadata", {}).get("source_uri")
        if not citation or not source_uri:
            continue

        if citation.startswith("[youtube:") and " " in citation:
            timestamp = citation.rstrip("]").split(" ", 1)[1]
            seconds = parse_youtube_timestamp(timestamp)
            separator = "&" if "?" in source_uri else "?"
            citation_map[citation] = f"{source_uri}{separator}t={seconds}s"
            continue

        citation_map[citation] = source_uri
    return citation_map


def replace_citations_with_urls(answer: str, citation_url_map: dict[str, str]) -> str:
    pieces: list[str] = []
    last_index = 0

    for match in CITATION_PATTERN.finditer(answer):
        citation = match.group(0)
        url = citation_url_map.get(citation)
        if not url:
            continue

        start, end = match.span()
        replacement = f"({url})"
        replace_start = start
        replace_end = end

        left = start - 1
        while left >= 0 and answer[left].isspace():
            left -= 1

        right = end
        while right < len(answer) and answer[right].isspace():
            right += 1

        if left >= 0 and right < len(answer) and answer[left] == "(" and answer[right] == ")":
            replacement = f"({url})"
            replace_start = left
            replace_end = right + 1

        pieces.append(answer[last_index:replace_start])
        pieces.append(replacement)
        last_index = replace_end

    pieces.append(answer[last_index:])
    return "".join(pieces)
