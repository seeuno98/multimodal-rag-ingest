from __future__ import annotations

import re
from collections.abc import Iterable

from openai import OpenAI

from src.rag.citations import (
    build_citation_url_map,
    extract_bracket_citations,
    replace_citations_with_urls,
)
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt


def extract_citations(text: str) -> list[str]:
    return extract_bracket_citations(text)


def validate_citations(text: str, allowed_set: set[str]) -> tuple[list[str], list[str]]:
    seen_valid: set[str] = set()
    seen_invalid: set[str] = set()
    valid: list[str] = []
    invalid: list[str] = []
    for citation in extract_citations(text):
        if citation in allowed_set:
            if citation not in seen_valid:
                seen_valid.add(citation)
                valid.append(citation)
        elif citation not in seen_invalid:
            seen_invalid.add(citation)
            invalid.append(citation)
    return valid, invalid


def contains_top_citation(text: str, top_citations: list[str]) -> bool:
    if not top_citations:
        return True
    extracted = set(extract_citations(text))
    return any(citation in extracted for citation in top_citations)


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _strip_invalid_citations(text: str, allowed_set: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        citation = match.group(0)
        return citation if citation in allowed_set else ""

    cleaned = re.sub(r"\[[^\]]+\]", replace, text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _finalize_answer_for_display(answer: str, retrieved_chunks: list[dict]) -> str:
    citation_url_map = build_citation_url_map(retrieved_chunks)
    return replace_citations_with_urls(answer, citation_url_map)


def _call_chat_model(
    client: OpenAI,
    chat_model: str,
    question: str,
    retrieved_chunks: list[dict],
    allowed_citations: list[str],
    top_citations: list[str],
    invalid_citations: list[str] | None = None,
    require_top_citation: bool = False,
    retry_instruction: str | None = None,
) -> str:
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    question=question,
                    contexts=retrieved_chunks,
                    allowed_citations=allowed_citations,
                    top_citations=top_citations,
                    invalid_citations=invalid_citations,
                    require_top_citation=require_top_citation,
                    retry_instruction=retry_instruction,
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict],
    openai_api_key: str,
    chat_model: str,
) -> str:
    if not retrieved_chunks:
        return "I could not find the answer in the retrieved sources."

    allowed_citations = _unique_preserve_order(
        (
            chunk.get("citation", "")
            or chunk.get("metadata", {}).get("citation", "")
            for chunk in retrieved_chunks
        )
    )
    allowed_set = set(allowed_citations)
    top_citations = allowed_citations[:2]

    if not openai_api_key:
        citations = " ".join(allowed_citations[:2])
        raw_answer = f"OPENAI_API_KEY is not set. Retrieved context available: {citations}".strip()
        return _finalize_answer_for_display(raw_answer, retrieved_chunks)

    client = OpenAI(api_key=openai_api_key)
    first_pass = _call_chat_model(
        client=client,
        chat_model=chat_model,
        question=question,
        retrieved_chunks=retrieved_chunks,
        allowed_citations=allowed_citations,
        top_citations=top_citations,
        require_top_citation=bool(top_citations),
    )
    _, invalid_first = validate_citations(first_pass, allowed_set)
    has_top_first = contains_top_citation(first_pass, top_citations)
    if not invalid_first and has_top_first:
        raw_answer = first_pass or "I could not find the answer in the retrieved sources."
        return _finalize_answer_for_display(raw_answer, retrieved_chunks)

    retry_invalid: list[str] | None = invalid_first if invalid_first else None
    retry_instruction: str | None = None
    retry_require_top = bool(top_citations) and not has_top_first
    if retry_require_top:
        top_list = ", ".join(top_citations)
        retry_instruction = (
            "Rewrite the answer to include at least one of these Top citations: "
            f"{top_list}. Use only allowed citations."
        )

    retry_pass = _call_chat_model(
        client=client,
        chat_model=chat_model,
        question=question,
        retrieved_chunks=retrieved_chunks,
        allowed_citations=allowed_citations,
        top_citations=top_citations,
        invalid_citations=retry_invalid,
        require_top_citation=bool(top_citations),
        retry_instruction=retry_instruction,
    )
    _, invalid_retry = validate_citations(retry_pass, allowed_set)
    has_top_retry = contains_top_citation(retry_pass, top_citations)
    if not invalid_retry and has_top_retry:
        raw_answer = retry_pass or "I could not find the answer in the retrieved sources."
        return _finalize_answer_for_display(raw_answer, retrieved_chunks)

    sanitized = _strip_invalid_citations(retry_pass, allowed_set)
    if top_citations and not contains_top_citation(sanitized, top_citations):
        sanitized = f"{sanitized}\n\n{top_citations[0]}".strip()
    if not sanitized:
        return "I could not find the answer in the retrieved sources."
    raw_answer = (
        sanitized
        + "\n\nNote: Some citations could not be verified; showing only verified citations."
    )
    return _finalize_answer_for_display(raw_answer, retrieved_chunks)
