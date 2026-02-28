from __future__ import annotations


SYSTEM_PROMPT = """You are a grounded RAG assistant.
Rules:
1) Use only the supplied context chunks.
2) Use only citations from the Allowed citations list exactly as written.
3) Do not invent numeric citations like [1] or any new citation formats.
4) Every non-trivial factual claim must include at least one allowed citation.
5) If the answer is not supported by context, say you do not know based on retrieved sources.
"""


def build_user_prompt(
    question: str,
    contexts: list[dict],
    allowed_citations: list[str],
    top_citations: list[str],
    invalid_citations: list[str] | None = None,
    require_top_citation: bool = False,
    retry_instruction: str | None = None,
) -> str:
    formatted_contexts: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        citation = ctx.get("metadata", {}).get("citation", "")
        text = ctx.get("text", "")
        formatted_contexts.append(f"Chunk {i}\nCitation: {citation}\nText: {text}")

    allowed_block = "\n".join(f"- {citation}" for citation in allowed_citations) or "- (none)"
    top_block = "\n".join(f"- {citation}" for citation in top_citations) or "- (none)"
    invalid_block = ""
    if invalid_citations:
        invalid_list = ", ".join(invalid_citations)
        invalid_block = (
            "\n\nRetry instruction:\n"
            f"You used invalid citations: {invalid_list}. "
            "Rewrite using only the allowed citations and keep factual claims grounded in context."
        )
    if retry_instruction:
        invalid_block += f"\n{retry_instruction}"
    top_rule_block = ""
    if require_top_citation and top_citations:
        top_rule_block = (
            "\n\nGrounding rule:\n"
            "Your answer MUST include at least one citation from the Top citations list."
        )

    context_blob = "\n\n".join(formatted_contexts)
    return (
        f"Question:\n{question}\n\n"
        "Top citations (use at least one):\n"
        f"{top_block}\n\n"
        "Allowed citations (MUST use exactly, do not create new ones):\n"
        f"{allowed_block}\n\n"
        f"Context:\n{context_blob}\n"
        f"{top_rule_block}"
        f"{invalid_block}\n\n"
        "Answer using only allowed citations. Prefer inline citations for key claims."
    )
