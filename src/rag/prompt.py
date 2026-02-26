SYSTEM_PROMPT = """You are a grounded RAG assistant.
Use only the supplied context chunks.
If the answer is not in context, explicitly say you cannot find it in the retrieved sources.
Every claim must include at least one citation token exactly as provided in the context.
Do not fabricate citations or external facts.
"""


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    formatted = []
    for i, ctx in enumerate(contexts, start=1):
        citation = ctx.get("metadata", {}).get("citation", "")
        formatted.append(f"[{i}] {citation}\n{ctx.get('text', '')}")
    context_blob = "\n\n".join(formatted)
    return f"Question:\n{question}\n\nContext:\n{context_blob}\n\nAnswer with citations."
