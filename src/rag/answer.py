from __future__ import annotations

from openai import OpenAI

from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt


def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict],
    openai_api_key: str,
    chat_model: str,
) -> str:
    if not retrieved_chunks:
        return "I could not find the answer in the retrieved sources."
    if not openai_api_key:
        citations = " ".join(chunk.get("metadata", {}).get("citation", "") for chunk in retrieved_chunks[:2])
        return f"OPENAI_API_KEY is not set. Retrieved context available: {citations}".strip()

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, retrieved_chunks)},
        ],
    )
    return response.choices[0].message.content or "I could not find the answer in the retrieved sources."
