from __future__ import annotations

from typing import Any

from pipeline.schemas import LLMConfig


def rewrite_query(query: str, enabled: bool, llm_config: LLMConfig) -> tuple[str, list[str]]:
    if not enabled:
        return query, []
    if not llm_config.api_key:
        return query, ["Query rewrite was enabled, but no API key was available."]

    try:
        client = _build_client(llm_config)
        response = client.chat.completions.create(
            model=llm_config.chat_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user query for document retrieval. "
                        "Preserve intent, add useful synonyms, and return one concise retrieval query only."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten or query, []
    except Exception as exc:
        return query, [f"Query rewrite failed and the original query was used: {exc}"]


def generate_answer(query: str, context: str, enabled: bool, llm_config: LLMConfig) -> tuple[str | None, list[str]]:
    if not enabled:
        return None, []
    if not llm_config.api_key:
        return None, ["Final answer generation was enabled, but no API key was available."]
    if not context.strip():
        return None, ["Final answer generation was skipped because the assembled context was empty."]

    try:
        client = _build_client(llm_config)
        response = client.chat.completions.create(
            model=llm_config.chat_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question only from the provided context. "
                        "If the context is insufficient, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{query}\n\nContext:\n{context}",
                },
            ],
        )
        return response.choices[0].message.content.strip(), []
    except Exception as exc:
        return None, [f"Final answer generation failed: {exc}"]


def _build_client(llm_config: LLMConfig) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url or None)
