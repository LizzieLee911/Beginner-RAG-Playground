from __future__ import annotations

import json
import re
from typing import Any

from pipeline.field_inference import infer_structured_fields
from pipeline.schemas import LLMConfig, StructuredFieldSelection


def suggest_structured_fields(
    rows: list[dict[str, Any]],
    file_preview: str,
    llm_config: LLMConfig,
    use_llm: bool,
) -> tuple[StructuredFieldSelection, list[str]]:
    rule_selection = infer_structured_fields(rows)
    warnings: list[str] = []
    if not use_llm:
        return rule_selection, warnings
    if not llm_config.api_key:
        warnings.append("Metadata inference was enabled, but no API key was available. Rule-based selection was used.")
        return rule_selection, warnings

    try:
        from openai import OpenAI

        client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url or None)
        response = client.chat.completions.create(
            model=llm_config.chat_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze structured dataset samples for a RAG tool. "
                        "Return JSON with keys text_fields, metadata_fields, identifier_fields, notes. "
                        "Prefer concise field names exactly as they appear in the sample."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Rule-based guess:\n"
                        f"text_fields={rule_selection.text_fields}\n"
                        f"metadata_fields={rule_selection.metadata_fields}\n"
                        f"identifier_fields={rule_selection.identifier_fields}\n\n"
                        "First 6000 characters of uploaded file:\n"
                        f"{file_preview[:6000]}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content.strip()
        payload = _extract_json(content)
        return StructuredFieldSelection(
            text_fields=[str(item) for item in payload.get("text_fields", []) if str(item).strip()],
            metadata_fields=[str(item) for item in payload.get("metadata_fields", []) if str(item).strip()],
            identifier_fields=[str(item) for item in payload.get("identifier_fields", []) if str(item).strip()],
            source="llm",
            notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
        ), warnings
    except Exception as exc:
        warnings.append(f"Metadata inference failed and rule-based selection was used: {exc}")
        return rule_selection, warnings


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
