from __future__ import annotations

import json
import re
from typing import Any

from pipeline.field_inference import infer_structured_fields
from pipeline.schemas import LLMConfig, StructuredFieldSelection


_FIELD_SPLIT_PATTERN = re.compile(r"[,;\n]+")


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
        return normalize_suggestion_payload(payload), warnings
    except Exception as exc:
        warnings.append(f"Metadata inference failed and rule-based selection was used: {exc}")
        return rule_selection, warnings


def normalize_suggestion_payload(payload: dict[str, Any]) -> StructuredFieldSelection:
    return StructuredFieldSelection(
        text_fields=_coerce_field_list(payload.get("text_fields")),
        metadata_fields=_coerce_field_list(payload.get("metadata_fields")),
        identifier_fields=_coerce_field_list(payload.get("identifier_fields")),
        source="llm",
        notes=_coerce_note_list(payload.get("notes")),
    )


def _coerce_field_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = [part.strip() for part in _FIELD_SPLIT_PATTERN.split(text) if part.strip()]
        return _dedupe(parts or [text])
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_coerce_field_list(item))
        return _dedupe(values)

    text = str(value).strip()
    return [text] if text else []


def _coerce_note_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        notes: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                notes.append(text)
        return _dedupe(notes)

    text = str(value).strip()
    return [text] if text else []


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
