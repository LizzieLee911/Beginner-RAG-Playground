from __future__ import annotations

import re
from typing import Any, Iterable


LIKELY_TEXT_FIELDS = [
    "text",
    "content",
    "document",
    "note",
    "summary",
    "description",
    "body",
    "question",
    "answer",
    "response",
    "context",
]

LIKELY_METADATA_FIELDS = [
    "record_id",
    "text_id",
    "source",
    "label",
    "record_type",
    "section",
    "page",
    "page_number",
]

LIKELY_IDENTIFIER_FIELDS = [
    "record_id",
    "text_id",
    "id",
    "uuid",
    "uid",
    "doc_id",
    "document_id",
]


def detect_text_fields(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []

    key_lookup = {str(key).lower(): str(key) for key in rows[0].keys()}
    matched = [key_lookup[candidate] for candidate in LIKELY_TEXT_FIELDS if candidate in key_lookup]
    if matched:
        return matched

    scored_fields: list[tuple[float, str]] = []
    for key in rows[0].keys():
        stats = field_text_stats(rows, str(key))
        if stats["coverage"] == 0:
            continue
        score = stats["average_length"] * stats["coverage"]
        scored_fields.append((score, str(key)))

    scored_fields.sort(reverse=True)
    return [field_name for _, field_name in scored_fields[:1]]


def detect_text_field(rows: list[dict[str, Any]]) -> str | None:
    text_fields = detect_text_fields(rows)
    return text_fields[0] if text_fields else None


def detect_identifier_fields(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []

    key_lookup = {str(key).lower(): str(key) for key in rows[0].keys()}
    identifiers: list[str] = []
    for candidate in LIKELY_IDENTIFIER_FIELDS:
        if candidate in key_lookup:
            identifiers.append(key_lookup[candidate])

    if identifiers:
        return identifiers

    for key in rows[0].keys():
        key_text = str(key).lower()
        if key_text.endswith("_id") or key_text == "id":
            identifiers.append(str(key))
    return identifiers


def detect_metadata_fields(
    rows: list[dict[str, Any]],
    text_fields: list[str],
    identifier_fields: list[str],
) -> list[str]:
    if not rows:
        return []

    text_field_set = set(text_fields)
    identifier_set = set(identifier_fields)
    metadata_fields: list[str] = []
    for key in rows[0].keys():
        key_text = str(key)
        if key_text in text_field_set:
            continue
        if key_text in identifier_set:
            metadata_fields.append(key_text)
            continue
        if key_text.lower() in LIKELY_METADATA_FIELDS:
            metadata_fields.append(key_text)
            continue
        stats = field_text_stats(rows, key_text)
        if stats["is_scalar"] and stats["average_length"] < 120:
            metadata_fields.append(key_text)
    return dedupe_preserve_order(metadata_fields)


def field_text_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row.get(key) for row in rows]
    rendered = [value_to_text(value) for value in values]
    non_empty = [value.strip() for value in rendered if value.strip()]
    scalar_flags = [is_scalar_like(value) for value in values if value is not None]
    return {
        "coverage": len(non_empty) / max(len(rows), 1),
        "average_length": (sum(len(value) for value in non_empty) / len(non_empty)) if non_empty else 0.0,
        "is_scalar": all(scalar_flags) if scalar_flags else True,
    }


def is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested_value in value.items():
            nested_text = value_to_text(nested_value).strip()
            if nested_text:
                parts.append(f"{key}: {nested_text}")
        return "\n".join(parts)
    if isinstance(value, list):
        parts = [value_to_text(item).strip() for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


def row_to_text(
    row: dict[str, Any],
    text_fields: list[str] | None = None,
    excluded_fields: set[str] | None = None,
) -> str:
    excluded = excluded_fields or set()
    selected_fields = text_fields or []
    if selected_fields:
        parts: list[str] = []
        for field_name in selected_fields:
            if field_name in excluded:
                continue
            rendered = value_to_text(row.get(field_name)).strip()
            if rendered:
                parts.append(f"{field_name}: {rendered}")
        if parts:
            return "\n".join(parts)

    parts = []
    for key, value in row.items():
        if key in excluded:
            continue
        rendered = value_to_text(value).strip()
        if rendered:
            parts.append(f"{key}: {rendered}")
    return "\n".join(parts)


def safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value_to_text(value)


def preview_text(text: str, limit: int = 180) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_raw_field_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    field_names: set[str] = set()
    for row in rows:
        field_names.update(str(key) for key in row.keys())
    return sorted(field_names)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def extract_candidate_metadata_fields(field_names: Iterable[str]) -> list[str]:
    ordered = ["source_file", "page_number", "chunk_id", *LIKELY_IDENTIFIER_FIELDS, *LIKELY_METADATA_FIELDS]
    ordered.extend(sorted(field_names))
    return dedupe_preserve_order(ordered)
