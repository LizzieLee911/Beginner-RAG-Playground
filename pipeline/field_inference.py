from __future__ import annotations

from pipeline.schemas import StructuredFieldSelection
from pipeline.utils import detect_identifier_fields, detect_metadata_fields, detect_text_fields, dedupe_preserve_order


def infer_structured_fields(rows: list[dict[str, object]]) -> StructuredFieldSelection:
    text_fields = detect_text_fields(rows)
    identifier_fields = detect_identifier_fields(rows)
    metadata_fields = detect_metadata_fields(rows, text_fields, identifier_fields)
    notes: list[str] = []
    if not text_fields:
        notes.append("No obvious text fields were detected; fallback text concatenation will be used.")
    return StructuredFieldSelection(
        text_fields=dedupe_preserve_order(text_fields),
        metadata_fields=dedupe_preserve_order(metadata_fields),
        identifier_fields=dedupe_preserve_order(identifier_fields),
        source="rule",
        notes=notes,
    )
