from __future__ import annotations

from dataclasses import replace
from typing import Any

from pipeline.schemas import ChunkRecord


def attach_metadata(chunks: list[ChunkRecord], mode: str) -> list[ChunkRecord]:
    return [replace(chunk, metadata=build_metadata(chunk, mode)) for chunk in chunks]


def build_metadata(chunk: ChunkRecord, mode: str) -> dict[str, Any]:
    if mode == "none":
        return {}

    metadata: dict[str, Any] = {
        "source_file": chunk.source_name,
        "chunk_id": chunk.chunk_id,
    }
    if mode in {"page_source", "full"} and chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number

    if mode == "full":
        optional_fields = {
            "source": chunk.source,
            "section": chunk.section,
            "record_id": chunk.record_id,
            "text_id": chunk.text_id,
            "label": chunk.label,
            "record_type": chunk.record_type,
        }
        for key, value in optional_fields.items():
            if value not in (None, ""):
                metadata[key] = value

        for key, value in chunk.raw_fields.items():
            if key in metadata:
                continue
            if isinstance(value, (str, int, float, bool)) and value not in ("", None):
                metadata[key] = value

    return metadata


def collect_metadata_fields(chunks: list[ChunkRecord]) -> list[str]:
    field_names: set[str] = set()
    for chunk in chunks:
        field_names.update(chunk.metadata.keys())
    return sorted(field_names)
