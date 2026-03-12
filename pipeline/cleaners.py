from __future__ import annotations

from dataclasses import replace

from pipeline.schemas import DocumentUnit
from pipeline.utils import normalize_whitespace


def clean_text(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    return normalize_whitespace(text)


def clean_units(units: list[DocumentUnit], mode: str) -> list[DocumentUnit]:
    return [replace(unit, text=clean_text(unit.text, mode)) for unit in units]
