from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader

from pipeline.schemas import DocumentUnit
from pipeline.utils import preview_text


def parse_pdf(
    file_bytes: bytes,
    filename: str,
    parser_mode: str,
) -> tuple[list[DocumentUnit], list[dict[str, Any]], dict[str, Any], list[str]]:
    if parser_mode == "unstructured":
        units, preview_rows, summary, warnings = _parse_pdf_with_unstructured(file_bytes, filename)
        if units:
            return units, preview_rows, summary, warnings
        warnings.append("Unstructured parser returned no units. Falling back to pypdf.")
        fallback_units, fallback_rows, fallback_summary, fallback_warnings = _parse_pdf_with_pypdf(file_bytes, filename)
        return fallback_units, fallback_rows, fallback_summary, warnings + fallback_warnings

    return _parse_pdf_with_pypdf(file_bytes, filename)


def _parse_pdf_with_pypdf(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[DocumentUnit], list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    units: list[DocumentUnit] = []

    reader = PdfReader(io.BytesIO(file_bytes))
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"Page {page_index} extraction failed: {exc}")
            page_text = ""

        if not page_text.strip():
            warnings.append(f"Page {page_index} returned no extractable text.")

        units.append(
            DocumentUnit(
                doc_id=f"page_{page_index}",
                source_name=filename,
                text=page_text,
                unit_type="page",
                page_number=page_index,
                raw_fields={"page_number": page_index},
            )
        )
        preview_rows.append({"page_number": page_index, "preview": preview_text(page_text)})

    summary = {
        "file_type": "pdf",
        "page_count": len(units),
        "non_empty_pages": sum(1 for unit in units if unit.text.strip()),
        "pdf_parser": "pypdf",
    }
    return units, preview_rows[:10], summary, warnings


def _parse_pdf_with_unstructured(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[DocumentUnit], list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    units: list[DocumentUnit] = []

    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception as exc:  # pragma: no cover - optional dependency path
        warnings.append(f"Unstructured parser is unavailable: {exc}")
        return [], [], {"file_type": "pdf", "pdf_parser": "unstructured"}, warnings

    try:
        elements = partition_pdf(file=io.BytesIO(file_bytes), strategy="fast")
    except Exception as exc:  # pragma: no cover - runtime fallback path
        warnings.append(f"Unstructured parsing failed: {exc}")
        return [], [], {"file_type": "pdf", "pdf_parser": "unstructured"}, warnings

    current_section: str | None = None
    for element_index, element in enumerate(elements, start=1):
        text = (getattr(element, "text", None) or "").strip()
        if not text:
            continue
        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", None) if metadata else None
        category = getattr(element, "category", type(element).__name__)
        if str(category).lower() == "title":
            current_section = text
        units.append(
            DocumentUnit(
                doc_id=f"element_{element_index}",
                source_name=filename,
                text=text,
                unit_type="record",
                page_number=page_number,
                section=current_section,
                record_type=str(category),
                source=filename,
                raw_fields={
                    "page_number": page_number,
                    "element_type": str(category),
                },
            )
        )
        preview_rows.append(
            {
                "element": element_index,
                "page_number": page_number,
                "section": current_section,
                "element_type": str(category),
                "preview": preview_text(text),
            }
        )

    summary = {
        "file_type": "pdf",
        "page_count": len({unit.page_number for unit in units if unit.page_number is not None}),
        "element_count": len(units),
        "pdf_parser": "unstructured",
    }
    return units, preview_rows[:20], summary, warnings
