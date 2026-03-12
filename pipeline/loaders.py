from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.field_inference import infer_structured_fields
from pipeline.pdf_parsers import parse_pdf
from pipeline.schemas import DocumentUnit, ParsedInput, StructuredFieldSelection
from pipeline.utils import collect_raw_field_names, preview_text, row_to_text, safe_scalar


def detect_file_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return "pdf"
    if extension == ".json":
        return "json"
    if extension == ".csv":
        return "csv"
    raise ValueError(f"Unsupported file type: {extension or 'unknown'}")


def load_input(
    file_bytes: bytes,
    filename: str,
    field_selection: StructuredFieldSelection | None = None,
    pdf_parser_mode: str = "pypdf",
) -> ParsedInput:
    file_type = detect_file_type(filename)
    if file_type == "pdf":
        units, preview_rows, summary, warnings = parse_pdf(file_bytes, filename, pdf_parser_mode)
        return ParsedInput(
            source_name=filename,
            file_type="pdf",
            units=units,
            preview_rows=preview_rows,
            summary=summary,
            warnings=warnings,
            field_selection=field_selection,
        )
    if file_type == "json":
        return _load_json(file_bytes, filename, field_selection)
    return _load_csv(file_bytes, filename, field_selection)


def _load_json(
    file_bytes: bytes,
    filename: str,
    field_selection: StructuredFieldSelection | None,
) -> ParsedInput:
    text = _decode_text(file_bytes)
    payload = json.loads(text)
    records = _normalize_json_payload(payload)
    return _load_structured_records(records, filename, "json", field_selection)


def _load_csv(
    file_bytes: bytes,
    filename: str,
    field_selection: StructuredFieldSelection | None,
) -> ParsedInput:
    text = _decode_text(file_bytes)
    dataframe = pd.read_csv(io.StringIO(text))
    records = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")
    return _load_structured_records(records, filename, "csv", field_selection)


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def _normalize_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            normalized.extend(_normalize_json_item(item, index))
        return normalized

    if isinstance(payload, dict):
        list_records: list[dict[str, Any]] = []
        for key, value in payload.items():
            if isinstance(value, list):
                nested_records = _normalize_json_payload(value)
                for record in nested_records:
                    if "container_key" not in record:
                        record["container_key"] = key
                list_records.extend(nested_records)
        if list_records:
            return list_records

        if payload and all(isinstance(value, dict) for value in payload.values()):
            records: list[dict[str, Any]] = []
            for outer_key, inner_value in payload.items():
                row = {str(key): value for key, value in inner_value.items()}
                row.setdefault("record_id", outer_key)
                row["json_key"] = outer_key
                records.append(row)
            return records

        return [{str(key): value for key, value in payload.items()}]

    return [{"value": payload}]


def _normalize_json_item(item: Any, index: int) -> list[dict[str, Any]]:
    if isinstance(item, dict):
        return [{str(key): value for key, value in item.items()}]
    return [{"value": item, "record_id": index}]


def _load_structured_records(
    rows: list[dict[str, Any]],
    filename: str,
    file_type: str,
    field_selection: StructuredFieldSelection | None,
) -> ParsedInput:
    warnings: list[str] = []
    sanitized_rows = [{str(key): safe_scalar(value) for key, value in row.items()} for row in rows]
    applied_selection = field_selection or infer_structured_fields(sanitized_rows)

    units: list[DocumentUnit] = []
    preview_rows: list[dict[str, Any]] = []
    structured_fields = collect_raw_field_names(sanitized_rows)

    identifier_candidates = applied_selection.identifier_fields or ["record_id", "id", "text_id"]
    for row_index, row in enumerate(sanitized_rows, start=1):
        record_id = _pick_first(row, identifier_candidates + ["record_id", "id", "recordid"])
        text_id = _pick_first(row, ["text_id", "textid"])
        source = _pick_first(row, ["source", "source_name", "origin"])
        label = _pick_first(row, ["label", "category"])
        record_type = _pick_first(row, ["record_type", "type", "element_type"])
        section = _pick_first(row, ["section", "heading", "title"])
        excluded_fields = set(applied_selection.identifier_fields)
        text = row_to_text(row, applied_selection.text_fields, excluded_fields=excluded_fields)
        if not text.strip():
            warnings.append(f"Record {row_index} produced empty text and was kept for preview.")

        resolved_record_id = coerce_if_present(record_id) or f"record_{row_index}"
        units.append(
            DocumentUnit(
                doc_id=f"record_{row_index}",
                source_name=filename,
                text=text,
                unit_type="record",
                section=section,
                record_id=resolved_record_id,
                text_id=coerce_if_present(text_id),
                label=coerce_if_present(label),
                record_type=coerce_if_present(record_type),
                source=coerce_if_present(source),
                raw_fields=row,
            )
        )
        preview_rows.append(
            {
                "row": row_index,
                "record_id": resolved_record_id,
                "text_id": text_id,
                "preview": preview_text(text),
            }
        )

    summary = {
        "file_type": file_type,
        "record_count": len(units),
        "detected_text_field": ", ".join(applied_selection.text_fields) if applied_selection.text_fields else "fallback_join",
        "detected_text_fields": applied_selection.text_fields,
        "detected_metadata_fields": applied_selection.metadata_fields,
        "detected_identifier_fields": applied_selection.identifier_fields,
        "structured_fields": structured_fields,
    }
    return ParsedInput(
        source_name=filename,
        file_type=file_type,
        units=units,
        preview_rows=preview_rows[:12],
        summary=summary,
        warnings=warnings,
        detected_text_field=summary["detected_text_field"],
        structured_fields=structured_fields,
        field_selection=applied_selection,
    )


def _pick_first(row: dict[str, Any], candidate_keys: list[str]) -> Any:
    lower_map = {str(key).lower(): key for key in row.keys()}
    for candidate in candidate_keys:
        key = lower_map.get(candidate.lower())
        if key is not None:
            return row.get(key)
    return None


def coerce_if_present(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
