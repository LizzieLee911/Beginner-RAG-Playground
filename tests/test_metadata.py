from pipeline.metadata import build_metadata
from pipeline.schemas import ChunkRecord


def test_full_metadata_contains_structured_fields() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="sample",
        source_name="sample.json",
        unit_type="record",
        page_number=2,
        section="plan",
        record_id="rec-1",
        text_id="txt-1",
        label="warning",
        record_type="instruction",
        source="clinic",
        raw_fields={"custom_field": "x"},
    )
    metadata = build_metadata(chunk, "full")
    assert metadata["chunk_id"] == "chunk-1"
    assert metadata["page_number"] == 2
    assert metadata["record_id"] == "rec-1"
    assert metadata["custom_field"] == "x"
