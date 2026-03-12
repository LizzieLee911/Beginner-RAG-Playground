from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pipeline.chunkers import chunk_units
from pipeline.cleaners import clean_units
from pipeline.context_builder import assemble_context
from pipeline.indexers import build_index_bundle
from pipeline.loaders import load_input
from pipeline.metadata import attach_metadata, collect_metadata_fields
from pipeline.query_ops import generate_answer, rewrite_query
from pipeline.rerankers import rerank_results
from pipeline.retrievers import retrieve, trace_items
from pipeline.schemas import ChunkRecord, DocumentUnit, ParsedInput, PipelineConfig, PreparedPreview, PreviewRun, RetrievedResult
from pipeline.utils import preview_text


DEFAULT_CHUNK_PREVIEW_LIMIT = 100


def prepare_preview_artifacts(file_bytes: bytes, filename: str, config: PipelineConfig) -> PreparedPreview:
    parsed_input = load_input(
        file_bytes,
        filename,
        field_selection=config.field_selection,
        pdf_parser_mode=config.pdf_parser_mode,
    )
    cleaned_units = clean_units(parsed_input.units, config.cleaning_mode)
    chunk_input_units = _merge_structured_units(cleaned_units, parsed_input, config)
    chunks = chunk_units(chunk_input_units, config.chunk_strategy, config.chunk_size, config.chunk_overlap)
    chunks = attach_metadata(chunks, config.metadata_mode)
    metadata_fields = collect_metadata_fields(chunks)
    index_bundle = build_index_bundle(chunks, config)
    warnings = list(parsed_input.warnings) + list(index_bundle.warnings)
    if not chunks:
        warnings.append("The current settings produced zero chunks.")
    return PreparedPreview(
        parsed_input=parsed_input,
        cleaned_units=chunk_input_units,
        chunks=chunks,
        metadata_fields=metadata_fields,
        index_bundle=index_bundle,
        warnings=warnings,
    )


def run_preview(prepared: PreparedPreview, config: PipelineConfig) -> PreviewRun:
    original_query = config.query.strip()
    warnings: list[str] = []
    if not original_query:
        return PreviewRun(
            original_query="",
            rewritten_query="",
            pre_rerank_results=[],
            results=[],
            final_context="",
            final_answer=None,
            warnings=["No query was provided."],
        )

    rewritten_query, rewrite_warnings = rewrite_query(original_query, config.use_query_rewrite, config.llm_config)
    warnings.extend(rewrite_warnings)

    retrieval_outcome = retrieve(prepared.index_bundle, rewritten_query, config)
    warnings.extend(retrieval_outcome.warnings)
    pre_rerank_results = retrieval_outcome.results
    results = pre_rerank_results
    if config.use_rerank:
        results = rerank_results(pre_rerank_results, rewritten_query)

    retrieval_debug = retrieval_outcome.debug
    retrieval_debug.query_rewrite_applied = rewritten_query != original_query
    retrieval_debug.final_results = trace_items(results)

    final_context = assemble_context(results)
    final_answer, answer_warnings = generate_answer(
        original_query,
        final_context,
        config.generate_answer,
        config.llm_config,
    )
    warnings.extend(answer_warnings)

    return PreviewRun(
        original_query=original_query,
        rewritten_query=rewritten_query,
        pre_rerank_results=pre_rerank_results,
        results=results,
        final_context=final_context,
        final_answer=final_answer,
        warnings=warnings,
        retrieval_debug=retrieval_debug,
    )


def parsed_preview_rows(parsed_input: ParsedInput, limit: int = 12) -> list[dict[str, Any]]:
    return parsed_input.preview_rows[:limit]


def chunk_preview_rows(chunks: list[ChunkRecord], limit: int = DEFAULT_CHUNK_PREVIEW_LIMIT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks[:limit]:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_file": chunk.source_name,
                "page_number": chunk.page_number,
                "record_id": chunk.record_id,
                "text_id": chunk.text_id,
                "chars": len(chunk.text),
                "preview": preview_text(chunk.text, 220),
            }
        )
    return rows


def metadata_preview_rows(chunks: list[ChunkRecord], limit: int = DEFAULT_CHUNK_PREVIEW_LIMIT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks[:limit]:
        row = {"chunk_id": chunk.chunk_id}
        row.update(chunk.metadata)
        rows.append(row)
    return rows


def retrieval_preview_rows(results: list[RetrievedResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "rank": result.rank,
            "chunk_id": result.chunk.chunk_id,
            "score": round(result.score, 4),
            "vector_score": round(result.vector_score, 4) if result.vector_score is not None else None,
            "keyword_score": round(result.keyword_score, 4) if result.keyword_score is not None else None,
            "rerank_score": round(result.rerank_score, 4) if result.rerank_score is not None else None,
            "source_file": result.chunk.source_name,
            "page_number": result.chunk.page_number,
            "record_id": result.chunk.record_id,
            "text_id": result.chunk.text_id,
            "label": result.chunk.label,
            "record_type": result.chunk.record_type,
            "preview": preview_text(result.chunk.text, 220),
        }
        rows.append(row)
    return rows


def chunk_details(result: RetrievedResult) -> dict[str, Any]:
    payload = asdict(result.chunk)
    payload["score"] = result.score
    payload["vector_score"] = result.vector_score
    payload["keyword_score"] = result.keyword_score
    payload["rerank_score"] = result.rerank_score
    payload["rank"] = result.rank
    payload["pre_rerank_rank"] = result.pre_rerank_rank
    return payload


def _merge_structured_units(
    cleaned_units: list[DocumentUnit],
    parsed_input: ParsedInput,
    config: PipelineConfig,
) -> list[DocumentUnit]:
    if parsed_input.file_type == "pdf" or config.structured_record_mode:
        return cleaned_units
    if not cleaned_units:
        return []

    merged_text = "\n\n".join(f"[{unit.doc_id}]\n{unit.text}" for unit in cleaned_units if unit.text.strip())
    return [
        DocumentUnit(
            doc_id="merged_records",
            source_name=parsed_input.source_name,
            text=merged_text,
            unit_type="record",
            record_type="merged_records",
            source=parsed_input.source_name,
            raw_fields={"record_count": len(cleaned_units)},
        )
    ]
