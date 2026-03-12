from __future__ import annotations

from pipeline.schemas import RetrievedResult


def assemble_context(results: list[RetrievedResult]) -> str:
    blocks: list[str] = []
    for result in results:
        metadata_parts = [
            f"chunk_id={result.chunk.chunk_id}",
            f"score={result.score:.4f}",
        ]
        if result.vector_score is not None:
            metadata_parts.append(f"vector_score={result.vector_score:.4f}")
        if result.keyword_score is not None:
            metadata_parts.append(f"keyword_score={result.keyword_score:.4f}")
        if result.chunk.source_name:
            metadata_parts.append(f"source_file={result.chunk.source_name}")
        if result.chunk.page_number is not None:
            metadata_parts.append(f"page_number={result.chunk.page_number}")
        if result.chunk.record_id:
            metadata_parts.append(f"record_id={result.chunk.record_id}")
        if result.chunk.text_id:
            metadata_parts.append(f"text_id={result.chunk.text_id}")

        block = "\n".join(
            [
                f"[Retrieved Rank {result.rank}]",
                "Metadata: " + ", ".join(metadata_parts),
                "Text:",
                result.chunk.text,
            ]
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)
