from __future__ import annotations

from dataclasses import replace

from pipeline.schemas import RetrievedResult
from pipeline.utils import tokenize


def rerank_results(results: list[RetrievedResult], query: str) -> list[RetrievedResult]:
    if not results:
        return []

    query_tokens = set(tokenize(query))
    max_base_score = max((result.score for result in results), default=1.0) or 1.0

    reranked: list[RetrievedResult] = []
    for original in results:
        text_tokens = set(tokenize(original.chunk.text))
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        metadata_blob = " ".join(str(value) for value in original.chunk.metadata.values()).lower()
        metadata_overlap = sum(1 for token in query_tokens if token in metadata_blob) / max(len(query_tokens), 1)
        exact_bonus = 0.15 if query.lower() in original.chunk.text.lower() else 0.0
        base_component = original.score / max_base_score
        rerank_score = 0.55 * base_component + 0.3 * overlap + 0.1 * metadata_overlap + exact_bonus
        reranked.append(
            replace(
                original,
                rerank_score=float(rerank_score),
                pre_rerank_rank=original.rank,
            )
        )

    reranked.sort(key=lambda result: result.rerank_score or 0.0, reverse=True)
    return [replace(result, rank=index) for index, result in enumerate(reranked, start=1)]
