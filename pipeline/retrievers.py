from __future__ import annotations

from dataclasses import replace

import numpy as np

from pipeline.indexers import IndexBundle, KeywordIndex, VectorIndex
from pipeline.schemas import (
    ChunkRecord,
    MetadataFilterConfig,
    PipelineConfig,
    RetrievalDebugInfo,
    RetrievalOutcome,
    RetrievedResult,
    RetrievalTraceItem,
)
from pipeline.utils import preview_text, tokenize


RAW_DEBUG_LIMIT = 12
ZERO_EPSILON = 1e-12


def retrieve(bundle: IndexBundle, query: str, config: PipelineConfig) -> RetrievalOutcome:
    warnings: list[str] = []
    filtered_chunks, filter_warning = apply_metadata_filter(bundle.chunks, config.metadata_filter)
    debug = RetrievalDebugInfo(
        total_indexed_chunks=len(bundle.chunks),
        eligible_chunk_count=len(filtered_chunks),
        index_type=config.index_type,
        vector_backend=bundle.vector_backend_used,
        retrieval_query=query,
        active_metadata_filter=_active_filter_payload(config.metadata_filter),
        filter_warning=filter_warning,
    )
    if filter_warning:
        warnings.append(filter_warning)

    eligible_ids = {chunk.chunk_id for chunk in filtered_chunks}
    if not eligible_ids:
        warnings.append("Metadata filter excluded all chunks.")
        return RetrievalOutcome(results=[], warnings=warnings, debug=debug)

    if config.index_type == "vector":
        if bundle.vector_index is None:
            warnings.append("Vector index is unavailable.")
            return RetrievalOutcome(results=[], warnings=warnings, debug=debug)
        vector_results, vector_raw_results, zero_query = vector_search(
            bundle.vector_index,
            bundle.chunk_lookup,
            query,
            config.top_k,
            eligible_ids,
            config.use_mmr,
        )
        debug.vector_raw_results = trace_items(vector_raw_results)
        debug.vector_zero_query = zero_query
        debug.final_results = trace_items(vector_results)
        if zero_query:
            warnings.append("The vector query had no usable terms for the current index, so no vector results were returned.")
        return RetrievalOutcome(results=vector_results, warnings=warnings, debug=debug)

    if config.index_type == "keyword":
        if bundle.keyword_index is None:
            warnings.append("Keyword index is unavailable.")
            return RetrievalOutcome(results=[], warnings=warnings, debug=debug)
        keyword_results, keyword_raw_results, zero_query = keyword_search(
            bundle.keyword_index,
            bundle.chunk_lookup,
            query,
            config.top_k,
            eligible_ids,
        )
        debug.keyword_raw_results = trace_items(keyword_raw_results)
        debug.keyword_zero_query = zero_query
        debug.final_results = trace_items(keyword_results)
        if zero_query:
            warnings.append("The keyword query had no lexical matches in the indexed chunks.")
        return RetrievalOutcome(results=keyword_results, warnings=warnings, debug=debug)

    if bundle.vector_index is None and bundle.keyword_index is None:
        warnings.append("Neither vector nor keyword index is available.")
        return RetrievalOutcome(results=[], warnings=warnings, debug=debug)

    results, vector_raw_results, keyword_raw_results, merged_raw_results, vector_zero_query, keyword_zero_query = hybrid_search(
        bundle.vector_index,
        bundle.keyword_index,
        bundle.chunk_lookup,
        query,
        config.top_k,
        eligible_ids,
        config.hybrid_vector_weight,
        config.use_mmr,
    )
    debug.vector_raw_results = trace_items(vector_raw_results)
    debug.keyword_raw_results = trace_items(keyword_raw_results)
    debug.merged_raw_results = trace_items(merged_raw_results)
    debug.vector_zero_query = vector_zero_query
    debug.keyword_zero_query = keyword_zero_query
    debug.final_results = trace_items(results)
    if not results and (vector_zero_query or keyword_zero_query):
        warnings.append("No hybrid results were returned because the query did not produce non-zero raw retrieval scores.")
    return RetrievalOutcome(results=results, warnings=warnings, debug=debug)


def apply_metadata_filter(
    chunks: list[ChunkRecord],
    filter_config: MetadataFilterConfig,
) -> tuple[list[ChunkRecord], str | None]:
    if not filter_config.enabled or not filter_config.field.strip() or not filter_config.value.strip():
        return chunks, None

    field_name = filter_config.field.strip()
    if not any(field_name in chunk.metadata or hasattr(chunk, field_name) for chunk in chunks):
        return chunks, f"Metadata field '{field_name}' was not found, so the filter was skipped."

    matched_chunks: list[ChunkRecord] = []
    for chunk in chunks:
        raw_value = chunk.metadata.get(field_name)
        if raw_value is None and hasattr(chunk, field_name):
            raw_value = getattr(chunk, field_name)
        if raw_value is None:
            continue

        left = str(raw_value).lower()
        right = filter_config.value.strip().lower()
        if filter_config.operator == "contains" and right in left:
            matched_chunks.append(chunk)
        elif filter_config.operator == "equals" and left == right:
            matched_chunks.append(chunk)

    return matched_chunks, None


def vector_search(
    vector_index: VectorIndex,
    chunk_lookup: dict[str, ChunkRecord],
    query: str,
    top_k: int,
    eligible_ids: set[str],
    use_mmr: bool,
) -> tuple[list[RetrievedResult], list[RetrievedResult], bool]:
    query_embedding = vector_index.backend.transform_query(query)
    if float(np.linalg.norm(query_embedding)) <= ZERO_EPSILON:
        return [], [], True

    scores = cosine_scores(vector_index.embeddings, query_embedding)
    candidate_indices = [index for index, chunk_id in enumerate(vector_index.chunk_ids) if chunk_id in eligible_ids]
    if not candidate_indices:
        return [], [], False

    ranked_indices = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)
    raw_results = _results_from_indices(vector_index.chunk_ids, chunk_lookup, ranked_indices[:RAW_DEBUG_LIMIT], scores, score_type="vector")
    if max((scores[index] for index in candidate_indices), default=0.0) <= ZERO_EPSILON:
        return [], raw_results, False

    if use_mmr and len(ranked_indices) > 1:
        ranked_indices = maximal_marginal_relevance(
            embeddings=vector_index.embeddings,
            query_embedding=query_embedding,
            candidate_indices=ranked_indices[: max(top_k * 4, top_k)],
            top_k=top_k,
        )
    else:
        ranked_indices = ranked_indices[:top_k]

    results = _results_from_indices(vector_index.chunk_ids, chunk_lookup, ranked_indices, scores, score_type="vector")
    return results, raw_results, False


def keyword_search(
    keyword_index: KeywordIndex,
    chunk_lookup: dict[str, ChunkRecord],
    query: str,
    top_k: int,
    eligible_ids: set[str],
) -> tuple[list[RetrievedResult], list[RetrievedResult], bool]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return [], [], True

    scores = keyword_index.bm25.get_scores(query_tokens)
    ranked_indices = [index for index, chunk_id in enumerate(keyword_index.chunk_ids) if chunk_id in eligible_ids]
    ranked_indices.sort(key=lambda index: scores[index], reverse=True)
    raw_results = _results_from_indices(keyword_index.chunk_ids, chunk_lookup, ranked_indices[:RAW_DEBUG_LIMIT], scores, score_type="keyword")
    if max((scores[index] for index in ranked_indices), default=0.0) <= ZERO_EPSILON:
        return [], raw_results, True

    results = _results_from_indices(keyword_index.chunk_ids, chunk_lookup, ranked_indices[:top_k], scores, score_type="keyword")
    return results, raw_results, False


def hybrid_search(
    vector_index: VectorIndex | None,
    keyword_index: KeywordIndex | None,
    chunk_lookup: dict[str, ChunkRecord],
    query: str,
    top_k: int,
    eligible_ids: set[str],
    vector_weight: float,
    use_mmr: bool,
) -> tuple[list[RetrievedResult], list[RetrievedResult], list[RetrievedResult], list[RetrievedResult], bool, bool]:
    vector_results: list[RetrievedResult] = []
    vector_raw_results: list[RetrievedResult] = []
    keyword_results: list[RetrievedResult] = []
    keyword_raw_results: list[RetrievedResult] = []
    vector_zero_query = False
    keyword_zero_query = False

    if vector_index is not None:
        vector_results, vector_raw_results, vector_zero_query = vector_search(
            vector_index,
            chunk_lookup,
            query,
            max(top_k * 4, top_k),
            eligible_ids,
            use_mmr,
        )
    if keyword_index is not None:
        keyword_results, keyword_raw_results, keyword_zero_query = keyword_search(
            keyword_index,
            chunk_lookup,
            query,
            max(top_k * 4, top_k),
            eligible_ids,
        )

    vector_scores = {result.chunk.chunk_id: result.vector_score or 0.0 for result in vector_raw_results if (result.vector_score or 0.0) > ZERO_EPSILON}
    keyword_scores = {result.chunk.chunk_id: result.keyword_score or 0.0 for result in keyword_raw_results if (result.keyword_score or 0.0) > ZERO_EPSILON}
    if not vector_scores and not keyword_scores:
        return [], vector_raw_results, keyword_raw_results, [], vector_zero_query, keyword_zero_query

    normalized_vector = normalize_score_map(vector_scores)
    normalized_keyword = normalize_score_map(keyword_scores)
    merged_ids = set(vector_scores) | set(keyword_scores)
    merged_results: list[RetrievedResult] = []
    for chunk_id in merged_ids:
        vector_score = vector_scores.get(chunk_id)
        keyword_score = keyword_scores.get(chunk_id)
        merged_score = vector_weight * normalized_vector.get(chunk_id, 0.0) + (1.0 - vector_weight) * normalized_keyword.get(chunk_id, 0.0)
        merged_results.append(
            RetrievedResult(
                chunk=chunk_lookup[chunk_id],
                rank=0,
                score=float(merged_score),
                vector_score=vector_score,
                keyword_score=keyword_score,
            )
        )

    merged_results.sort(key=lambda result: result.score, reverse=True)
    merged_raw_results = [replace(result, rank=index) for index, result in enumerate(merged_results[:RAW_DEBUG_LIMIT], start=1)]
    final_results = [replace(result, rank=index) for index, result in enumerate(merged_results[:top_k], start=1)]
    return final_results, vector_raw_results, keyword_raw_results, merged_raw_results, vector_zero_query, keyword_zero_query


def _results_from_indices(
    chunk_ids: list[str],
    chunk_lookup: dict[str, ChunkRecord],
    indices: list[int],
    scores: np.ndarray,
    score_type: str,
) -> list[RetrievedResult]:
    results: list[RetrievedResult] = []
    for rank, index in enumerate(indices, start=1):
        chunk_id = chunk_ids[index]
        score = float(scores[index])
        result = RetrievedResult(
            chunk=chunk_lookup[chunk_id],
            rank=rank,
            score=score,
        )
        if score_type == "vector":
            result.vector_score = score
        else:
            result.keyword_score = score
        results.append(result)
    return results


def trace_items(results: list[RetrievedResult]) -> list[RetrievalTraceItem]:
    return [
        RetrievalTraceItem(
            rank=result.rank,
            chunk_id=result.chunk.chunk_id,
            score=result.score,
            vector_score=result.vector_score,
            keyword_score=result.keyword_score,
            source_file=result.chunk.source_name,
            page_number=result.chunk.page_number,
            record_id=result.chunk.record_id,
            text_id=result.chunk.text_id,
            preview=preview_text(result.chunk.text, 160),
        )
        for result in results
    ]


def _active_filter_payload(filter_config: MetadataFilterConfig) -> dict[str, str] | None:
    if not filter_config.enabled or not filter_config.field.strip() or not filter_config.value.strip():
        return None
    return {
        "field": filter_config.field,
        "operator": filter_config.operator,
        "value": filter_config.value,
    }


def cosine_scores(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(embeddings, axis=1)
    denominator = np.clip(doc_norms * max(query_norm, ZERO_EPSILON), ZERO_EPSILON, None)
    return np.dot(embeddings, query_embedding) / denominator


def maximal_marginal_relevance(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    candidate_indices: list[int],
    top_k: int,
    lambda_weight: float = 0.7,
) -> list[int]:
    if not candidate_indices:
        return []

    candidate_matrix = embeddings[candidate_indices]
    query_scores = cosine_scores(candidate_matrix, query_embedding)
    selected_positions: list[int] = [int(np.argmax(query_scores))]
    remaining_positions = set(range(len(candidate_indices))) - set(selected_positions)

    while remaining_positions and len(selected_positions) < top_k:
        best_position = None
        best_score = float("-inf")
        for position in remaining_positions:
            relevance = query_scores[position]
            diversity = max(
                cosine_scores(candidate_matrix[[position]], candidate_matrix[selected_position])[0]
                for selected_position in selected_positions
            )
            mmr_score = lambda_weight * relevance - (1.0 - lambda_weight) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_position = position

        if best_position is None:
            break
        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    return [candidate_indices[position] for position in selected_positions[:top_k]]


def normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    if max_score <= ZERO_EPSILON:
        return {key: 0.0 for key in score_map}
    return {key: value / max_score for key, value in score_map.items()}
