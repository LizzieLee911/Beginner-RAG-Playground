from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.schemas import ChunkRecord, LLMConfig, PipelineConfig
from pipeline.utils import tokenize


@dataclass(slots=True)
class VectorIndex:
    chunk_ids: list[str]
    embeddings: np.ndarray
    backend_name: str
    backend: Any


@dataclass(slots=True)
class KeywordIndex:
    chunk_ids: list[str]
    bm25: BM25Okapi
    tokenized_texts: list[list[str]]


@dataclass(slots=True)
class IndexBundle:
    chunks: list[ChunkRecord]
    chunk_lookup: dict[str, ChunkRecord]
    vector_index: VectorIndex | None = None
    keyword_index: KeywordIndex | None = None
    warnings: list[str] = field(default_factory=list)
    vector_backend_used: str = "local_tfidf"


class LocalTfidfEmbeddingBackend:
    name = "local_tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.toarray().astype(np.float32)

    def transform_query(self, query: str) -> np.ndarray:
        matrix = self.vectorizer.transform([query])
        return matrix.toarray().astype(np.float32)[0]


class OpenAIEmbeddingBackend:
    name = "openai_compatible"

    def __init__(self, llm_config: LLMConfig) -> None:
        if not llm_config.api_key:
            raise ValueError("No API key provided for OpenAI-compatible embeddings.")
        from openai import OpenAI

        self.client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url or None)
        self.model = llm_config.embedding_model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._embed_many(texts)

    def transform_query(self, query: str) -> np.ndarray:
        return self._embed_many([query])[0]

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        embeddings: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return np.asarray(embeddings, dtype=np.float32)


def build_index_bundle(chunks: list[ChunkRecord], config: PipelineConfig) -> IndexBundle:
    bundle = IndexBundle(
        chunks=chunks,
        chunk_lookup={chunk.chunk_id: chunk for chunk in chunks},
        vector_backend_used=config.llm_config.embedding_backend,
    )
    if not chunks:
        bundle.warnings.append("No chunks were created, so no index was built.")
        return bundle

    if config.index_type in {"vector", "hybrid"}:
        vector_index, warning, backend_used = build_vector_index(chunks, config.llm_config)
        bundle.vector_index = vector_index
        bundle.vector_backend_used = backend_used
        if warning:
            bundle.warnings.append(warning)

    if config.index_type in {"keyword", "hybrid"}:
        bundle.keyword_index = build_keyword_index(chunks)

    return bundle


def build_vector_index(
    chunks: list[ChunkRecord],
    llm_config: LLMConfig,
) -> tuple[VectorIndex | None, str | None, str]:
    texts = [chunk.text for chunk in chunks]
    requested_backend = llm_config.embedding_backend
    if requested_backend == "openai_compatible":
        try:
            backend = OpenAIEmbeddingBackend(llm_config)
            embeddings = backend.fit_transform(texts)
            return (
                VectorIndex(
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    embeddings=embeddings,
                    backend_name=backend.name,
                    backend=backend,
                ),
                None,
                backend.name,
            )
        except Exception as exc:
            warning = f"OpenAI-compatible embeddings unavailable, fell back to local TF-IDF: {exc}"
            local_backend = LocalTfidfEmbeddingBackend()
            embeddings = local_backend.fit_transform(texts)
            return (
                VectorIndex(
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    embeddings=embeddings,
                    backend_name=local_backend.name,
                    backend=local_backend,
                ),
                warning,
                local_backend.name,
            )

    backend = LocalTfidfEmbeddingBackend()
    embeddings = backend.fit_transform(texts)
    return (
        VectorIndex(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            backend_name=backend.name,
            backend=backend,
        ),
        None,
        backend.name,
    )


def build_keyword_index(chunks: list[ChunkRecord]) -> KeywordIndex:
    tokenized_texts = [tokenize(chunk.text) or ["_empty_"] for chunk in chunks]
    bm25 = BM25Okapi(tokenized_texts)
    return KeywordIndex(
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        bm25=bm25,
        tokenized_texts=tokenized_texts,
    )
