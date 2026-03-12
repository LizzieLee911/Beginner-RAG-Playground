from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


CleaningMode = Literal["raw", "basic_clean"]
ChunkStrategy = Literal["fixed_chunk", "sentence_window"]
MetadataMode = Literal["none", "basic", "page_source", "full"]
IndexType = Literal["vector", "keyword", "hybrid"]
EmbeddingBackend = Literal["local_tfidf", "openai_compatible"]
FilterOperator = Literal["equals", "contains"]
ProviderPreset = Literal["openai", "siliconflow", "custom"]
PDFParserMode = Literal["pypdf", "unstructured"]
InferenceSource = Literal["none", "rule", "llm", "manual"]


@dataclass(slots=True)
class StructuredFieldSelection:
    text_fields: list[str] = field(default_factory=list)
    metadata_fields: list[str] = field(default_factory=list)
    identifier_fields: list[str] = field(default_factory=list)
    source: InferenceSource = "none"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentUnit:
    doc_id: str
    source_name: str
    text: str
    unit_type: Literal["page", "record"]
    page_number: int | None = None
    section: str | None = None
    record_id: str | None = None
    text_id: str | None = None
    label: str | None = None
    record_type: str | None = None
    source: str | None = None
    raw_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedInput:
    source_name: str
    file_type: Literal["pdf", "json", "csv"]
    units: list[DocumentUnit]
    preview_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    detected_text_field: str | None = None
    structured_fields: list[str] = field(default_factory=list)
    field_selection: StructuredFieldSelection | None = None


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    source_name: str
    unit_type: Literal["page", "record"]
    start_char: int | None = None
    end_char: int | None = None
    page_number: int | None = None
    section: str | None = None
    record_id: str | None = None
    text_id: str | None = None
    label: str | None = None
    record_type: str | None = None
    source: str | None = None
    raw_fields: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetadataFilterConfig:
    enabled: bool = False
    field: str = ""
    operator: FilterOperator = "equals"
    value: str = ""


@dataclass(slots=True)
class LLMConfig:
    api_key: str | None = None
    base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_backend: EmbeddingBackend = "local_tfidf"
    provider_preset: ProviderPreset = "openai"


@dataclass(slots=True)
class PipelineConfig:
    cleaning_mode: CleaningMode = "basic_clean"
    chunk_strategy: ChunkStrategy = "fixed_chunk"
    chunk_size: int = 600
    chunk_overlap: int = 100
    structured_record_mode: bool = True
    metadata_mode: MetadataMode = "basic"
    index_type: IndexType = "hybrid"
    hybrid_vector_weight: float = 0.5
    top_k: int = 5
    use_mmr: bool = False
    use_rerank: bool = False
    use_query_rewrite: bool = False
    metadata_filter: MetadataFilterConfig = field(default_factory=MetadataFilterConfig)
    query: str = ""
    generate_answer: bool = False
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    field_selection: StructuredFieldSelection | None = None
    pdf_parser_mode: PDFParserMode = "pypdf"
    use_metadata_inference: bool = False


@dataclass(slots=True)
class RetrievedResult:
    chunk: ChunkRecord
    rank: int
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    pre_rerank_rank: int | None = None


@dataclass(slots=True)
class RetrievalTraceItem:
    rank: int
    chunk_id: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    source_file: str | None = None
    page_number: int | None = None
    record_id: str | None = None
    text_id: str | None = None
    preview: str = ""


@dataclass(slots=True)
class RetrievalDebugInfo:
    total_indexed_chunks: int = 0
    eligible_chunk_count: int = 0
    index_type: str = ""
    vector_backend: str = ""
    query_rewrite_applied: bool = False
    retrieval_query: str = ""
    active_metadata_filter: dict[str, Any] | None = None
    filter_warning: str | None = None
    vector_zero_query: bool = False
    keyword_zero_query: bool = False
    vector_raw_results: list[RetrievalTraceItem] = field(default_factory=list)
    keyword_raw_results: list[RetrievalTraceItem] = field(default_factory=list)
    merged_raw_results: list[RetrievalTraceItem] = field(default_factory=list)
    final_results: list[RetrievalTraceItem] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalOutcome:
    results: list[RetrievedResult]
    warnings: list[str] = field(default_factory=list)
    debug: RetrievalDebugInfo = field(default_factory=RetrievalDebugInfo)


@dataclass(slots=True)
class PreparedPreview:
    parsed_input: ParsedInput
    cleaned_units: list[DocumentUnit]
    chunks: list[ChunkRecord]
    metadata_fields: list[str]
    index_bundle: Any
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PreviewRun:
    original_query: str
    rewritten_query: str
    pre_rerank_results: list[RetrievedResult]
    results: list[RetrievedResult]
    final_context: str
    final_answer: str | None
    warnings: list[str] = field(default_factory=list)
    retrieval_debug: RetrievalDebugInfo = field(default_factory=RetrievalDebugInfo)
