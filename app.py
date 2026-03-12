from __future__ import annotations

import math
import os
from hashlib import md5

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline.loaders import detect_file_type, load_input
from pipeline.metadata_inference import suggest_structured_fields
from pipeline.preview import (
    chunk_details,
    chunk_preview_rows,
    metadata_preview_rows,
    parsed_preview_rows,
    prepare_preview_artifacts,
    retrieval_preview_rows,
    run_preview,
)
from pipeline.providers import provider_defaults
from pipeline.schemas import (
    LLMConfig,
    MetadataFilterConfig,
    PipelineConfig,
    RetrievalTraceItem,
    StructuredFieldSelection,
)
from pipeline.utils import extract_candidate_metadata_fields


load_dotenv()

st.set_page_config(page_title="RAG Preview", layout="wide")


def _apply_provider_defaults(preset: str) -> None:
    defaults = provider_defaults(preset)
    st.session_state["provider_base_url"] = defaults["base_url"]
    st.session_state["provider_chat_model"] = defaults["chat_model"]
    st.session_state["provider_embedding_model"] = defaults["embedding_model"]
    st.session_state["provider_defaults_applied"] = preset



def _trace_rows(items: list[RetrievalTraceItem]) -> list[dict[str, object]]:
    return [
        {
            "rank": item.rank,
            "chunk_id": item.chunk_id,
            "score": round(item.score, 4),
            "vector_score": round(item.vector_score, 4) if item.vector_score is not None else None,
            "keyword_score": round(item.keyword_score, 4) if item.keyword_score is not None else None,
            "source_file": item.source_file,
            "page_number": item.page_number,
            "record_id": item.record_id,
            "text_id": item.text_id,
            "preview": item.preview,
        }
        for item in items
    ]



def _valid_defaults(items: list[str], options: list[str]) -> list[str]:
    option_set = set(options)
    return [item for item in items if item in option_set]



def _selection_changed(current: StructuredFieldSelection, suggested: StructuredFieldSelection) -> bool:
    return (
        current.text_fields != suggested.text_fields
        or current.metadata_fields != suggested.metadata_fields
        or current.identifier_fields != suggested.identifier_fields
    )


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8f4ec 0%, #fcfbf8 100%);
        }
        [data-testid="stSidebar"] {
            background: #f1ede5;
        }
        .rag-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(125, 99, 63, 0.12);
            box-shadow: 0 10px 24px rgba(58, 45, 28, 0.06);
            margin-bottom: 1rem;
        }
        .rag-tag {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            margin-right: 0.35rem;
            border-radius: 999px;
            background: #dce9e2;
            color: #264336;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Preview")
st.caption(
    "A local-first Streamlit workbench for testing parsing, field selection, chunking, retrieval, reranking, provider settings, and optional answer generation."
)

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Upload PDF, JSON, or CSV", type=["pdf", "json", "csv"])

if uploaded_file is None:
    st.info("Upload a document to start building a preview pipeline.")
    st.markdown(
        """
        **Try the bundled samples**

        - `sample_data/medec_sample.json`
        - `sample_data/medec_sample.csv`
        - `sample_data/qa_map_sample.json`
        """
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
file_type = detect_file_type(uploaded_file.name)

provider_options = ["openai", "siliconflow", "custom"]
if "provider_preset" not in st.session_state:
    st.session_state["provider_preset"] = "openai"
if st.session_state.get("provider_defaults_applied") != st.session_state["provider_preset"]:
    _apply_provider_defaults(st.session_state["provider_preset"])

with st.sidebar:
    st.header("Provider")
    provider_preset = st.selectbox(
        "Provider preset",
        options=provider_options,
        index=provider_options.index(st.session_state["provider_preset"]),
        help="Presets only affect API-based features. Local preview still works without an API key.",
    )
    if provider_preset != st.session_state.get("provider_preset"):
        st.session_state["provider_preset"] = provider_preset
    if st.session_state.get("provider_defaults_applied") != provider_preset:
        _apply_provider_defaults(provider_preset)

    api_key = st.text_input(
        "OpenAI-compatible API key",
        value=os.getenv("OPENAI_API_KEY", st.session_state.get("provider_api_key", "")),
        type="password",
        help="Only required for metadata inference, query rewrite, final answer generation, and optional API embeddings.",
    )
    st.session_state["provider_api_key"] = api_key
    base_url = st.text_input("Base URL", value=st.session_state.get("provider_base_url", ""))
    chat_model = st.text_input("Chat model", value=st.session_state.get("provider_chat_model", ""))
    embedding_model = st.text_input("Embedding model", value=st.session_state.get("provider_embedding_model", ""))
    st.session_state["provider_base_url"] = base_url
    st.session_state["provider_chat_model"] = chat_model
    st.session_state["provider_embedding_model"] = embedding_model

with st.sidebar:
    st.header("Document Processing")
    cleaning_mode = st.selectbox(
        "Cleaning mode",
        options=["raw", "basic_clean"],
        format_func=lambda value: "raw" if value == "raw" else "basic clean",
    )
    chunk_strategy = st.selectbox(
        "Chunk strategy",
        options=["fixed_chunk", "sentence_window"],
        format_func=lambda value: "fixed chunk" if value == "fixed_chunk" else "sentence window",
    )
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1600, value=600, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=100, step=20)
    structured_record_mode = st.checkbox(
        "One record per document unit",
        value=True,
        disabled=file_type == "pdf",
        help="Disable this only when you explicitly want to merge all structured rows into one large document before chunking.",
    )
    pdf_parser_mode = "pypdf"
    if file_type == "pdf":
        pdf_parser_mode = st.selectbox(
            "PDF parser",
            options=["pypdf", "unstructured"],
            help="Unstructured is section-aware when available, and falls back gracefully if it fails.",
        )

base_llm_config = LLMConfig(
    api_key=api_key or None,
    base_url=base_url or None,
    embedding_model=embedding_model,
    chat_model=chat_model,
    embedding_backend="local_tfidf",
    provider_preset=provider_preset,
)

try:
    inspection = load_input(file_bytes, uploaded_file.name, pdf_parser_mode=pdf_parser_mode)
except Exception as exc:
    st.error(f"Failed to parse the uploaded file: {exc}")
    st.stop()

field_selection = inspection.field_selection or StructuredFieldSelection(source="none")
metadata_inference_warnings: list[str] = []
use_metadata_inference = False
if file_type in {"json", "csv"}:
    suggested_selection = field_selection
    with st.sidebar:
        st.header("Field Selection")
        use_metadata_inference = st.checkbox(
            "Auto metadata suggestion",
            value=bool(api_key),
            help="Rule-based field detection always runs. With an API key, the chat model can refine likely text, metadata, and identifier fields using the first 6000 characters.",
        )
    if use_metadata_inference and api_key:
        suggestion_key = md5(
            (
                uploaded_file.name
                + str(len(file_bytes))
                + provider_preset
                + (base_url or "")
                + chat_model
                + file_bytes[:6000].decode("utf-8", errors="ignore")
            ).encode("utf-8", errors="ignore")
        ).hexdigest()
        cache_key = "metadata_suggestion_cache"
        if st.session_state.get("metadata_suggestion_key") != suggestion_key:
            suggestion, metadata_inference_warnings = suggest_structured_fields(
                rows=[unit.raw_fields for unit in inspection.units],
                file_preview=file_bytes[:6000].decode("utf-8", errors="ignore"),
                llm_config=base_llm_config,
                use_llm=True,
            )
            st.session_state[cache_key] = suggestion
            st.session_state["metadata_suggestion_key"] = suggestion_key
        suggested_selection = st.session_state.get(cache_key, suggested_selection)
    elif use_metadata_inference and not api_key:
        metadata_inference_warnings.append("Auto metadata suggestion needs an API key. Rule-based field selection is being used.")

    available_fields = inspection.structured_fields
    suggested_selection = StructuredFieldSelection(
        text_fields=_valid_defaults(suggested_selection.text_fields, available_fields),
        metadata_fields=_valid_defaults(suggested_selection.metadata_fields, available_fields),
        identifier_fields=_valid_defaults(suggested_selection.identifier_fields, available_fields),
        source=suggested_selection.source,
        notes=list(suggested_selection.notes),
    )
    if not suggested_selection.text_fields and inspection.field_selection:
        suggested_selection.text_fields = _valid_defaults(inspection.field_selection.text_fields, available_fields)

    with st.sidebar:
        text_fields = st.multiselect(
            "Text fields",
            options=available_fields,
            default=suggested_selection.text_fields,
            help="Selected fields are concatenated into the text that gets chunked. Leave empty to fall back to joining non-identifier fields.",
        )
        identifier_fields = st.multiselect(
            "Identifier fields",
            options=available_fields,
            default=suggested_selection.identifier_fields,
            help="Identifier fields stay out of the fallback text join and remain available as metadata.",
        )
        metadata_defaults = [
            field for field in suggested_selection.metadata_fields if field in available_fields and field not in text_fields
        ]
        metadata_fields = st.multiselect(
            "Metadata fields",
            options=available_fields,
            default=metadata_defaults,
            help="These fields remain visible in chunk metadata when metadata mode is set to full.",
        )
        if not text_fields:
            st.caption("No text field selected. The loader will fall back to joining non-identifier fields into a text block per record.")

    identifier_fields = [field for field in identifier_fields if field not in text_fields]
    metadata_fields = [field for field in metadata_fields if field not in text_fields]
    field_selection = StructuredFieldSelection(
        text_fields=text_fields,
        metadata_fields=metadata_fields,
        identifier_fields=identifier_fields,
        source=suggested_selection.source,
        notes=list(suggested_selection.notes),
    )
    if _selection_changed(field_selection, suggested_selection):
        field_selection.source = "manual"
        if "Manual field selection override applied in the sidebar." not in field_selection.notes:
            field_selection.notes.append("Manual field selection override applied in the sidebar.")

common_filter_fields = extract_candidate_metadata_fields(
    [*inspection.structured_fields, *field_selection.metadata_fields, *field_selection.identifier_fields]
)

with st.sidebar:
    st.header("Metadata")
    metadata_mode = st.selectbox("Metadata mode", options=["none", "basic", "page_source", "full"])
    enable_filter = st.checkbox("Enable metadata filter", value=False)
    filter_field = st.selectbox("Filter field", options=common_filter_fields, disabled=not enable_filter)
    filter_operator = st.selectbox("Operator", options=["equals", "contains"], disabled=not enable_filter)
    filter_value = st.text_input("Filter value", disabled=not enable_filter)

    st.header("Index")
    index_type = st.selectbox("Index type", options=["vector", "keyword", "hybrid"])
    embedding_backend = "local_tfidf"
    if index_type in {"vector", "hybrid"}:
        embedding_backend = st.selectbox(
            "Vector backend",
            options=["local_tfidf", "openai_compatible"],
            format_func=lambda value: "local TF-IDF" if value == "local_tfidf" else "OpenAI-compatible embeddings",
            help="API embeddings are optional. Keep local TF-IDF selected if you only want local preview behavior.",
        )
    hybrid_vector_weight = 0.5
    if index_type == "hybrid":
        hybrid_vector_weight = st.slider("Hybrid vector weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.header("Retrieval")
    top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
    use_mmr = st.checkbox("Use MMR diversification", value=False, disabled=index_type == "keyword")
    use_rerank = st.checkbox("Use rerank", value=False)
    use_query_rewrite = st.checkbox("Use query rewrite", value=False)

    st.header("Query")
    query = st.text_area("Query text", value="", height=120)
    generate_answer = st.checkbox("Generate final answer", value=False)
    run_button = st.button("Run Preview", type="primary", use_container_width=True)

llm_config = LLMConfig(
    api_key=api_key or None,
    base_url=base_url or None,
    embedding_model=embedding_model,
    chat_model=chat_model,
    embedding_backend=embedding_backend,
    provider_preset=provider_preset,
)
config = PipelineConfig(
    cleaning_mode=cleaning_mode,
    chunk_strategy=chunk_strategy,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    structured_record_mode=structured_record_mode,
    metadata_mode=metadata_mode,
    index_type=index_type,
    hybrid_vector_weight=hybrid_vector_weight,
    top_k=top_k,
    use_mmr=use_mmr,
    use_rerank=use_rerank,
    use_query_rewrite=use_query_rewrite,
    metadata_filter=MetadataFilterConfig(
        enabled=enable_filter,
        field=filter_field if enable_filter else "",
        operator=filter_operator,
        value=filter_value,
    ),
    query=query,
    generate_answer=generate_answer,
    llm_config=llm_config,
    field_selection=field_selection,
    pdf_parser_mode=pdf_parser_mode,
    use_metadata_inference=use_metadata_inference,
)

try:
    prepared = prepare_preview_artifacts(file_bytes, uploaded_file.name, config)
except Exception as exc:
    st.error(f"Failed to prepare the pipeline preview: {exc}")
    st.stop()

summary_left, summary_right = st.columns([1.7, 1])
with summary_left:
    text_field_label = ", ".join(prepared.parsed_input.summary.get("detected_text_fields", []) or [])
    metadata_field_label = ", ".join(prepared.parsed_input.summary.get("detected_metadata_fields", []) or [])
    st.markdown(
        f"""
        <div class="rag-card">
            <div class="rag-tag">file: {prepared.parsed_input.file_type}</div>
            <div class="rag-tag">chunks: {len(prepared.chunks)}</div>
            <div class="rag-tag">index: {index_type}</div>
            <div class="rag-tag">vector backend: {prepared.index_bundle.vector_backend_used}</div>
            <div class="rag-tag">field source: {field_selection.source}</div>
            <p style="margin-top: 0.9rem; margin-bottom: 0.2rem; color: #4a4338;">
                Text fields: {text_field_label or 'fallback_join'}
            </p>
            <p style="margin-top: 0.2rem; margin-bottom: 0; color: #4a4338;">
                Metadata fields: {metadata_field_label or 'none detected'}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_right:
    stats = prepared.parsed_input.summary
    metric_cols = st.columns(2)
    metric_cols[0].metric("Document units", len(prepared.parsed_input.units))
    metric_cols[1].metric("Chunks", len(prepared.chunks))
    if prepared.parsed_input.file_type == "pdf":
        metric_cols = st.columns(2)
        metric_cols[0].metric("Pages", stats.get("page_count", 0))
        metric_cols[1].metric("Parser", stats.get("pdf_parser", "pypdf"))
    else:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Records", stats.get("record_count", 0))
        metric_cols[1].metric("Text fields", len(stats.get("detected_text_fields", [])))
        if len(prepared.cleaned_units) != len(prepared.parsed_input.units):
            st.caption(f"Chunk input units after merge: {len(prepared.cleaned_units)}")

for warning in metadata_inference_warnings + prepared.warnings:
    st.warning(warning)
for note in field_selection.notes:
    st.info(note)

preview_run = None
if run_button:
    with st.spinner("Running retrieval preview..."):
        preview_run = run_preview(prepared, config)
    for warning in preview_run.warnings:
        st.warning(warning)

parsed_tab, chunk_tab, metadata_tab, retrieval_tab, rerank_tab, debug_tab, context_tab, answer_tab = st.tabs(
    [
        "Parsed document preview",
        "Chunk preview",
        "Metadata preview",
        "Retrieval results",
        "Rerank comparison",
        "Debug",
        "Final context",
        "Final answer",
    ]
)

with parsed_tab:
    st.subheader("Parsed summary")
    st.json(prepared.parsed_input.summary)
    preview_df = pd.DataFrame(parsed_preview_rows(prepared.parsed_input))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    if prepared.parsed_input.structured_fields:
        st.caption("Detected structured fields: " + ", ".join(prepared.parsed_input.structured_fields))

with chunk_tab:
    st.subheader("Generated chunks")
    page_size = st.selectbox("Chunk page size", options=[25, 50, 100], index=2)
    total_pages = max(1, math.ceil(len(prepared.chunks) / page_size))
    current_page = st.number_input("Chunk page", min_value=1, max_value=total_pages, value=1, step=1)
    chunk_start = (current_page - 1) * page_size
    chunk_end = min(chunk_start + page_size, len(prepared.chunks))
    current_chunks = prepared.chunks[chunk_start:chunk_end]
    st.write({"total_chunks": len(prepared.chunks), "showing": f"{chunk_start + 1}-{chunk_end} of {len(prepared.chunks)}"})
    st.caption("Chunk preview is paginated. By default the first 100 chunks are visible on page 1.")
    chunk_df = pd.DataFrame(chunk_preview_rows(current_chunks, limit=len(current_chunks)))
    st.dataframe(chunk_df, use_container_width=True, hide_index=True)
    show_chunk_text = st.checkbox("Show chunk text for current page", value=False)
    if show_chunk_text:
        for chunk in current_chunks[:20]:
            with st.expander(f"{chunk.chunk_id} | {chunk.source_name}"):
                st.write(chunk.text)

with metadata_tab:
    st.subheader("Chunk metadata")
    metadata_page_size = st.selectbox("Metadata page size", options=[25, 50, 100], index=2)
    metadata_total_pages = max(1, math.ceil(len(prepared.chunks) / metadata_page_size))
    metadata_page = st.number_input("Metadata page", min_value=1, max_value=metadata_total_pages, value=1, step=1)
    metadata_start = (metadata_page - 1) * metadata_page_size
    metadata_end = min(metadata_start + metadata_page_size, len(prepared.chunks))
    st.write({"total_chunks": len(prepared.chunks), "showing": f"{metadata_start + 1}-{metadata_end} of {len(prepared.chunks)}"})
    metadata_df = pd.DataFrame(metadata_preview_rows(prepared.chunks[metadata_start:metadata_end], limit=metadata_page_size))
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

with retrieval_tab:
    if preview_run is None:
        st.info("Enter a query and click Run Preview to inspect retrieval outputs.")
    elif not preview_run.results:
        st.warning("No retrieval results were returned.")
    else:
        st.subheader("Retrieval inputs")
        st.write({"original_query": preview_run.original_query, "retrieval_query": preview_run.rewritten_query})
        result_df = pd.DataFrame(retrieval_preview_rows(preview_run.results))
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        for result in preview_run.results:
            title = f"Rank {result.rank} | {result.chunk.chunk_id} | score={result.score:.4f}"
            with st.expander(title):
                st.json(chunk_details(result))
                st.write(result.chunk.text)

with rerank_tab:
    if preview_run is None:
        st.info("Rerank comparison appears after a preview run.")
    elif not config.use_rerank:
        st.info("Enable rerank in the sidebar to compare before and after ordering.")
    else:
        before_col, after_col = st.columns(2)
        with before_col:
            st.subheader("Before rerank")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(preview_run.pre_rerank_results)), use_container_width=True, hide_index=True)
        with after_col:
            st.subheader("After rerank")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(preview_run.results)), use_container_width=True, hide_index=True)

with debug_tab:
    if preview_run is None:
        st.info("Run a preview to see retrieval debug details.")
    else:
        debug = preview_run.retrieval_debug
        st.write(
            {
                "total_indexed_chunks": debug.total_indexed_chunks,
                "eligible_chunk_count": debug.eligible_chunk_count,
                "index_type": debug.index_type,
                "vector_backend": debug.vector_backend,
                "query_rewrite_applied": debug.query_rewrite_applied,
                "retrieval_query": debug.retrieval_query,
                "active_metadata_filter": debug.active_metadata_filter,
                "filter_warning": debug.filter_warning,
                "vector_zero_query": debug.vector_zero_query,
                "keyword_zero_query": debug.keyword_zero_query,
                "rerank_applied": config.use_rerank,
                "rebuild_policy": "The preview pipeline is rebuilt on every Streamlit rerun, so uploading a new file or changing settings rebuilds the index.",
            }
        )
        st.subheader("Raw vector results")
        st.dataframe(pd.DataFrame(_trace_rows(debug.vector_raw_results)), use_container_width=True, hide_index=True)
        st.subheader("Raw keyword results")
        st.dataframe(pd.DataFrame(_trace_rows(debug.keyword_raw_results)), use_container_width=True, hide_index=True)
        st.subheader("Raw merged results")
        st.dataframe(pd.DataFrame(_trace_rows(debug.merged_raw_results)), use_container_width=True, hide_index=True)
        if config.use_rerank:
            st.subheader("Reranked results")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(preview_run.results)), use_container_width=True, hide_index=True)
        else:
            st.subheader("Final retrieval results")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(preview_run.results)), use_container_width=True, hide_index=True)

with context_tab:
    if preview_run is None:
        st.info("The assembled context block will appear here after a preview run.")
    else:
        st.code(preview_run.final_context or "", language="markdown")

with answer_tab:
    if preview_run is None:
        st.info("Optional final answer output appears here after a preview run.")
    elif preview_run.final_answer:
        st.subheader("Generated answer")
        st.write(preview_run.final_answer)
    elif config.generate_answer:
        st.warning("Answer generation was enabled, but no answer was produced. Check warnings above.")
    else:
        st.info("Enable final answer generation in the sidebar to call the chat model.")
