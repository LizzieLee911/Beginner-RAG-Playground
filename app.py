from __future__ import annotations

import json
import math
import os
from hashlib import md5
from pathlib import Path
from typing import Any

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


PREVIEW_STATE_KEYS = [
    "preview_prepared",
    "preview_run",
    "preview_field_selection",
    "preview_warnings",
    "preview_notes",
    "preview_signature",
    "preview_file_key",
]

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_FILES = {
    "MEDEC JSON": BASE_DIR / "sample_data" / "medec_sample.json",
    "MEDEC CSV": BASE_DIR / "sample_data" / "medec_sample.csv",
    "QA Map JSON": BASE_DIR / "sample_data" / "qa_map_sample.json",
}

load_dotenv()

st.set_page_config(page_title="Drag & RAG", layout="wide")


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



def _file_fingerprint(filename: str, file_bytes: bytes) -> str:
    hasher = md5()
    hasher.update(filename.encode("utf-8", errors="ignore"))
    hasher.update(str(len(file_bytes)).encode("utf-8"))
    hasher.update(file_bytes[:4096])
    hasher.update(file_bytes[-4096:])
    return hasher.hexdigest()



def _reset_preview_state() -> None:
    for key in PREVIEW_STATE_KEYS:
        st.session_state.pop(key, None)



def _selection_to_payload(selection: StructuredFieldSelection) -> dict[str, Any]:
    return {
        "text_fields": selection.text_fields,
        "metadata_fields": selection.metadata_fields,
        "identifier_fields": selection.identifier_fields,
        "source": selection.source,
    }



def _build_request_signature(file_key: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps({"file_key": file_key, **payload}, sort_keys=True, ensure_ascii=True)
    return md5(serialized.encode("utf-8")).hexdigest()



def _resolve_structured_selection(
    base_selection: StructuredFieldSelection,
    manual_selection: StructuredFieldSelection,
    manual_controls_available: bool,
    available_fields: list[str],
    use_metadata_inference: bool,
    llm_config: LLMConfig,
    rows: list[dict[str, Any]],
    file_preview: str,
) -> tuple[StructuredFieldSelection, list[str]]:
    warnings: list[str] = []
    suggested_selection = base_selection
    if use_metadata_inference and llm_config.api_key:
        suggested_selection, inference_warnings = suggest_structured_fields(
            rows=rows,
            file_preview=file_preview,
            llm_config=llm_config,
            use_llm=True,
        )
        warnings.extend(inference_warnings)
    elif use_metadata_inference and not llm_config.api_key:
        warnings.append("Auto metadata suggestion was enabled, but no API key was provided. Rule-based field detection was used.")

    suggested_selection = StructuredFieldSelection(
        text_fields=_valid_defaults(suggested_selection.text_fields, available_fields),
        metadata_fields=_valid_defaults(suggested_selection.metadata_fields, available_fields),
        identifier_fields=_valid_defaults(suggested_selection.identifier_fields, available_fields),
        source=suggested_selection.source,
        notes=list(suggested_selection.notes),
    )
    if not suggested_selection.text_fields and base_selection.text_fields:
        suggested_selection.text_fields = _valid_defaults(base_selection.text_fields, available_fields)

    effective_selection = suggested_selection
    if manual_controls_available:
        manual_text_fields = _valid_defaults(manual_selection.text_fields, available_fields)
        manual_identifier_fields = [
            field for field in _valid_defaults(manual_selection.identifier_fields, available_fields) if field not in manual_text_fields
        ]
        manual_metadata_fields = [
            field for field in _valid_defaults(manual_selection.metadata_fields, available_fields) if field not in manual_text_fields
        ]
        manual = StructuredFieldSelection(
            text_fields=manual_text_fields,
            metadata_fields=manual_metadata_fields,
            identifier_fields=manual_identifier_fields,
            source=suggested_selection.source,
            notes=list(suggested_selection.notes),
        )
        if _selection_changed(manual, suggested_selection):
            manual.source = "manual"
            if "Manual field selection override applied in the sidebar." not in manual.notes:
                manual.notes.append("Manual field selection override applied in the sidebar.")
            effective_selection = manual
        else:
            effective_selection = manual

    return effective_selection, warnings


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

st.markdown('<h1 style="font-size: 3.2rem; line-height: 1; margin-bottom: 0.35rem; color: #1f2432;">Drag & RAG</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style=\"font-size: 1.12rem; color: #1f2432; margin-top: 0; margin-bottom: 1.3rem;\">"
    "Upload a document, experiment with chunking and retrieval strategies, "
    "and quickly preview how your RAG pipeline behaves."
    "</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Upload PDF, JSON, or CSV", type=["pdf", "json", "csv"])

if uploaded_file is not None and st.session_state.get("selected_sample_name"):
    st.session_state.pop("selected_sample_name", None)

selected_sample_name = st.session_state.get("selected_sample_name")
if uploaded_file is None and selected_sample_name:
    sample_path = SAMPLE_FILES.get(selected_sample_name)
    if sample_path is None or not sample_path.exists():
        st.session_state.pop("selected_sample_name", None)
        st.error("The selected bundled sample is unavailable.")
        st.stop()
    active_filename = sample_path.name
    file_bytes = sample_path.read_bytes()
    st.info(f"Using bundled sample: {active_filename}")
    if st.button("Clear bundled sample"):
        st.session_state.pop("selected_sample_name", None)
        st.rerun()
elif uploaded_file is None:
    st.info("Upload a document to start building a preview pipeline.")
    st.subheader("Try the bundled samples")
    sample_columns = st.columns(len(SAMPLE_FILES))
    for column, (sample_label, sample_path) in zip(sample_columns, SAMPLE_FILES.items()):
        with column:
            if st.button(sample_label, use_container_width=True):
                st.session_state["selected_sample_name"] = sample_label
                st.rerun()
            st.caption(sample_path.name)
    st.stop()
else:
    active_filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

file_type = detect_file_type(active_filename)
file_key = _file_fingerprint(active_filename, file_bytes)
if st.session_state.get("active_file_key") != file_key:
    _reset_preview_state()
    st.session_state["active_file_key"] = file_key

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
        help="Only required for API-based metadata suggestion, query rewrite, final answer generation, and optional embedding generation.",
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
        "IMPORTANT: keep each structured record separate before chunking (recommended for JSON/CSV)",
        value=True,
        disabled=file_type == "pdf",
        help=(
            "Use this for most JSON and CSV datasets. Each row or record stays isolated, so its metadata remains attached to its own chunks instead of mixing with neighboring records. "
            "Example: if three rows describe three different questions, this setting keeps each question and its label separate during chunking. "
            "Turn it off only when you intentionally want to treat the whole structured file as one long document. "
            "Disabling it can merge records together and make retrieval previews or metadata inspection harder to interpret."
        ),
    )
    pdf_parser_mode = "pypdf"
    if file_type == "pdf":
        pdf_parser_mode = st.selectbox(
            "PDF parser",
            options=["pypdf", "unstructured"],
            help="Unstructured is section-aware when available, and falls back gracefully if it fails.",
        )

stored_prepared = st.session_state.get("preview_prepared")
stored_preview_run = st.session_state.get("preview_run")
stored_field_selection = st.session_state.get("preview_field_selection", StructuredFieldSelection(source="none"))
stored_warnings = st.session_state.get("preview_warnings", [])
stored_notes = st.session_state.get("preview_notes", [])
stored_signature = st.session_state.get("preview_signature")
if st.session_state.get("preview_file_key") != file_key:
    stored_prepared = None
    stored_preview_run = None
    stored_field_selection = StructuredFieldSelection(source="none")
    stored_warnings = []
    stored_notes = []
    stored_signature = None

available_fields = stored_prepared.parsed_input.structured_fields if stored_prepared and file_type in {"json", "csv"} else []
selection_defaults = stored_field_selection if stored_field_selection else StructuredFieldSelection(source="none")
selection_defaults = StructuredFieldSelection(
    text_fields=_valid_defaults(selection_defaults.text_fields, available_fields),
    metadata_fields=_valid_defaults(selection_defaults.metadata_fields, available_fields),
    identifier_fields=_valid_defaults(selection_defaults.identifier_fields, available_fields),
    source=selection_defaults.source,
    notes=list(selection_defaults.notes),
)

use_metadata_inference = False
text_fields: list[str] = []
identifier_fields: list[str] = []
metadata_fields: list[str] = []
if file_type in {"json", "csv"}:
    with st.sidebar:
        st.header("Field Selection")
        use_metadata_inference = st.checkbox(
            "IMPORTANT: use API to auto-suggest text and metadata fields",
            value=False,
            help=(
                "This sends a preview of the uploaded structured file to the selected chat model and can incur API cost. "
                "Use it when the JSON or CSV schema is unclear and you want help deciding which fields are text versus metadata. "
                "Example: the model may suggest `question` and `answer` as text fields, while keeping `label` and `source` as metadata. "
                "If this option is off, the app uses local rule-based field detection only. "
                "The API call happens only when you click Run Preview."
            ),
        )
        if available_fields:
            text_fields = st.multiselect(
                "Text fields",
                options=available_fields,
                default=selection_defaults.text_fields,
                help="Selected fields are concatenated into the text that gets chunked. Leave empty to fall back to joining non-identifier fields after the next Run Preview.",
            )
            identifier_fields = st.multiselect(
                "Identifier fields",
                options=available_fields,
                default=selection_defaults.identifier_fields,
                help="Identifier fields stay out of the fallback text join and remain available as metadata after the next Run Preview.",
            )
            metadata_defaults = [
                field for field in selection_defaults.metadata_fields if field in available_fields and field not in text_fields
            ]
            metadata_fields = st.multiselect(
                "Metadata fields",
                options=available_fields,
                default=metadata_defaults,
                help="These fields remain visible in chunk metadata when metadata mode is set to full after the next Run Preview.",
            )
        else:
            st.caption(
                "Field lists appear after the first Run Preview. The first run uses rule-based field detection, or the API-based suggestion if you enable it here and then click Run Preview."
            )

manual_field_selection = StructuredFieldSelection(
    text_fields=text_fields,
    metadata_fields=[field for field in metadata_fields if field not in text_fields],
    identifier_fields=[field for field in identifier_fields if field not in text_fields],
    source=selection_defaults.source,
    notes=list(selection_defaults.notes),
)

common_filter_fields = extract_candidate_metadata_fields(
    [*available_fields, *selection_defaults.metadata_fields, *selection_defaults.identifier_fields]
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
            help="API embeddings are optional and now run only when you click Run Preview.",
        )
    hybrid_vector_weight = 0.5
    if index_type == "hybrid":
        hybrid_vector_weight = st.slider("Hybrid vector weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.header("Retrieval")
    top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
    use_mmr = st.checkbox("Use MMR diversification", value=False, disabled=index_type == "keyword")
    use_rerank = st.checkbox("Use rerank", value=False)
    use_query_rewrite = st.checkbox("Use query rewrite", value=False)

    st.header("Query (Optional)")
    query = st.text_area("Query text", value="", height=120)
    generate_answer = st.checkbox("Generate final answer", value=False)
    run_button = st.button("Run Preview", type="primary", use_container_width=True)

current_signature = _build_request_signature(
    file_key,
    {
        "provider_preset": provider_preset,
        "base_url": base_url,
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "cleaning_mode": cleaning_mode,
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "structured_record_mode": structured_record_mode,
        "pdf_parser_mode": pdf_parser_mode,
        "use_metadata_inference": use_metadata_inference,
        "manual_field_selection": _selection_to_payload(manual_field_selection),
        "metadata_mode": metadata_mode,
        "enable_filter": enable_filter,
        "filter_field": filter_field,
        "filter_operator": filter_operator,
        "filter_value": filter_value,
        "index_type": index_type,
        "embedding_backend": embedding_backend,
        "hybrid_vector_weight": hybrid_vector_weight,
        "top_k": top_k,
        "use_mmr": use_mmr,
        "use_rerank": use_rerank,
        "use_query_rewrite": use_query_rewrite,
        "query": query,
        "generate_answer": generate_answer,
    },
)

if run_button:
    base_llm_config = LLMConfig(
        api_key=api_key or None,
        base_url=base_url or None,
        embedding_model=embedding_model,
        chat_model=chat_model,
        embedding_backend="local_tfidf",
        provider_preset=provider_preset,
    )
    run_llm_config = LLMConfig(
        api_key=api_key or None,
        base_url=base_url or None,
        embedding_model=embedding_model,
        chat_model=chat_model,
        embedding_backend=embedding_backend,
        provider_preset=provider_preset,
    )

    try:
        with st.spinner("Running preview pipeline..."):
            inspection = load_input(file_bytes, active_filename, pdf_parser_mode=pdf_parser_mode)
            effective_field_selection = inspection.field_selection or StructuredFieldSelection(source="none")
            inference_warnings: list[str] = []
            if file_type in {"json", "csv"}:
                effective_field_selection, inference_warnings = _resolve_structured_selection(
                    base_selection=effective_field_selection,
                    manual_selection=manual_field_selection,
                    manual_controls_available=bool(available_fields),
                    available_fields=inspection.structured_fields,
                    use_metadata_inference=use_metadata_inference,
                    llm_config=base_llm_config,
                    rows=[unit.raw_fields for unit in inspection.units],
                    file_preview=file_bytes[:6000].decode("utf-8", errors="ignore"),
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
                llm_config=run_llm_config,
                field_selection=effective_field_selection,
                pdf_parser_mode=pdf_parser_mode,
                use_metadata_inference=use_metadata_inference,
            )
            prepared = prepare_preview_artifacts(file_bytes, active_filename, config)
            preview_run = run_preview(prepared, config)

        combined_warnings = [*inference_warnings, *prepared.warnings, *preview_run.warnings]
        st.session_state["preview_prepared"] = prepared
        st.session_state["preview_run"] = preview_run
        st.session_state["preview_field_selection"] = effective_field_selection
        st.session_state["preview_warnings"] = combined_warnings
        st.session_state["preview_notes"] = list(effective_field_selection.notes)
        st.session_state["preview_signature"] = current_signature
        st.session_state["preview_file_key"] = file_key
        stored_prepared = prepared
        stored_preview_run = preview_run
        stored_field_selection = effective_field_selection
        stored_warnings = combined_warnings
        stored_notes = list(effective_field_selection.notes)
        stored_signature = current_signature
    except Exception as exc:
        st.error(f"Failed to run the preview pipeline: {exc}")
results_are_stale = stored_prepared is not None and stored_signature != current_signature
if results_are_stale:
    st.warning("Settings have changed since the last preview run. Click Run Preview to apply the current configuration.")

if stored_prepared is None:
    st.info(
        "Configure the sidebar and click Run Preview. Parsing, chunking, indexing, retrieval, reranking, context assembly, and any API call are all gated behind that button."
    )
    st.stop()

for warning in stored_warnings:
    st.warning(warning)
for note in stored_notes:
    st.info(note)

summary_left, summary_right = st.columns([1.7, 1])
with summary_left:
    text_field_label = ", ".join(stored_prepared.parsed_input.summary.get("detected_text_fields", []) or [])
    metadata_field_label = ", ".join(stored_prepared.parsed_input.summary.get("detected_metadata_fields", []) or [])
    st.markdown(
        f"""
        <div class="rag-card">
            <div class="rag-tag">file: {stored_prepared.parsed_input.file_type}</div>
            <div class="rag-tag">chunks: {len(stored_prepared.chunks)}</div>
            <div class="rag-tag">index: {index_type}</div>
            <div class="rag-tag">vector backend: {stored_prepared.index_bundle.vector_backend_used}</div>
            <div class="rag-tag">field source: {stored_field_selection.source}</div>
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
    stats = stored_prepared.parsed_input.summary
    metric_cols = st.columns(2)
    metric_cols[0].metric("Document units", len(stored_prepared.parsed_input.units))
    metric_cols[1].metric("Chunks", len(stored_prepared.chunks))
    if stored_prepared.parsed_input.file_type == "pdf":
        metric_cols = st.columns(2)
        metric_cols[0].metric("Pages", stats.get("page_count", 0))
        metric_cols[1].metric("Parser", stats.get("pdf_parser", "pypdf"))
    else:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Records", stats.get("record_count", 0))
        metric_cols[1].metric("Text fields", len(stats.get("detected_text_fields", [])))
        if len(stored_prepared.cleaned_units) != len(stored_prepared.parsed_input.units):
            st.caption(f"Chunk input units after merge: {len(stored_prepared.cleaned_units)}")

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
    st.json(stored_prepared.parsed_input.summary)
    preview_df = pd.DataFrame(parsed_preview_rows(stored_prepared.parsed_input))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    if stored_prepared.parsed_input.structured_fields:
        st.caption("Detected structured fields: " + ", ".join(stored_prepared.parsed_input.structured_fields))

with chunk_tab:
    st.subheader("Generated chunks")
    page_size = st.selectbox("Chunk page size", options=[25, 50, 100], index=2)
    total_pages = max(1, math.ceil(len(stored_prepared.chunks) / page_size))
    current_page = st.number_input("Chunk page", min_value=1, max_value=total_pages, value=1, step=1)
    chunk_start = (current_page - 1) * page_size
    chunk_end = min(chunk_start + page_size, len(stored_prepared.chunks))
    current_chunks = stored_prepared.chunks[chunk_start:chunk_end]
    st.write({"total_chunks": len(stored_prepared.chunks), "showing": f"{chunk_start + 1}-{chunk_end} of {len(stored_prepared.chunks)}"})
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
    metadata_total_pages = max(1, math.ceil(len(stored_prepared.chunks) / metadata_page_size))
    metadata_page = st.number_input("Metadata page", min_value=1, max_value=metadata_total_pages, value=1, step=1)
    metadata_start = (metadata_page - 1) * metadata_page_size
    metadata_end = min(metadata_start + metadata_page_size, len(stored_prepared.chunks))
    st.write({"total_chunks": len(stored_prepared.chunks), "showing": f"{metadata_start + 1}-{metadata_end} of {len(stored_prepared.chunks)}"})
    metadata_df = pd.DataFrame(metadata_preview_rows(stored_prepared.chunks[metadata_start:metadata_end], limit=metadata_page_size))
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)
with retrieval_tab:
    if stored_preview_run is None:
        st.info("Enter a query and click Run Preview to inspect retrieval outputs.")
    elif not stored_preview_run.results:
        st.warning("No retrieval results were returned.")
    else:
        st.subheader("Retrieval inputs")
        st.write({"original_query": stored_preview_run.original_query, "retrieval_query": stored_preview_run.rewritten_query})
        result_df = pd.DataFrame(retrieval_preview_rows(stored_preview_run.results))
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        for result in stored_preview_run.results:
            title = f"Rank {result.rank} | {result.chunk.chunk_id} | score={result.score:.4f}"
            with st.expander(title):
                st.json(chunk_details(result))
                st.write(result.chunk.text)

with rerank_tab:
    if stored_preview_run is None:
        st.info("Rerank comparison appears after a preview run.")
    elif not use_rerank:
        st.info("Enable rerank in the sidebar to compare before and after ordering, then click Run Preview.")
    else:
        before_col, after_col = st.columns(2)
        with before_col:
            st.subheader("Before rerank")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(stored_preview_run.pre_rerank_results)), use_container_width=True, hide_index=True)
        with after_col:
            st.subheader("After rerank")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(stored_preview_run.results)), use_container_width=True, hide_index=True)

with debug_tab:
    if stored_preview_run is None:
        st.info("Run a preview to see retrieval debug details.")
    else:
        debug = stored_preview_run.retrieval_debug
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
                "rerank_applied": use_rerank,
                "execution_policy": "All heavy pipeline work is triggered only by the Run Preview button.",
            }
        )
        st.subheader("Raw vector results")
        st.dataframe(pd.DataFrame(_trace_rows(debug.vector_raw_results)), use_container_width=True, hide_index=True)
        st.subheader("Raw keyword results")
        st.dataframe(pd.DataFrame(_trace_rows(debug.keyword_raw_results)), use_container_width=True, hide_index=True)
        st.subheader("Raw merged results")
        st.dataframe(pd.DataFrame(_trace_rows(debug.merged_raw_results)), use_container_width=True, hide_index=True)
        if use_rerank:
            st.subheader("Reranked results")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(stored_preview_run.results)), use_container_width=True, hide_index=True)
        else:
            st.subheader("Final retrieval results")
            st.dataframe(pd.DataFrame(retrieval_preview_rows(stored_preview_run.results)), use_container_width=True, hide_index=True)

with context_tab:
    if stored_preview_run is None:
        st.info("The assembled context block will appear here after a preview run.")
    else:
        st.code(stored_preview_run.final_context or "", language="markdown")

with answer_tab:
    if stored_preview_run is None:
        st.info("Optional final answer output appears here after a preview run.")
    elif stored_preview_run.final_answer:
        st.subheader("Generated answer")
        st.write(stored_preview_run.final_answer)
    elif generate_answer:
        st.warning("Answer generation was enabled, but no answer was produced. Check warnings above.")
    else:
        st.info("Enable final answer generation in the sidebar, then click Run Preview to call the chat model.")

