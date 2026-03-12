# RAG Preview

RAG Preview is a local-first Streamlit MVP for experimenting with configurable RAG pipelines on PDFs and structured datasets. It is designed as a transparent research workbench: upload a file, inspect how records are parsed, choose chunking and retrieval settings, compare raw and reranked results, and optionally call an OpenAI-compatible model for metadata inference, query rewrite, embeddings, or final answer generation.

## Who It Is For

- People learning RAG who want to see the intermediate artifacts instead of only the final answer
- Researchers comparing chunking, metadata, and retrieval strategies on small local datasets
- Teams that need a lightweight local demo before building a larger RAG platform

## MVP Capabilities

- Upload PDF, JSON, or CSV
- Parse PDFs page by page with `pypdf`
- Optionally try an `unstructured` PDF parser mode with graceful fallback to `pypdf`
- Parse structured JSON and CSV into one document unit per record by default
- Detect and optionally override likely text fields, metadata fields, and identifier fields for structured files
- Toggle cleaning mode, chunk strategy, chunk size, and overlap
- Switch metadata mode: `none`, `basic`, `page_source`, `full`
- Run vector, keyword, or hybrid retrieval
- Apply optional metadata filters with `equals` or `contains`
- Use optional query rewrite and optional rerank
- Inspect parsed previews, chunk previews, metadata, raw retrieval scores, rerank order, final context, and optional final answer
- Choose provider presets for OpenAI, SiliconFlow, or a custom OpenAI-compatible endpoint
- Keep parsing, chunking, indexing, and retrieval preview working without any API key

## Supported File Types

- PDF: page-level text extraction using `pypdf`; optional `unstructured` mode when the dependency is installed
- JSON:
  - list of records
  - dict containing one or more lists of records
  - dict-of-records payloads such as `{ "123": {...}, "124": {...} }`
  - single dict payloads
- CSV: row-oriented structured data, one row per document unit

## Structured Mode

For JSON and CSV, the loader first performs rule-based field inspection:

- `text fields`: likely long-form fields such as `text`, `content`, `question`, `answer`, or `summary`
- `identifier fields`: likely IDs such as `record_id`, `text_id`, or `id`
- `metadata fields`: short scalar fields such as `label`, `record_type`, `section`, or `source`

If an API key is provided, the app can optionally send the first 6000 characters of the uploaded JSON/CSV sample to a chat model and ask for a better guess. The sidebar then lets the user override the selected text, metadata, and identifier fields manually.

Chunking happens across all parsed records by default. If `One record per document unit` is disabled, the structured rows are merged into one large document before chunking.

## Provider Presets

The UI includes three provider modes for API-based features:

- OpenAI
  - Base URL default: `https://api.openai.com/v1`
  - Chat model default: `gpt-4o-mini`
  - Embedding model default: `text-embedding-3-small`
- SiliconFlow
  - Base URL default: `https://api.siliconflow.cn/v1`
  - Chat model default: `deepseek-ai/DeepSeek-V3.2`
  - Embedding model default: `Qwen/Qwen3-Embedding-8B`
- Custom
  - User provides base URL, chat model, and embedding model

No API key is required for parsing, chunking, metadata preview, or local retrieval. An API key is only required for:

- metadata inference with the chat model
- query rewrite
- final answer generation
- optional OpenAI-compatible embedding generation

## Retrieval Debugging

The `Debug` tab exposes the current retrieval state so failure modes are easy to diagnose:

- total indexed chunks
- eligible chunks after metadata filtering
- selected index type and vector backend
- whether query rewrite ran
- the final query string used for retrieval
- active metadata filters
- raw vector scores
- raw keyword scores
- raw hybrid merge scores
- final retrieval or reranked results

## Architecture Overview

The app uses a small modular pipeline:

1. `pipeline/loaders.py`: detect file type and parse PDF pages or structured records
2. `pipeline/pdf_parsers.py`: `pypdf` parsing plus optional `unstructured` parsing
3. `pipeline/field_inference.py`: rule-based structured field detection
4. `pipeline/metadata_inference.py`: optional LLM-based field suggestion for structured data
5. `pipeline/cleaners.py`: raw or basic-clean normalization
6. `pipeline/chunkers.py`: fixed-size or sentence-window chunking
7. `pipeline/metadata.py`: attach chunk metadata by mode
8. `pipeline/indexers.py`: build local TF-IDF vector index, optional OpenAI-compatible embedding index, and BM25 keyword index
9. `pipeline/retrievers.py`: metadata filtering, vector retrieval, keyword retrieval, hybrid score merge, optional MMR, and retrieval debug traces
10. `pipeline/rerankers.py`: lightweight heuristic reranking
11. `pipeline/context_builder.py`: assemble the exact context block sent to the LLM
12. `pipeline/query_ops.py`: optional query rewrite and optional final answer generation
13. `app.py`: Streamlit control panel and artifact rendering

## Module Overview

- `app.py`: Streamlit UI
- `pipeline/schemas.py`: typed dataclasses for documents, chunks, configs, debug traces, and retrieval results
- `sample_data/`: small JSON and CSV examples, including a dict-of-records JSON sample
- `tests/`: chunking, metadata, structured parsing, and retrieval sanity tests

## Run Locally

From the project directory:

```bash
cd rag_preview
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

If you want API-based features, copy `.env.example` to `.env` and populate the API key. The UI lets you override base URL and model names.

Optional PDF metadata mode:

```bash
pip install unstructured
```

If `unstructured` is not installed or fails at runtime, the app falls back to `pypdf` automatically.

## Run With Docker

From the project directory:

```bash
cd rag_preview
docker build -t rag-preview .
docker run -p 8501:8501 rag-preview
```

If you want API-based features in Docker, pass the key through environment variables or enter it in the UI for the current session.

## Tests

```bash
cd rag_preview
pytest
```

## Known Limitations

- PDF extraction is still mostly text-oriented; `unstructured` support is optional and not bundled by default
- The default local vector backend is an in-memory TF-IDF representation, chosen for portability over ANN performance
- Reranking is heuristic, not a dedicated cross-encoder
- Metadata filters support a single field/operator/value condition in this MVP
- Large files may rebuild slowly because the preview favors transparency over aggressive caching
- LLM-based field inference depends on the quality of the first 6000-character sample and provider compatibility

## Roadmap Ideas

- Parent-child chunking
- Context compression
- Persistent local vector stores such as FAISS or Chroma
- Multiple metadata filters
- Additional retrievers and rerankers
- Query routing
- Session-level experiment comparison and export
