from __future__ import annotations

import re

from pipeline.schemas import ChunkRecord, DocumentUnit


def chunk_units(
    units: list[DocumentUnit],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for unit in units:
        if not unit.text.strip():
            continue
        if strategy == "sentence_window":
            segments = sentence_window_chunks(unit.text, chunk_size, chunk_overlap)
        else:
            segments = fixed_chunks(unit.text, chunk_size, chunk_overlap)

        for segment_index, (text, start_char, end_char) in enumerate(segments, start=1):
            chunk_id = f"{unit.doc_id}_chunk_{segment_index:03d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=unit.doc_id,
                    text=text,
                    source_name=unit.source_name,
                    unit_type=unit.unit_type,
                    start_char=start_char,
                    end_char=end_char,
                    page_number=unit.page_number,
                    section=unit.section,
                    record_id=unit.record_id,
                    text_id=unit.text_id,
                    label=unit.label,
                    record_type=unit.record_type,
                    source=unit.source,
                    raw_fields=dict(unit.raw_fields),
                )
            )
    return chunks


def fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    clean_text = text.strip()
    if not clean_text:
        return []

    size = max(chunk_size, 1)
    overlap = min(max(chunk_overlap, 0), max(size - 1, 0))
    segments: list[tuple[str, int, int]] = []
    start = 0
    length = len(clean_text)

    while start < length:
        end = min(length, start + size)
        chunk_text = clean_text[start:end].strip()
        if chunk_text:
            segments.append((chunk_text, start, end))
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return segments


def sentence_window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    sentences = _split_sentences(text)
    if not sentences:
        return fixed_chunks(text, chunk_size, chunk_overlap)

    segments: list[tuple[str, int, int]] = []
    start_index = 0

    while start_index < len(sentences):
        end_index = start_index
        current_length = 0
        while end_index < len(sentences):
            sentence_text = sentences[end_index][0]
            projected = current_length + len(sentence_text) + (1 if current_length else 0)
            if end_index > start_index and projected > chunk_size:
                break
            current_length = projected
            end_index += 1

        window_sentences = sentences[start_index:end_index]
        window_text = " ".join(sentence[0] for sentence in window_sentences).strip()
        window_start = window_sentences[0][1]
        window_end = window_sentences[-1][2]
        if window_text:
            segments.append((window_text, window_start, window_end))

        if end_index >= len(sentences):
            break

        overlap_chars = 0
        next_start = end_index
        while next_start > start_index:
            next_start -= 1
            overlap_chars += len(sentences[next_start][0]) + 1
            if overlap_chars >= chunk_overlap:
                break

        if next_start <= start_index:
            start_index = end_index
        else:
            start_index = next_start

    return segments


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    pattern = re.compile(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)", re.MULTILINE)
    sentences: list[tuple[str, int, int]] = []
    for match in pattern.finditer(text):
        sentence = match.group().strip()
        if sentence:
            sentences.append((sentence, match.start(), match.end()))
    return sentences
