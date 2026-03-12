from pipeline.chunkers import fixed_chunks, sentence_window_chunks


def test_fixed_chunks_respect_overlap() -> None:
    chunks = fixed_chunks("abcdefghijklmnopqrstuvwxyz", chunk_size=10, chunk_overlap=2)
    assert len(chunks) >= 3
    assert chunks[0][0] == "abcdefghij"
    assert chunks[1][1] == 8


def test_sentence_window_chunks_group_sentences() -> None:
    text = "Sentence one. Sentence two is longer. Sentence three ends here."
    chunks = sentence_window_chunks(text, chunk_size=35, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(chunk[0] for chunk in chunks)
