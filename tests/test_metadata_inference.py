from pipeline.metadata_inference import normalize_suggestion_payload



def test_normalize_suggestion_payload_keeps_string_note_whole() -> None:
    selection = normalize_suggestion_payload(
        {
            "text_fields": ["question", "answer"],
            "metadata_fields": ["label", "source"],
            "identifier_fields": ["record_id"],
            "notes": "The JSON keys suggest that question and answer should be treated as text.",
        }
    )
    assert selection.text_fields == ["question", "answer"]
    assert selection.metadata_fields == ["label", "source"]
    assert selection.identifier_fields == ["record_id"]
    assert selection.notes == ["The JSON keys suggest that question and answer should be treated as text."]



def test_normalize_suggestion_payload_splits_field_strings() -> None:
    selection = normalize_suggestion_payload(
        {
            "text_fields": "question, answer",
            "metadata_fields": "label\nsource",
            "identifier_fields": "record_id",
            "notes": ["Use question and answer as the chunk text."],
        }
    )
    assert selection.text_fields == ["question", "answer"]
    assert selection.metadata_fields == ["label", "source"]
    assert selection.identifier_fields == ["record_id"]
