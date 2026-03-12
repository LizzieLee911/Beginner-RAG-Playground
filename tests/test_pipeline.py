import json

from pipeline.loaders import load_input
from pipeline.preview import prepare_preview_artifacts, run_preview
from pipeline.schemas import PipelineConfig


CSV_BYTES = b"record_id,text_id,record_type,label,source,section,text\nrec-1,txt-1,summary,medication,ward_a,assessment,Insulin should be reviewed before discharge.\nrec-2,txt-2,summary,follow_up,ward_b,plan,The patient should repeat renal labs next week.\n"

DICT_JSON_BYTES = json.dumps(
    {
        "11380492": {
            "question": "What are symptoms of diabetes?",
            "answer": "Frequent urination and thirst.",
            "label": "qa",
        },
        "11380493": {
            "question": "How is asthma treated?",
            "answer": "Inhaled bronchodilators.",
            "label": "qa",
        },
    }
).encode("utf-8")



def test_structured_json_dict_of_records_creates_multiple_units() -> None:
    parsed = load_input(DICT_JSON_BYTES, "sample.json")
    assert len(parsed.units) == 2
    assert parsed.summary["record_count"] == 2
    assert parsed.summary["detected_text_fields"] == ["question", "answer"]
    assert {unit.record_id for unit in parsed.units} == {"11380492", "11380493"}



def test_preview_pipeline_returns_relevant_result() -> None:
    config = PipelineConfig(
        cleaning_mode="basic_clean",
        chunk_strategy="fixed_chunk",
        chunk_size=120,
        chunk_overlap=20,
        metadata_mode="full",
        index_type="hybrid",
        query="insulin discharge",
        top_k=2,
    )
    prepared = prepare_preview_artifacts(CSV_BYTES, "sample.csv", config)
    result = run_preview(prepared, config)
    assert prepared.chunks
    assert result.results
    assert "insulin" in result.results[0].chunk.text.lower()
    assert "chunk_id=" in result.final_context



def test_different_queries_hit_different_structured_records() -> None:
    diabetes_config = PipelineConfig(metadata_mode="full", index_type="hybrid", query="diabetes thirst", top_k=1)
    asthma_config = PipelineConfig(metadata_mode="full", index_type="hybrid", query="asthma inhaled", top_k=1)

    diabetes_result = run_preview(prepare_preview_artifacts(DICT_JSON_BYTES, "sample.json", diabetes_config), diabetes_config)
    asthma_result = run_preview(prepare_preview_artifacts(DICT_JSON_BYTES, "sample.json", asthma_config), asthma_config)

    assert diabetes_result.results[0].chunk.record_id == "11380492"
    assert asthma_result.results[0].chunk.record_id == "11380493"



def test_no_match_query_returns_no_results() -> None:
    config = PipelineConfig(metadata_mode="full", index_type="hybrid", query="zzzxxyyqq", top_k=3)
    prepared = prepare_preview_artifacts(DICT_JSON_BYTES, "sample.json", config)
    result = run_preview(prepared, config)
    assert result.results == []
    assert any("non-zero raw retrieval scores" in warning for warning in result.warnings)
