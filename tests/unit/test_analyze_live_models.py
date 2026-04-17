from __future__ import annotations

from src.api.v1.analyze_live_models import (
    AnalysisOptions,
    AnalysisResult,
    BatchClassifyResponse,
    BatchClassifyResultItem,
    SimilarityResult,
    SimilarityTopKItem,
    SimilarityTopKResponse,
)


def test_analysis_options_history_file_path_round_trip():
    options = AnalysisOptions(
        extract_features=False,
        enable_ocr=True,
        history_file_path="/tmp/history.h5",
    )

    assert options.extract_features is False
    assert options.enable_ocr is True
    assert options.history_file_path == "/tmp/history.h5"


def test_analysis_result_schema_defaults():
    result = AnalysisResult(
        id="analysis-1",
        timestamp="2026-04-18T00:00:00Z",
        file_name="sample.dxf",
        file_format="dxf",
        results={},
        processing_time=1.23,
    )

    assert result.cache_hit is False
    assert result.feature_version == "v1"
    assert result.cad_document is None


def test_batch_classify_response_embeds_result_items():
    payload = BatchClassifyResponse(
        total=1,
        success=1,
        failed=0,
        processing_time=0.5,
        results=[
            BatchClassifyResultItem(
                file_name="a.dxf",
                category="零件",
                confidence=0.9,
                needs_review=False,
            )
        ],
    )

    assert payload.results[0].file_name == "a.dxf"
    assert payload.results[0].confidence == 0.9


def test_similarity_models_support_optional_label_fields():
    result = SimilarityResult(
        reference_id="ref",
        target_id="target",
        score=0.88,
        method="cosine",
        dimension=256,
        reference_part_type="轴",
        target_coarse_part_type="传动件",
    )
    topk = SimilarityTopKResponse(
        target_id="target",
        k=1,
        results=[SimilarityTopKItem(id="match", score=0.77, coarse_part_type="传动件")],
    )

    assert result.reference_part_type == "轴"
    assert result.target_coarse_part_type == "传动件"
    assert topk.results[0].id == "match"
