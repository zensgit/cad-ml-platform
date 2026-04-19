from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api.v1.analyze_live_models import AnalysisResult
from src.core.analysis_response_builder import build_analysis_response


@pytest.mark.asyncio
async def test_build_analysis_response_finalizes_and_wraps_result_model() -> None:
    captured: dict[str, object] = {}
    timestamp = datetime.now(timezone.utc)

    async def _finalize(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "id": kwargs["analysis_id"],
            "timestamp": kwargs["start_time"],
            "file_name": kwargs["file_name"],
            "file_format": kwargs["file_format"],
            "results": {"statistics": {"entity_count": 1}},
            "processing_time": 0.25,
            "cache_hit": False,
            "cad_document": {"file_name": kwargs["file_name"]},
            "feature_version": "v-test",
        }

    result = await build_analysis_response(
        result_model_cls=AnalysisResult,
        finalize_analysis_success_fn=_finalize,
        analysis_id="analysis-1",
        start_time=timestamp,
        file_name="part.dxf",
        file_format="dxf",
        results={},
        doc=object(),
        stage_times={"parse": 0.1},
        analysis_cache_key="analysis:key",
        vector_context={"registered": True},
        material="steel",
        unified_data={"preview_image": "stub"},
        logger_instance=object(),
    )

    assert isinstance(result, AnalysisResult)
    assert result.id == "analysis-1"
    assert result.feature_version == "v-test"
    assert captured["analysis_cache_key"] == "analysis:key"
    assert captured["material"] == "steel"


@pytest.mark.asyncio
async def test_build_analysis_response_propagates_finalize_errors() -> None:
    async def _finalize(**_kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await build_analysis_response(
            result_model_cls=AnalysisResult,
            finalize_analysis_success_fn=_finalize,
            analysis_id="analysis-2",
            start_time=datetime.now(timezone.utc),
            file_name="part.dxf",
            file_format="dxf",
            results={},
            doc=object(),
            stage_times={},
            analysis_cache_key="analysis:key",
            vector_context={},
            material=None,
            unified_data={},
            logger_instance=object(),
        )
