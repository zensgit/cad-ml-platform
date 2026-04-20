from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.analysis_live_pipeline import run_analysis_live_pipeline


class _Analyzer:
    def classify_part(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return {"part_type": "support", "confidence": 0.8}

    def check_quality(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return {"score": 0.9}

    def recommend_process(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return {"primary_process": "laser_cutting"}


@pytest.mark.asyncio
async def test_run_analysis_live_pipeline_delegates_full_flow(monkeypatch):
    captured = {}

    class _Options:
        extract_features = True
        calculate_similarity = True
        reference_id = "ref-1"
        enable_ocr = True
        ocr_provider = "mock"

    class _Result(dict):
        pass

    async def _run_preflight(**kwargs):  # noqa: ANN003, ANN201
        captured["preflight"] = kwargs
        return {
            "analysis_options": _Options(),
            "analysis_cache_key": "cache-key",
            "cached_response": None,
        }

    async def _run_document(**kwargs):  # noqa: ANN003, ANN201
        captured["document"] = kwargs
        return {
            "file_format": "dxf",
            "doc": {"id": "doc-1"},
            "unified_data": {"preview_image": "img"},
            "parse_stage_duration": 0.12,
        }

    async def _run_feature(**kwargs):  # noqa: ANN003, ANN201
        captured["feature"] = kwargs
        return {
            "features": {"geometric": [1], "semantic": [2]},
            "features_3d": {"bbox": [1, 2, 3]},
            "results_patch": {"features_ready": True},
            "features_3d_stage_duration": None,
            "features_stage_duration": 0.08,
        }

    async def _run_parallel(**kwargs):  # noqa: ANN003, ANN201
        captured["parallel"] = kwargs
        kwargs["results"]["classification"] = {"part_type": "support"}
        kwargs["results"]["quality"] = {"score": 0.91}
        kwargs["results"]["process"] = {"primary_process": "laser_cutting"}
        return {"classification": 0.3}

    def _attach_manufacturing(**kwargs):  # noqa: ANN003, ANN202
        captured["manufacturing"] = kwargs
        kwargs["results"]["manufacturing_decision"] = {"ready": True}

    async def _attach_drift(**kwargs):  # noqa: ANN003, ANN201
        captured["drift"] = kwargs

    async def _attach_vector(**kwargs):  # noqa: ANN003, ANN201
        captured["vector"] = kwargs
        kwargs["results"]["similarity"] = {"score": 0.88}
        return {"registered": True}

    async def _attach_ocr(**kwargs):  # noqa: ANN003, ANN201
        captured["ocr"] = kwargs
        kwargs["results"]["ocr"] = {"status": "ok"}

    async def _build_response(**kwargs):  # noqa: ANN003, ANN201
        captured["response"] = kwargs
        return {"analysis_id": kwargs["analysis_id"], "ok": True}

    result = await run_analysis_live_pipeline(
        file_name="demo.dxf",
        content=b"0\nEOF\n",
        options_raw="{}",
        material="steel",
        project_id="p-1",
        analysis_id="analysis-1",
        start_time=datetime(2026, 4, 20, tzinfo=timezone.utc),
        options_model_cls=object,
        result_model_cls=_Result,
        analyzer_factory=_Analyzer,
        run_preflight_fn=_run_preflight,
        run_document_pipeline_fn=_run_document,
        run_feature_pipeline_fn=_run_feature,
        run_parallel_pipeline_fn=_run_parallel,
        attach_manufacturing_summary_fn=_attach_manufacturing,
        build_manufacturing_summary_fn=lambda **_kwargs: {"ready": True},
        attach_drift_fn=_attach_drift,
        drift_state={},
        run_drift_pipeline_fn=lambda **_kwargs: None,
        attach_vector_context_fn=_attach_vector,
        run_vector_pipeline_fn=lambda **_kwargs: None,
        get_qdrant_store_fn=lambda: None,
        compute_qdrant_similarity_fn=lambda **_kwargs: None,
        attach_ocr_payload_fn=_attach_ocr,
        run_ocr_pipeline_fn=lambda **_kwargs: None,
        build_response_fn=_build_response,
        finalize_analysis_success_fn=lambda **_kwargs: None,
        classification_pipeline_fn=lambda **_kwargs: None,
        quality_pipeline_fn=lambda **_kwargs: None,
        process_pipeline_fn=lambda **_kwargs: None,
        logger_instance=type("L", (), {"info": lambda *_args, **_kwargs: None})(),
    )

    assert result == {"analysis_id": "analysis-1", "ok": True}
    assert captured["preflight"]["file_name"] == "demo.dxf"
    assert captured["document"]["project_id"] == "p-1"
    assert captured["parallel"]["quality_pipeline"] is not None
    assert captured["vector"]["classification_meta"] == {"part_type": "support"}
    assert captured["ocr"]["results"]["manufacturing_decision"] == {"ready": True}
    assert captured["response"]["vector_context"] == {"registered": True}


@pytest.mark.asyncio
async def test_run_analysis_live_pipeline_returns_cached_result_without_running_pipelines():
    called = {"document": False}

    class _Result(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    async def _run_preflight(**_kwargs):  # noqa: ANN201
        return {
            "analysis_options": object(),
            "analysis_cache_key": "cache-key",
            "cached_response": {"analysis_id": "cached-1", "status": "cached"},
        }

    async def _run_document(**_kwargs):  # noqa: ANN201
        called["document"] = True
        raise AssertionError("document pipeline should not run on cache hit")

    result = await run_analysis_live_pipeline(
        file_name="cached.dxf",
        content=b"",
        options_raw="{}",
        material=None,
        project_id=None,
        analysis_id="analysis-cache",
        start_time=datetime(2026, 4, 20, tzinfo=timezone.utc),
        options_model_cls=object,
        result_model_cls=_Result,
        analyzer_factory=_Analyzer,
        run_preflight_fn=_run_preflight,
        run_document_pipeline_fn=_run_document,
        run_feature_pipeline_fn=lambda **_kwargs: None,
        run_parallel_pipeline_fn=lambda **_kwargs: None,
        attach_manufacturing_summary_fn=lambda **_kwargs: None,
        build_manufacturing_summary_fn=lambda **_kwargs: None,
        attach_drift_fn=lambda **_kwargs: None,
        drift_state={},
        run_drift_pipeline_fn=lambda **_kwargs: None,
        attach_vector_context_fn=lambda **_kwargs: None,
        run_vector_pipeline_fn=lambda **_kwargs: None,
        get_qdrant_store_fn=lambda: None,
        compute_qdrant_similarity_fn=lambda **_kwargs: None,
        attach_ocr_payload_fn=lambda **_kwargs: None,
        run_ocr_pipeline_fn=lambda **_kwargs: None,
        build_response_fn=lambda **_kwargs: None,
        finalize_analysis_success_fn=lambda **_kwargs: None,
        classification_pipeline_fn=lambda **_kwargs: None,
        quality_pipeline_fn=lambda **_kwargs: None,
        process_pipeline_fn=lambda **_kwargs: None,
        logger_instance=type("L", (), {"info": lambda *_args, **_kwargs: None})(),
    )

    assert dict(result) == {"analysis_id": "cached-1", "status": "cached"}
    assert called["document"] is False
