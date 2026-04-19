from __future__ import annotations

from typing import Any, Dict

import pytest

from src.core.analysis_vector_attachment import attach_analysis_vector_context


@pytest.mark.asyncio
async def test_attach_analysis_vector_context_writes_similarity_and_stage_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    stage_times = {"parse": 0.5, "features": 0.25}
    observed_similarity_stage: list[float] = []

    async def _vector_pipeline(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {
            "registered": True,
            "similarity": {"reference_id": "ref-1", "score": 0.9},
            "vector_metadata": {},
        }

    time_values = iter([10.0])
    monkeypatch.setattr("src.core.analysis_vector_attachment.time.time", lambda: next(time_values))

    vector_context = await attach_analysis_vector_context(
        analysis_id="analysis-1",
        doc=object(),
        features={"geometric": [1.0]},
        features_3d={},
        material="steel",
        classification_meta={"part_type": "plate"},
        calculate_similarity=True,
        reference_id="ref-1",
        results=results,
        stage_times=stage_times,
        started_at=8.0,
        vector_pipeline=_vector_pipeline,
        get_qdrant_store=lambda: "store",
        compute_qdrant_similarity=lambda *_args, **_kwargs: 0.9,
        vector_material_observer=lambda material: material,
        feature_dimension_observer=lambda value: value,
        similarity_stage_observer=observed_similarity_stage.append,
    )

    assert vector_context["registered"] is True
    assert results["similarity"] == {"reference_id": "ref-1", "score": 0.9}
    assert stage_times["similarity"] == pytest.approx(1.25)
    assert observed_similarity_stage == [pytest.approx(1.25)]
    assert captured["classification_meta"] == {"part_type": "plate"}
    assert captured["calculate_similarity"] is True
    assert captured["reference_id"] == "ref-1"


@pytest.mark.asyncio
async def test_attach_analysis_vector_context_skips_similarity_stage_without_result() -> None:
    results: Dict[str, Any] = {}
    stage_times = {"parse": 0.5}
    observed_similarity_stage: list[float] = []

    async def _vector_pipeline(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "registered": True,
            "similarity": None,
            "vector_metadata": {},
        }

    vector_context = await attach_analysis_vector_context(
        analysis_id="analysis-2",
        doc=object(),
        features={},
        features_3d={},
        material=None,
        classification_meta={},
        calculate_similarity=False,
        reference_id=None,
        results=results,
        stage_times=stage_times,
        started_at=1.0,
        vector_pipeline=_vector_pipeline,
        get_qdrant_store=lambda: None,
        compute_qdrant_similarity=lambda *_args, **_kwargs: None,
        vector_material_observer=lambda material: material,
        feature_dimension_observer=lambda value: value,
        similarity_stage_observer=observed_similarity_stage.append,
    )

    assert vector_context["registered"] is True
    assert "similarity" not in results
    assert "similarity" not in stage_times
    assert observed_similarity_stage == []
