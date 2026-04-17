from datetime import datetime, timedelta, timezone

import pytest

from src.core.analysis_result_envelope import (
    build_analysis_cad_document_payload,
    build_analysis_statistics,
    finalize_analysis_success,
)
from src.models.cad_document import BoundingBox, CadDocument, CadEntity


class _Logger:
    def __init__(self) -> None:
        self.calls = []

    def info(self, event: str, *, extra):  # noqa: ANN001
        self.calls.append((event, extra))


def _make_doc(entity_count: int = 205) -> CadDocument:
    return CadDocument(
        file_name="sample.dxf",
        format="dxf",
        entities=[CadEntity(kind="LINE", layer="0") for _ in range(entity_count)],
        layers={"0": entity_count},
        bounding_box=BoundingBox(max_x=10, max_y=5),
        metadata={"material": "steel"},
        raw_stats={"segments": entity_count},
    )


def test_build_analysis_statistics_and_cad_document_payload():
    doc = _make_doc(entity_count=3)

    statistics = build_analysis_statistics(doc=doc, stage_times={"parse": 0.5})
    cad_document = build_analysis_cad_document_payload(doc)

    assert statistics == {
        "entity_count": 3,
        "layer_count": 1,
        "bounding_box": doc.bounding_box.model_dump(),
        "complexity": "low",
        "stages": {"parse": 0.5},
    }
    assert cad_document["file_name"] == "sample.dxf"
    assert cad_document["format"] == "dxf"
    assert len(cad_document["entities"]) == 3
    assert cad_document["raw_stats"] == {"segments": 3}


@pytest.mark.asyncio
async def test_finalize_analysis_success_persists_and_builds_response(monkeypatch):
    monkeypatch.setenv("FEATURE_VERSION", "v11")
    doc = _make_doc()
    logger = _Logger()
    persisted = []

    async def _cache_result(key, payload, ttl):  # noqa: ANN001, ANN202
        persisted.append(("cache", key, ttl, payload.copy()))

    async def _set_cache(key, payload, ttl_seconds):  # noqa: ANN001, ANN202
        persisted.append(("set", key, ttl_seconds, payload.copy()))

    async def _store(analysis_id, payload):  # noqa: ANN001, ANN202
        persisted.append(("store", analysis_id, payload.copy()))

    start_time = datetime(2026, 4, 17, tzinfo=timezone.utc)
    result = await finalize_analysis_success(
        analysis_id="analysis-123",
        start_time=start_time,
        file_name="sample.dxf",
        file_format="dxf",
        results={"classification": {"part_type": "bracket"}},
        doc=doc,
        stage_times={"parse": 0.5, "features": 0.8},
        analysis_cache_key="analysis:sample:key",
        vector_context={"feature_vector_dim": 128},
        material="steel",
        unified_data={"complexity": "medium"},
        logger_instance=logger,
        cache_result_fn=_cache_result,
        set_cache_fn=_set_cache,
        store_analysis_result_fn=_store,
        time_factory=lambda: start_time + timedelta(seconds=2.5),
    )

    assert result["file_format"] == "DXF"
    assert result["processing_time"] == 2.5
    assert result["cache_hit"] is False
    assert result["feature_version"] == "v11"
    assert result["results"]["statistics"]["entity_count"] == 205
    assert result["results"]["statistics"]["complexity"] == "medium"
    assert len(result["cad_document"]["entities"]) == 200
    assert result["cad_document"]["metadata"] == {"material": "steel"}

    assert persisted[0][0:3] == ("cache", "analysis:sample:key", 3600)
    assert persisted[1][0:3] == ("set", "analysis_result:analysis-123", 3600)
    assert persisted[2][0:2] == ("store", "analysis-123")

    assert logger.calls
    event, extra = logger.calls[0]
    assert event == "analysis.completed"
    assert extra["file"] == "sample.dxf"
    assert extra["analysis_id"] == "analysis-123"
    assert extra["feature_vector_dim"] == 128
    assert extra["material"] == "steel"
    assert extra["complexity"] == "medium"
