from __future__ import annotations

import logging

import pytest

from src.core.dfm.quality_pipeline import run_quality_pipeline


@pytest.mark.asyncio
async def test_run_quality_pipeline_prefers_dfm_and_uses_classification_getter():
    captured: dict[str, object] = {}
    observed_latency: list[float] = []
    geometry_calls: list[str] = []

    class _StubDfmAnalyzer:
        def analyze(self, dfm_features, part_type):  # noqa: ANN001, ANN201
            captured["dfm_features"] = dict(dfm_features)
            captured["part_type"] = part_type
            return {
                "dfm_score": 92.5,
                "issues": [{"code": "THIN_WALL"}],
                "manufacturability": "medium",
            }

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when DFM succeeds")

    result = await run_quality_pipeline(
        doc=object(),
        features={"geometric": [1], "semantic": [2]},
        features_3d={"stock_removal_ratio": 0.1},
        check_quality=_fallback,
        classification_payload_getter=lambda: {"part_type": "shaft"},
        dfm_analyzer_factory=lambda: _StubDfmAnalyzer(),
        geometry_engine_factory=lambda: geometry_calls.append("called") or object(),
        dfm_latency_observer=observed_latency.append,
    )

    assert result == {
        "mode": "L4_DFM",
        "score": 92.5,
        "issues": [{"code": "THIN_WALL"}],
        "manufacturability": "medium",
    }
    assert captured["part_type"] == "shaft"
    assert captured["dfm_features"] == {"stock_removal_ratio": 0.1}
    assert geometry_calls == ["called"]
    assert observed_latency and observed_latency[0] >= 0.0


@pytest.mark.asyncio
async def test_run_quality_pipeline_skips_geometry_probe_when_dfm_features_present():
    geometry_calls: list[str] = []

    class _StubDfmAnalyzer:
        def analyze(self, dfm_features, part_type):  # noqa: ANN001, ANN201
            return {
                "dfm_score": 88.0,
                "issues": [],
                "manufacturability": "high",
            }

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when DFM succeeds")

    result = await run_quality_pipeline(
        doc=object(),
        features={},
        features_3d={"thin_walls_detected": False},
        check_quality=_fallback,
        classification_payload={"part_type": "simple_plate"},
        dfm_analyzer_factory=lambda: _StubDfmAnalyzer(),
        geometry_engine_factory=lambda: geometry_calls.append("called") or object(),
    )

    assert result["mode"] == "L4_DFM"
    assert geometry_calls == []


@pytest.mark.asyncio
async def test_run_quality_pipeline_falls_back_to_check_quality_on_dfm_error(caplog):
    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        assert features == {"geometric": [1]}
        return {
            "score": 0.7,
            "issues": ["fallback_issue"],
            "suggestions": ["fallback_suggestion"],
            "legacy_payload": True,
        }

    with caplog.at_level(logging.ERROR):
        result = await run_quality_pipeline(
            doc=object(),
            features={"geometric": [1]},
            features_3d={"thin_walls_detected": True},
            check_quality=_fallback,
            classification_payload={"part_type": "shaft"},
            dfm_analyzer_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("dfm unavailable")
            ),
        )

    assert result == {
        "score": 0.7,
        "issues": ["fallback_issue"],
        "suggestions": ["fallback_suggestion"],
        "legacy_payload": True,
    }
    assert "DFM check failed: dfm unavailable" in caplog.text


@pytest.mark.asyncio
async def test_run_quality_pipeline_normalizes_non_dfm_quality_payload():
    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        assert features == {"semantic": [2]}
        return {
            "score": 0.95,
            "issues": ["no_layers_defined"],
            "suggestions": ["organize_entities_into_layers"],
            "legacy_payload": True,
        }

    result = await run_quality_pipeline(
        doc=object(),
        features={"semantic": [2]},
        features_3d={},
        check_quality=_fallback,
    )

    assert result == {
        "score": 0.95,
        "issues": ["no_layers_defined"],
        "suggestions": ["organize_entities_into_layers"],
    }


@pytest.mark.asyncio
async def test_run_quality_pipeline_treats_none_features_3d_as_non_dfm_path():
    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        return {
            "score": 0.82,
            "issues": [],
            "suggestions": ["fallback_only"],
        }

    result = await run_quality_pipeline(
        doc=object(),
        features={"semantic": [3]},
        features_3d=None,
        check_quality=_fallback,
    )

    assert result == {
        "score": 0.82,
        "issues": [],
        "suggestions": ["fallback_only"],
    }


@pytest.mark.asyncio
async def test_run_quality_pipeline_defaults_none_getter_payload_to_unknown():
    captured: dict[str, object] = {}

    class _StubDfmAnalyzer:
        def analyze(self, dfm_features, part_type):  # noqa: ANN001, ANN201
            captured["part_type"] = part_type
            return {
                "dfm_score": 80.0,
                "issues": [],
                "manufacturability": "high",
            }

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when DFM succeeds")

    result = await run_quality_pipeline(
        doc=object(),
        features={},
        features_3d={"thin_walls_detected": True},
        check_quality=_fallback,
        classification_payload_getter=lambda: None,
        dfm_analyzer_factory=lambda: _StubDfmAnalyzer(),
    )

    assert result["mode"] == "L4_DFM"
    assert captured["part_type"] == "unknown"


@pytest.mark.asyncio
async def test_run_quality_pipeline_falls_back_when_dfm_payload_is_incomplete():
    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        return {
            "score": 0.61,
            "issues": ["fallback_issue"],
            "suggestions": ["fallback_suggestion"],
        }

    class _StubDfmAnalyzer:
        def analyze(self, dfm_features, part_type):  # noqa: ANN001, ANN201
            return {"issues": []}

    result = await run_quality_pipeline(
        doc=object(),
        features={"geometric": [1]},
        features_3d={"thin_walls_detected": True},
        check_quality=_fallback,
        classification_payload={"part_type": "shaft"},
        dfm_analyzer_factory=lambda: _StubDfmAnalyzer(),
    )

    assert result == {
        "score": 0.61,
        "issues": ["fallback_issue"],
        "suggestions": ["fallback_suggestion"],
    }
