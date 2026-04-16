from __future__ import annotations

import logging

import pytest

from src.core.process.process_pipeline import run_process_pipeline


@pytest.mark.asyncio
async def test_run_process_pipeline_prefers_ai_recommendation_and_cost():
    captured: dict[str, object] = {}
    observed_cost_latency: list[float] = []

    class _StubRecommender:
        def recommend(self, dfm_features, part_type, material):  # noqa: ANN001, ANN201
            captured["dfm_features"] = dict(dfm_features)
            captured["part_type"] = part_type
            captured["material"] = material
            return {
                "primary_recommendation": {
                    "process": "turning",
                    "method": "cnc_lathe",
                },
                "alternatives": [],
                "analysis_mode": "L4_AI_Heuristic",
            }

    class _StubEstimator:
        def estimate(self, features_3d, primary_proc, material="steel"):  # noqa: ANN001, ANN201
            captured["cost_features"] = dict(features_3d)
            captured["primary_proc"] = dict(primary_proc)
            captured["cost_material"] = material
            return {"total_unit_cost": 12.3, "currency": "USD"}

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when AI process succeeds")

    result = await run_process_pipeline(
        doc=object(),
        features={"geometric": [1]},
        features_3d={"volume": 10.0, "stock_removal_ratio": 0.1},
        recommend_process=_fallback,
        material="aluminum",
        estimate_cost=True,
        classification_payload_getter=lambda: {"part_type": "shaft"},
        process_recommender_factory=lambda: _StubRecommender(),
        cost_estimator_factory=lambda: _StubEstimator(),
        cost_latency_observer=observed_cost_latency.append,
    )

    assert result["process"]["analysis_mode"] == "L4_AI_Heuristic"
    assert result["cost_estimation"] == {"total_unit_cost": 12.3, "currency": "USD"}
    assert captured["part_type"] == "shaft"
    assert captured["material"] == "aluminum"
    assert captured["primary_proc"] == {
        "process": "turning",
        "method": "cnc_lathe",
    }
    assert observed_cost_latency and observed_cost_latency[0] >= 0.0


@pytest.mark.asyncio
async def test_run_process_pipeline_defaults_none_getter_payload_to_unknown():
    captured: dict[str, object] = {}

    class _StubRecommender:
        def recommend(self, dfm_features, part_type, material):  # noqa: ANN001, ANN201
            captured["part_type"] = part_type
            return {
                "primary_recommendation": {"process": "general_machining"},
                "alternatives": [],
            }

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when AI process succeeds")

    result = await run_process_pipeline(
        doc=object(),
        features={},
        features_3d={"volume": 10.0},
        recommend_process=_fallback,
        classification_payload_getter=lambda: None,
        process_recommender_factory=lambda: _StubRecommender(),
    )

    assert result["process"]["primary_recommendation"]["process"] == "general_machining"
    assert captured["part_type"] == "unknown"


@pytest.mark.asyncio
async def test_run_process_pipeline_accepts_static_classification_payload():
    captured: dict[str, object] = {}

    class _StubRecommender:
        def recommend(self, dfm_features, part_type, material):  # noqa: ANN001, ANN201
            captured["part_type"] = part_type
            return {
                "primary_recommendation": {"process": "cnc_milling"},
                "alternatives": [],
            }

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when AI process succeeds")

    result = await run_process_pipeline(
        doc=object(),
        features={},
        features_3d={"volume": 11.0},
        recommend_process=_fallback,
        classification_payload={"part_type": "housing"},
        process_recommender_factory=lambda: _StubRecommender(),
    )

    assert result["process"]["primary_recommendation"]["process"] == "cnc_milling"
    assert captured["part_type"] == "housing"


@pytest.mark.asyncio
async def test_run_process_pipeline_falls_back_to_rule_process_on_ai_error(caplog):
    captured: dict[str, object] = {}

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        captured["fallback_features"] = features
        return {
            "process": "cnc_milling",
            "method": "standard",
            "rule_version": "rules-v1",
        }

    class _StubEstimator:
        def estimate(self, features_3d, primary_proc, material="steel"):  # noqa: ANN001, ANN201
            captured["primary_proc"] = dict(primary_proc)
            return {"total_unit_cost": 8.4}

    with caplog.at_level(logging.ERROR):
        result = await run_process_pipeline(
            doc=object(),
            features={"semantic": [2]},
            features_3d={"volume": 20.0},
            recommend_process=_fallback,
            estimate_cost=True,
            classification_payload={"part_type": "plate"},
            process_recommender_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("ai unavailable")
            ),
            cost_estimator_factory=lambda: _StubEstimator(),
        )

    assert result["process"] == {
        "process": "cnc_milling",
        "method": "standard",
        "rule_version": "rules-v1",
    }
    assert result["cost_estimation"] == {"total_unit_cost": 8.4}
    assert captured["primary_proc"] == {
        "process": "cnc_milling",
        "method": "standard",
    }
    assert "AI Process failed: ai unavailable" in caplog.text


@pytest.mark.asyncio
async def test_run_process_pipeline_non_3d_path_observes_rule_version():
    observed_versions: list[str] = []

    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        assert features == {"semantic": [5]}
        return {
            "recommended_process": "laser_cutting",
            "rule_version": "rules-v2",
        }

    result = await run_process_pipeline(
        doc=object(),
        features={"semantic": [5]},
        features_3d=None,
        recommend_process=_fallback,
        estimate_cost=True,
        process_rule_version_observer=observed_versions.append,
    )

    assert result == {
        "process": {
            "recommended_process": "laser_cutting",
            "rule_version": "rules-v2",
        },
        "cost_estimation": None,
    }
    assert observed_versions == ["rules-v2"]


@pytest.mark.asyncio
async def test_run_process_pipeline_logs_cost_failures_without_breaking_process(caplog):
    async def _fallback(doc, features):  # noqa: ANN001, ANN201
        raise AssertionError("fallback should not run when AI process succeeds")

    class _StubRecommender:
        def recommend(self, dfm_features, part_type, material):  # noqa: ANN001, ANN201
            return {
                "primary_recommendation": {
                    "process": "turning",
                    "method": "cnc_lathe",
                }
            }

    with caplog.at_level(logging.ERROR):
        result = await run_process_pipeline(
            doc=object(),
            features={},
            features_3d={"volume": 3.0},
            recommend_process=_fallback,
            estimate_cost=True,
            process_recommender_factory=lambda: _StubRecommender(),
            cost_estimator_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("cost unavailable")
            ),
        )

    assert result == {
        "process": {
            "primary_recommendation": {
                "process": "turning",
                "method": "cnc_lathe",
            }
        },
        "cost_estimation": None,
    }
    assert "Cost estimation failed: cost unavailable" in caplog.text
