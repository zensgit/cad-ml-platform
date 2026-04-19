from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.analysis_parallel_pipeline import run_analysis_parallel_pipeline


@pytest.mark.asyncio
async def test_run_analysis_parallel_pipeline_runs_enabled_tasks_and_records_metrics():
    observed: dict[str, object] = {
        "classification_latency": [],
        "dfm_latency": [],
        "process_latency": [],
        "process_rule_versions": [],
        "cost_latency": [],
        "stages": [],
        "parallel_enabled": [],
        "parallel_savings": [],
    }
    results: dict[str, object] = {}

    async def _classify_pipeline(**kwargs):  # noqa: ANN003, ANN201
        assert kwargs["analysis_id"] == "analysis-1"
        return {"part_type": "plate"}

    async def _quality_pipeline(**kwargs):  # noqa: ANN003, ANN201
        classification_payload = kwargs["classification_payload_getter"]()
        assert classification_payload in ({}, {"part_type": "plate"})
        kwargs["dfm_latency_observer"](0.12)
        return {"score": 91.0}

    async def _process_pipeline(**kwargs):  # noqa: ANN003, ANN201
        classification_payload = kwargs["classification_payload_getter"]()
        assert classification_payload in ({}, {"part_type": "plate"})
        kwargs["process_rule_version_observer"]("v-test")
        kwargs["cost_latency_observer"](0.34)
        return {
            "process": {"primary_recommendation": {"process": "laser_cutting"}},
            "cost_estimation": {"total_unit_cost": 8.5},
        }

    stage_times = await run_analysis_parallel_pipeline(
        analysis_id="analysis-1",
        analysis_options=SimpleNamespace(
            classify_parts=True,
            quality_check=True,
            process_recommendation=True,
            estimate_cost=True,
        ),
        doc=object(),
        features={"geometric": [1.0]},
        features_3d={},
        file_name="part.dxf",
        file_format="dxf",
        content=b"0\nEOF\n",
        material="steel",
        results=results,
        classify_pipeline=_classify_pipeline,
        classify_part=lambda *_args, **_kwargs: None,
        quality_pipeline=_quality_pipeline,
        check_quality=lambda *_args, **_kwargs: None,
        process_pipeline=_process_pipeline,
        recommend_process=lambda *_args, **_kwargs: None,
        logger_instance=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        classification_latency_observer=lambda value: observed[
            "classification_latency"
        ].append(value),
        dfm_latency_observer=lambda value: observed["dfm_latency"].append(value),
        process_latency_observer=lambda value: observed["process_latency"].append(value),
        process_rule_version_observer=lambda value: observed[
            "process_rule_versions"
        ].append(value),
        cost_latency_observer=lambda value: observed["cost_latency"].append(value),
        stage_duration_observer=lambda stage, value: observed["stages"].append(
            (stage, value)
        ),
        parallel_enabled_setter=lambda value: observed["parallel_enabled"].append(value),
        parallel_savings_observer=lambda value: observed["parallel_savings"].append(value),
    )

    assert results["classification"] == {"part_type": "plate"}
    assert results["quality"] == {"score": 91.0}
    assert results["process"] == {
        "primary_recommendation": {"process": "laser_cutting"}
    }
    assert results["cost_estimation"] == {"total_unit_cost": 8.5}
    assert set(stage_times) == {"classify", "quality", "process"}
    assert observed["parallel_enabled"] == [1]
    assert len(observed["classification_latency"]) == 1
    assert observed["dfm_latency"] == [0.12]
    assert len(observed["process_latency"]) == 1
    assert observed["process_rule_versions"] == ["v-test"]
    assert observed["cost_latency"] == [0.34]
    assert len(observed["stages"]) == 3
    assert len(observed["parallel_savings"]) == 1
    assert observed["parallel_savings"][0] >= 0.0


@pytest.mark.asyncio
async def test_run_analysis_parallel_pipeline_sets_parallel_disabled_when_nothing_enabled():
    observed_parallel: list[int] = []
    results: dict[str, object] = {}

    stage_times = await run_analysis_parallel_pipeline(
        analysis_id="analysis-2",
        analysis_options=SimpleNamespace(
            classify_parts=False,
            quality_check=False,
            process_recommendation=False,
            estimate_cost=False,
        ),
        doc=object(),
        features={},
        features_3d={},
        file_name="part.dxf",
        file_format="dxf",
        content=b"",
        material=None,
        results=results,
        classify_pipeline=lambda **_kwargs: None,  # type: ignore[arg-type]
        classify_part=lambda *_args, **_kwargs: None,
        quality_pipeline=lambda **_kwargs: None,  # type: ignore[arg-type]
        check_quality=lambda *_args, **_kwargs: None,
        process_pipeline=lambda **_kwargs: None,  # type: ignore[arg-type]
        recommend_process=lambda *_args, **_kwargs: None,
        logger_instance=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        classification_latency_observer=lambda _value: None,
        dfm_latency_observer=lambda _value: None,
        process_latency_observer=lambda _value: None,
        process_rule_version_observer=lambda _value: None,
        cost_latency_observer=lambda _value: None,
        stage_duration_observer=lambda _stage, _value: None,
        parallel_enabled_setter=observed_parallel.append,
        parallel_savings_observer=lambda _value: None,
    )

    assert stage_times == {}
    assert results == {}
    assert observed_parallel == [0]
