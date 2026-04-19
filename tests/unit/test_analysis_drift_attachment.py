from __future__ import annotations

from typing import Any, Dict

import pytest

from src.core.analysis_drift_attachment import attach_analysis_drift


@pytest.mark.asyncio
async def test_attach_analysis_drift_runs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    async def _drift_pipeline(**kwargs: Any) -> None:
        captured.update(kwargs)

    fake_client = object()
    monkeypatch.setattr(
        "src.utils.cache.get_client",
        lambda: fake_client,
    )

    await attach_analysis_drift(
        drift_state={"materials": []},
        material="steel",
        classification_payload={"part_type": "plate"},
        drift_pipeline=_drift_pipeline,
        material_drift_observer=lambda score: score,
        prediction_drift_observer=lambda score: score,
    )

    assert captured["drift_state"] == {"materials": []}
    assert captured["material"] == "steel"
    assert captured["classification_payload"] == {"part_type": "plate"}
    assert captured["cache_client_factory"]() is fake_client


@pytest.mark.asyncio
async def test_attach_analysis_drift_swallows_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _drift_pipeline(**_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.utils.cache.get_client", lambda: object())

    await attach_analysis_drift(
        drift_state={"materials": []},
        material=None,
        classification_payload={},
        drift_pipeline=_drift_pipeline,
        material_drift_observer=lambda score: score,
        prediction_drift_observer=lambda score: score,
    )
