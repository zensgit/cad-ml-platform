from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.ocr.analysis_ocr_pipeline import run_analysis_ocr_pipeline


class _Item:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _Manager:
    def __init__(self, result=None):
        self.result = result
        self.providers = {}
        self.extract_calls = []

    def register_provider(self, name, provider):  # noqa: ANN001
        self.providers[name] = provider

    async def extract(self, image_bytes, strategy):  # noqa: ANN001
        self.extract_calls.append((image_bytes, strategy))
        return self.result


@pytest.mark.asyncio
async def test_run_analysis_ocr_pipeline_disabled():
    result = await run_analysis_ocr_pipeline(
        enable_ocr=False,
        ocr_provider_strategy="auto",
        unified_data={},
    )

    assert result is None


@pytest.mark.asyncio
async def test_run_analysis_ocr_pipeline_without_preview_image():
    boot = []
    manager = _Manager()

    result = await run_analysis_ocr_pipeline(
        enable_ocr=True,
        ocr_provider_strategy="auto",
        unified_data={},
        bootstrap_registry_fn=lambda: boot.append("boot"),
        provider_getter=lambda kind, name: f"{kind}:{name}",
        manager_factory=lambda: manager,
    )

    assert result == {"status": "no_preview_image"}
    assert boot == ["boot"]
    assert manager.providers == {
        "paddle": "ocr:paddle",
        "deepseek_hf": "ocr:deepseek_hf",
    }
    assert manager.extract_calls == []


@pytest.mark.asyncio
async def test_run_analysis_ocr_pipeline_success_prefers_calibrated_confidence():
    manager = _Manager(
        result=SimpleNamespace(
            provider="paddle",
            calibrated_confidence=0.93,
            confidence=0.81,
            fallback_level="none",
            dimensions=[_Item({"text": "10", "confidence": 0.9})],
            symbols=[_Item({"text": "R", "confidence": 0.8})],
            completeness=0.88,
        )
    )

    result = await run_analysis_ocr_pipeline(
        enable_ocr=True,
        ocr_provider_strategy="paddle",
        unified_data={"preview_image_bytes": b"fake-image"},
        bootstrap_registry_fn=lambda: None,
        provider_getter=lambda kind, name: f"{kind}:{name}",
        manager_factory=lambda: manager,
    )

    assert result == {
        "provider": "paddle",
        "confidence": 0.93,
        "fallback_level": "none",
        "dimensions": [{"text": "10", "confidence": 0.9}],
        "symbols": [{"text": "R", "confidence": 0.8}],
        "completeness": 0.88,
    }
    assert manager.extract_calls == [(b"fake-image", "paddle")]
