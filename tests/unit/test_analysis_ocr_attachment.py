from __future__ import annotations

from typing import Any, Dict

import pytest

from src.core.analysis_ocr_attachment import attach_analysis_ocr_payload


@pytest.mark.asyncio
async def test_attach_analysis_ocr_payload_writes_result() -> None:
    captured: Dict[str, Any] = {}
    results: Dict[str, Any] = {}

    async def _ocr_pipeline(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"provider": "paddle", "text": "A1"}

    payload = await attach_analysis_ocr_payload(
        enable_ocr=True,
        ocr_provider_strategy="paddle",
        unified_data={"preview_image": "stub"},
        results=results,
        ocr_pipeline=_ocr_pipeline,
    )

    assert payload == {"provider": "paddle", "text": "A1"}
    assert results["ocr"] == {"provider": "paddle", "text": "A1"}
    assert captured == {
        "enable_ocr": True,
        "ocr_provider_strategy": "paddle",
        "unified_data": {"preview_image": "stub"},
    }


@pytest.mark.asyncio
async def test_attach_analysis_ocr_payload_skips_none_result() -> None:
    results: Dict[str, Any] = {}

    async def _ocr_pipeline(**_kwargs: Any) -> None:
        return None

    payload = await attach_analysis_ocr_payload(
        enable_ocr=False,
        ocr_provider_strategy="auto",
        unified_data={},
        results=results,
        ocr_pipeline=_ocr_pipeline,
    )

    assert payload is None
    assert results == {}
