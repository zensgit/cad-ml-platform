from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from src.core.analysis_preflight import (
    build_analysis_cache_key,
    run_analysis_request_preflight,
)


class _Options(BaseModel):
    extract_features: bool = True
    classify_parts: bool = True


def test_build_analysis_cache_key_appends_part_shadow_suffix(monkeypatch):
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_INCLUDE_IN_CACHE_KEY", "true")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS", "dxf,dwg")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_ENABLED", "true")
    monkeypatch.setenv("PART_CLASSIFIER_PROVIDER_NAME", "v16")

    key = build_analysis_cache_key(
        file_name="sample.dxf",
        content=b"DXF-DATA",
        options_raw='{"extract_features": true}',
    )

    assert key.startswith("analysis:sample.dxf:")
    assert key.endswith(":part_shadow=1:v16")


@pytest.mark.asyncio
async def test_run_analysis_request_preflight_cache_hit(monkeypatch):
    monkeypatch.setenv("FEATURE_VERSION", "v9")

    async def _cache_getter(key: str):
        assert key.startswith("analysis:sample.dxf:")
        return {"classification": {"part_type": "bracket"}}

    result = await run_analysis_request_preflight(
        file_name="sample.dxf",
        options_raw='{"extract_features": true, "classify_parts": false}',
        content=b"DXF-DATA",
        analysis_id="analysis-1",
        timestamp=datetime(2026, 4, 17, tzinfo=timezone.utc),
        options_model_cls=_Options,
        cache_getter=_cache_getter,
    )

    assert result["analysis_options"].classify_parts is False
    assert result["cached"] == {"classification": {"part_type": "bracket"}}
    assert result["cached_response"]["cache_hit"] is True
    assert result["cached_response"]["feature_version"] == "v9"
    assert result["cached_response"]["file_format"] == "DXF"


@pytest.mark.asyncio
async def test_run_analysis_request_preflight_cache_miss():
    async def _cache_getter(_: str):
        return None

    result = await run_analysis_request_preflight(
        file_name="sample.dxf",
        options_raw='{"extract_features": false, "classify_parts": true}',
        content=b"DXF-DATA",
        analysis_id="analysis-2",
        timestamp=datetime(2026, 4, 17, tzinfo=timezone.utc),
        options_model_cls=_Options,
        cache_getter=_cache_getter,
    )

    assert result["analysis_options"].extract_features is False
    assert result["cached"] is None
    assert result["cached_response"] is None
    assert result["analysis_cache_key"].startswith("analysis:sample.dxf:")
