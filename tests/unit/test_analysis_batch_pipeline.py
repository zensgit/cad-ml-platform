from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.analysis_batch_pipeline import run_batch_analysis


@pytest.mark.asyncio
async def test_run_batch_analysis_uses_keyword_api_key_and_aggregates():
    calls = []

    async def _analyze_file_fn(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        name = kwargs["file"].filename
        if name == "bad.dxf":
            raise RuntimeError("boom")
        return SimpleNamespace(model_dump=lambda: {"file_name": name, "ok": True})

    files = [
        SimpleNamespace(filename="good1.dxf"),
        SimpleNamespace(filename="bad.dxf"),
        SimpleNamespace(filename="good2.dxf"),
    ]

    result = await run_batch_analysis(
        files=files,
        options='{"extract_features": true}',
        api_key="secret",
        analyze_file_fn=_analyze_file_fn,
    )

    assert [call["api_key"] for call in calls] == ["secret", "secret", "secret"]
    assert all(call["options"] == '{"extract_features": true}' for call in calls)
    assert result["total"] == 3
    assert result["successful"] == 2
    assert result["failed"] == 1
    assert result["results"][0] == {"file_name": "good1.dxf", "ok": True}
    assert result["results"][1] == {"file_name": "bad.dxf", "error": "boom"}


@pytest.mark.asyncio
async def test_run_batch_analysis_preserves_plain_mapping_results():
    async def _analyze_file_fn(**kwargs):  # noqa: ANN003, ANN202
        return {"file_name": kwargs["file"].filename, "cache_hit": False}

    files = [SimpleNamespace(filename="one.dxf")]
    result = await run_batch_analysis(
        files=files,
        options="{}",
        api_key="secret",
        analyze_file_fn=_analyze_file_fn,
    )

    assert result == {
        "total": 1,
        "successful": 1,
        "failed": 0,
        "results": [{"file_name": "one.dxf", "cache_hit": False}],
    }
