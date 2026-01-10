"""Coverage tests for analysis result store utilities."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.utils.analysis_result_store import (
    cleanup_analysis_results,
    get_analysis_result_store_stats,
    load_analysis_result,
    store_analysis_result,
)


@pytest.mark.asyncio
async def test_store_disabled_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANALYSIS_RESULT_STORE_DIR", raising=False)
    stored = await store_analysis_result("abc123", {"status": "ok"})
    assert stored is False
    loaded = await load_analysis_result("abc123")
    assert loaded is None


@pytest.mark.asyncio
async def test_store_and_load_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
    payload = {"status": "ok", "nested": {"count": 2}}
    stored = await store_analysis_result("abc123", payload)
    assert stored is True
    loaded = await load_analysis_result("abc123")
    assert loaded == payload
    assert (tmp_path / "abc123.json").exists()


@pytest.mark.asyncio
async def test_store_rejects_invalid_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
    stored = await store_analysis_result("../bad", {"status": "nope"})
    assert stored is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_cleanup_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANALYSIS_RESULT_STORE_DIR", raising=False)
    result = await cleanup_analysis_results()
    assert result["status"] == "disabled"


@pytest.mark.asyncio
async def test_cleanup_skipped_without_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
    await store_analysis_result("abc123", {"status": "ok"})
    result = await cleanup_analysis_results()
    assert result["status"] == "skipped"
    assert result["total_files"] == 1


@pytest.mark.asyncio
async def test_cleanup_by_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
    await store_analysis_result("old", {"status": "old"})
    await store_analysis_result("new", {"status": "new"})
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    now = time.time()
    os.utime(old_path, (now - 3600, now - 3600))
    os.utime(new_path, (now, now))

    preview = await cleanup_analysis_results(max_age_seconds=60, dry_run=True, sample_limit=5)
    assert preview["status"] == "dry_run"
    assert preview["eligible_count"] == 1
    assert preview["expired_count"] == 1
    assert preview["sample_ids"] == ["old"]

    result = await cleanup_analysis_results(max_age_seconds=60, dry_run=False)
    assert result["deleted_count"] == 1
    assert not old_path.exists()
    assert new_path.exists()


@pytest.mark.asyncio
async def test_cleanup_by_max_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
    await store_analysis_result("first", {"status": "first"})
    await store_analysis_result("second", {"status": "second"})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    now = time.time()
    os.utime(first_path, (now - 120, now - 120))
    os.utime(second_path, (now, now))

    preview = await cleanup_analysis_results(max_files=1, dry_run=True)
    assert preview["overflow_count"] == 1
    assert preview["eligible_count"] == 1

    result = await cleanup_analysis_results(max_files=1, dry_run=False)
    assert result["deleted_count"] == 1
    assert not first_path.exists()
    assert second_path.exists()


def test_stats_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANALYSIS_RESULT_STORE_DIR", raising=False)
    stats = get_analysis_result_store_stats()
    assert stats["enabled"] is False
    assert stats["total_files"] == 0


@pytest.mark.asyncio
async def test_stats_with_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
    await store_analysis_result("stat1", {"status": "ok"})
    stats = get_analysis_result_store_stats()
    assert stats["enabled"] is True
    assert stats["total_files"] == 1
    assert stats["oldest_mtime"] is not None
