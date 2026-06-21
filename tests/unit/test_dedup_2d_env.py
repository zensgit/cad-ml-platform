"""Unit + facade-compat tests for the extracted dedup 2D env helpers."""

from __future__ import annotations

import src.api.v1.dedup as dedup_router
from src.core.dedup_2d_env import (
    _check_forced_async,
    _get_bool_env,
    _get_dedup2d_async_backend,
    _get_int_env,
)


def test_helpers_reexported_from_router_module() -> None:
    assert dedup_router._check_forced_async is _check_forced_async
    assert dedup_router._get_int_env is _get_int_env
    assert dedup_router._get_bool_env is _get_bool_env
    assert dedup_router._get_dedup2d_async_backend is _get_dedup2d_async_backend


def test_get_int_env(monkeypatch) -> None:
    monkeypatch.delenv("X_DEDUP_INT", raising=False)
    assert _get_int_env("X_DEDUP_INT", default=7) == 7
    monkeypatch.setenv("X_DEDUP_INT", "12")
    assert _get_int_env("X_DEDUP_INT", default=7) == 12
    monkeypatch.setenv("X_DEDUP_INT", "nope")
    assert _get_int_env("X_DEDUP_INT", default=7) == 7  # invalid -> default


def test_get_bool_env(monkeypatch) -> None:
    monkeypatch.setenv("X_DEDUP_BOOL", "yes")
    assert _get_bool_env("X_DEDUP_BOOL", default=False) is True
    monkeypatch.setenv("X_DEDUP_BOOL", "off")
    assert _get_bool_env("X_DEDUP_BOOL", default=True) is False
    monkeypatch.delenv("X_DEDUP_BOOL", raising=False)
    assert _get_bool_env("X_DEDUP_BOOL", default=True) is True


def test_check_forced_async_thresholds() -> None:
    assert _check_forced_async(10 * 1024 * 1024, False, "fast", None) == "file_size>5MB"
    assert (
        _check_forced_async(1, True, "fast", {"a": 1})
        == "enable_precision_with_geom_json"
    )
    assert _check_forced_async(1, False, "precise", None) == "mode=precise"
    assert _check_forced_async(1, False, "fast", None) is None
