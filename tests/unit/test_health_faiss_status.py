"""Unit + facade-compat tests for the extracted faiss_health computation."""

from __future__ import annotations

import src.api.v1.health as health_router
from src.core.health_faiss_status import compute_faiss_health


def test_helper_reexported_from_router_module() -> None:
    assert health_router.compute_faiss_health is compute_faiss_health


def test_status_priority_degraded_wins_over_unavailable() -> None:
    # degraded takes precedence even when the index is unavailable
    r = compute_faiss_health(
        available=False, degraded=True, last_export_ts=None, last_import=None
    )
    assert r["status"] == "degraded"


def test_status_unavailable_when_not_available_and_not_degraded() -> None:
    r = compute_faiss_health(
        available=False, degraded=False, last_export_ts=None, last_import=None
    )
    assert r["status"] == "unavailable"


def test_status_ok_when_available_and_not_degraded() -> None:
    r = compute_faiss_health(
        available=True, degraded=False, last_export_ts=None, last_import=None
    )
    assert r["status"] == "ok"


def test_age_uses_export_ts_when_present() -> None:
    r = compute_faiss_health(
        available=True,
        degraded=False,
        last_export_ts=100.0,
        last_import=50.0,
        now=160.0,
    )
    assert r["age_seconds"] == 60


def test_age_falls_back_to_import_when_no_export() -> None:
    r = compute_faiss_health(
        available=True,
        degraded=False,
        last_export_ts=None,
        last_import=50.0,
        now=160.0,
    )
    assert r["age_seconds"] == 110


def test_age_none_when_no_timestamps() -> None:
    r = compute_faiss_health(
        available=True, degraded=False, last_export_ts=None, last_import=None
    )
    assert r["age_seconds"] is None


def test_age_is_int_truncated() -> None:
    r = compute_faiss_health(
        available=True,
        degraded=False,
        last_export_ts=100.0,
        last_import=None,
        now=160.9,
    )
    # int() truncates toward zero, matching the prior inline behaviour
    assert r["age_seconds"] == 60
    assert isinstance(r["age_seconds"], int)
