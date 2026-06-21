"""Unit + facade-compat tests for the extracted model_health computation."""

from __future__ import annotations

import src.api.v1.health as health_router
from src.core.health_model_status import compute_model_health


def test_helper_reexported_from_router_module() -> None:
    assert health_router.compute_model_health is compute_model_health


def test_status_absent_when_not_loaded() -> None:
    assert compute_model_health({"loaded": False})["status"] == "absent"


def test_status_rollback_when_level_positive() -> None:
    r = compute_model_health({"loaded": True, "rollback_level": 2})
    assert r["status"] == "rollback"
    assert r["rollback_level"] == 2


def test_status_ok_with_snapshots_and_uptime() -> None:
    r = compute_model_health(
        {
            "loaded": True,
            "rollback_level": 0,
            "has_prev": True,
            "has_prev2": True,
            "has_prev3": None,
            "loaded_at": 100.0,
        },
        now=160.0,
    )
    assert r["status"] == "ok"
    assert r["snapshots_available"] == 2
    assert r["uptime_seconds"] == 60.0


def test_uptime_none_without_loaded_at() -> None:
    r = compute_model_health({"loaded": True}, now=200.0)
    assert r["uptime_seconds"] is None
    assert r["snapshots_available"] == 0
