"""Canonical tool status semantics for honest degradation (§2.C)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# Non-success statuses required by the design-lock (exact enum).
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DEGRADED = "degraded"
STATUS_OK = "ok"

CANONICAL_NON_OK = frozenset({STATUS_FAILED, STATUS_UNAVAILABLE, STATUS_DEGRADED})
CANONICAL_ALL = CANONICAL_NON_OK | {STATUS_OK}


def failure_result(
    status: str,
    reason_code: str,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a structured failure/degraded tool payload.

    ``status`` must be one of failed/unavailable/degraded. Free-form detail goes
    in ``reason_code`` (and optional extra fields), never as a fake success body.
    """
    if status not in CANONICAL_NON_OK:
        raise ValueError(f"status must be one of {sorted(CANONICAL_NON_OK)}, got {status!r}")
    out: Dict[str, Any] = {"status": status, "reason_code": reason_code}
    out.update(extra)
    return out


def is_citable_tool_result(result: Any) -> bool:
    """False when the tool result must not be used as decision evidence."""
    if not isinstance(result, Mapping):
        return False
    status = result.get("status")
    if status is None:
        # Legacy success payloads without status remain citable only if they
        # do not carry a non-ok marker.
        return True
    if status in CANONICAL_NON_OK:
        return False
    return status == STATUS_OK or status == "ok"


def assert_canonical_status(result: Mapping[str, Any]) -> None:
    """Raise AssertionError if a failure payload lacks a canonical status."""
    status = result.get("status")
    if status not in CANONICAL_NON_OK:
        raise AssertionError(f"expected canonical non-ok status, got {status!r}")


__all__ = [
    "CANONICAL_ALL",
    "CANONICAL_NON_OK",
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "assert_canonical_status",
    "failure_result",
    "is_citable_tool_result",
]
