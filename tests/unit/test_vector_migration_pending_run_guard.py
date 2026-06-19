from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.core.errors_extended import ErrorCode, build_error
from src.core.vector_migration_pending_run_guard import (
    ensure_pending_run_scan_is_allowed,
)


def test_pending_run_guard_allows_complete_qdrant_scan() -> None:
    ensure_pending_run_scan_is_allowed(
        pending={
            "backend": "qdrant",
            "distribution_complete": True,
            "scanned_vectors": 10,
            "scan_limit": 5000,
        },
        allow_partial_scan=False,
        target_version="v4",
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
    )


def test_pending_run_guard_allows_partial_qdrant_override() -> None:
    ensure_pending_run_scan_is_allowed(
        pending={
            "backend": "qdrant",
            "distribution_complete": False,
            "scanned_vectors": 2,
            "scan_limit": 2,
        },
        allow_partial_scan=True,
        target_version="v4",
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
    )


def test_pending_run_guard_allows_memory_scan_without_override() -> None:
    ensure_pending_run_scan_is_allowed(
        pending={
            "backend": "memory",
            "distribution_complete": True,
            "scanned_vectors": 2,
            "scan_limit": 2,
        },
        allow_partial_scan=False,
        target_version="v4",
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
    )


def test_pending_run_guard_rejects_partial_qdrant_without_override() -> None:
    with pytest.raises(HTTPException) as exc_info:
        ensure_pending_run_scan_is_allowed(
            pending={
                "backend": "qdrant",
                "distribution_complete": False,
                "scanned_vectors": 2,
                "scan_limit": 2,
            },
            allow_partial_scan=False,
            target_version="v4",
            error_code_cls=ErrorCode,
            build_error_fn=build_error,
        )

    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["code"] == ErrorCode.CONSTRAINT_VIOLATION.value
    assert detail["stage"] == "vector_migrate_pending_run"
    assert detail["context"] == {
        "target_version": "v4",
        "scanned_vectors": 2,
        "scan_limit": 2,
    }
