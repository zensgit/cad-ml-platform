from __future__ import annotations

from src.api.v1 import vectors as vectors_module
from src.core.vector_migration_config import (
    coerce_optional_int,
    resolve_vector_migration_scan_limit,
    resolve_vector_migration_target_version,
)


def test_resolve_vector_migration_scan_limit_defaults_and_clamps(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VECTOR_MIGRATION_SCAN_LIMIT", raising=False)
    assert resolve_vector_migration_scan_limit() == 5000

    monkeypatch.setenv("VECTOR_MIGRATION_SCAN_LIMIT", "0")
    assert resolve_vector_migration_scan_limit() == 1

    monkeypatch.setenv("VECTOR_MIGRATION_SCAN_LIMIT", "-10")
    assert resolve_vector_migration_scan_limit() == 1


def test_resolve_vector_migration_scan_limit_reads_env_and_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VECTOR_MIGRATION_SCAN_LIMIT", "321")
    assert resolve_vector_migration_scan_limit() == 321

    monkeypatch.setenv("VECTOR_MIGRATION_SCAN_LIMIT", "invalid")
    assert resolve_vector_migration_scan_limit(default=77) == 77


def test_resolve_vector_migration_target_version_normalizes_allowed_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VECTOR_MIGRATION_TARGET_VERSION", " V3 ")
    assert resolve_vector_migration_target_version() == "v3"

    monkeypatch.setenv("VECTOR_MIGRATION_TARGET_VERSION", "v2")
    assert resolve_vector_migration_target_version() == "v2"


def test_resolve_vector_migration_target_version_falls_back(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VECTOR_MIGRATION_TARGET_VERSION", raising=False)
    assert resolve_vector_migration_target_version() == "v4"

    monkeypatch.setenv("VECTOR_MIGRATION_TARGET_VERSION", "v9")
    assert resolve_vector_migration_target_version(default="v3") == "v3"


def test_coerce_optional_int_handles_missing_and_invalid_values() -> None:
    assert coerce_optional_int(None) is None
    assert coerce_optional_int("12") == 12
    assert coerce_optional_int("invalid") is None


def test_vectors_facade_preserves_migration_config_exports() -> None:
    assert (
        vectors_module._resolve_vector_migration_scan_limit
        is resolve_vector_migration_scan_limit
    )
    assert (
        vectors_module._resolve_vector_migration_target_version
        is resolve_vector_migration_target_version
    )
    assert vectors_module._coerce_int is coerce_optional_int
