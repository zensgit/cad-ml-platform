from __future__ import annotations

from src.api.v1 import vectors as vectors_module
from src.core.vector_migration_readiness import build_vector_migration_readiness


def test_build_vector_migration_readiness_for_complete_distribution() -> None:
    assert build_vector_migration_readiness(
        {"v4": 3, "v3": 1},
        total_vectors=4,
        distribution_complete=True,
        resolve_target_version_fn=lambda: "v4",
    ) == {
        "target_version": "v4",
        "target_version_vectors": 3,
        "target_version_ratio": 0.75,
        "pending_vectors": 1,
        "migration_ready": False,
    }


def test_build_vector_migration_readiness_for_partial_distribution() -> None:
    assert build_vector_migration_readiness(
        {"v4": 3},
        total_vectors=10,
        distribution_complete=False,
        resolve_target_version_fn=lambda: "v4",
    ) == {
        "target_version": "v4",
        "target_version_vectors": None,
        "target_version_ratio": None,
        "pending_vectors": None,
        "migration_ready": False,
    }


def test_build_vector_migration_readiness_handles_empty_total() -> None:
    assert build_vector_migration_readiness(
        {},
        total_vectors=0,
        distribution_complete=True,
        resolve_target_version_fn=lambda: "v4",
    ) == {
        "target_version": "v4",
        "target_version_vectors": 0,
        "target_version_ratio": 0.0,
        "pending_vectors": 0,
        "migration_ready": True,
    }


def test_vectors_facade_readiness_uses_facade_target_version_resolver(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vectors_module,
        "_resolve_vector_migration_target_version",
        lambda: "v3",
    )

    assert vectors_module._build_vector_migration_readiness(
        {"v4": 3, "v3": 1},
        total_vectors=4,
        distribution_complete=True,
    ) == {
        "target_version": "v3",
        "target_version_vectors": 1,
        "target_version_ratio": 0.25,
        "pending_vectors": 3,
        "migration_ready": False,
    }
