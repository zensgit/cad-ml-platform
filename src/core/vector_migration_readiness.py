from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.vector_migration_config import resolve_vector_migration_target_version


def build_vector_migration_readiness(
    version_distribution: dict[str, int],
    *,
    total_vectors: int,
    distribution_complete: bool,
    resolve_target_version_fn: Callable[[], str] = resolve_vector_migration_target_version,
) -> dict[str, Any]:
    target_version = resolve_target_version_fn()
    readiness: dict[str, Any] = {
        "target_version": target_version,
        "target_version_vectors": None,
        "target_version_ratio": None,
        "pending_vectors": None,
        "migration_ready": False,
    }
    if not distribution_complete:
        return readiness

    target_vectors = int(version_distribution.get(target_version, 0))
    pending_vectors = max(int(total_vectors) - target_vectors, 0)
    readiness["target_version_vectors"] = target_vectors
    readiness["target_version_ratio"] = round(target_vectors / max(int(total_vectors), 1), 4)
    readiness["pending_vectors"] = pending_vectors
    readiness["migration_ready"] = pending_vectors == 0
    return readiness


__all__ = ["build_vector_migration_readiness"]
