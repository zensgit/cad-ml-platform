from __future__ import annotations

from typing import Any

from src.core.vector_migration_pending_run_request import (
    build_pending_run_migrate_request,
)


class _Request:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_build_pending_run_migrate_request_maps_pending_ids() -> None:
    request = build_pending_run_migrate_request(
        pending={"pending_ids": ["vec2", "vec3"]},
        target_version="v4",
        dry_run=True,
        request_cls=_Request,
    )

    assert request.ids == ["vec2", "vec3"]
    assert request.to_version == "v4"
    assert request.dry_run is True


def test_build_pending_run_migrate_request_preserves_empty_ids() -> None:
    request = build_pending_run_migrate_request(
        pending={"pending_ids": []},
        target_version="v4",
        dry_run=False,
        request_cls=_Request,
    )

    assert request.ids == []
    assert request.to_version == "v4"
    assert request.dry_run is False
