from __future__ import annotations

from typing import Any, Mapping


def build_pending_run_migrate_request(
    *,
    pending: Mapping[str, Any],
    target_version: str,
    dry_run: bool,
    request_cls: type[Any],
) -> Any:
    return request_cls(
        ids=pending["pending_ids"],
        to_version=target_version,
        dry_run=dry_run,
    )


__all__ = ["build_pending_run_migrate_request"]
