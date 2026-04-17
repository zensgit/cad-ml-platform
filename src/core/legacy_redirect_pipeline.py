from __future__ import annotations

from fastapi import HTTPException

from src.core.errors_extended import create_migration_error


def build_legacy_redirect_exception(
    *, old_path: str, new_path: str, method: str = "GET"
) -> HTTPException:
    return HTTPException(
        status_code=410,
        detail=create_migration_error(
            old_path=old_path,
            new_path=new_path,
            method=method,
        ),
    )


def raise_legacy_redirect(
    *, old_path: str, new_path: str, method: str = "GET"
) -> None:
    raise build_legacy_redirect_exception(
        old_path=old_path,
        new_path=new_path,
        method=method,
    )


__all__ = [
    "build_legacy_redirect_exception",
    "raise_legacy_redirect",
]
