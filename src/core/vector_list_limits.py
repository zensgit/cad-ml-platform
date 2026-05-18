from __future__ import annotations

import os
from collections.abc import Callable


def resolve_vector_list_limits(
    requested_limit: int,
    *,
    getenv_fn: Callable[[str, str], str | None] = os.getenv,
) -> tuple[int, int]:
    max_limit = int(getenv_fn("VECTOR_LIST_LIMIT", "200"))
    scan_limit = int(getenv_fn("VECTOR_LIST_SCAN_LIMIT", "5000"))
    return min(requested_limit, max_limit), scan_limit


__all__ = ["resolve_vector_list_limits"]
