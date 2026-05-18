from __future__ import annotations

import pytest

from src.core.vector_list_limits import resolve_vector_list_limits


def test_resolve_vector_list_limits_uses_defaults() -> None:
    values: dict[str, str] = {}

    limit, scan_limit = resolve_vector_list_limits(
        250,
        getenv_fn=lambda key, default: values.get(key, default),
    )

    assert limit == 200
    assert scan_limit == 5000


def test_resolve_vector_list_limits_honors_env_values() -> None:
    values = {
        "VECTOR_LIST_LIMIT": "50",
        "VECTOR_LIST_SCAN_LIMIT": "123",
    }

    limit, scan_limit = resolve_vector_list_limits(
        40,
        getenv_fn=lambda key, default: values.get(key, default),
    )

    assert limit == 40
    assert scan_limit == 123


def test_resolve_vector_list_limits_preserves_invalid_env_failure() -> None:
    values = {
        "VECTOR_LIST_LIMIT": "bad",
        "VECTOR_LIST_SCAN_LIMIT": "5000",
    }

    with pytest.raises(ValueError):
        resolve_vector_list_limits(
            10,
            getenv_fn=lambda key, default: values.get(key, default),
        )


def test_resolve_vector_list_limits_preserves_invalid_scan_env_failure() -> None:
    values = {
        "VECTOR_LIST_LIMIT": "200",
        "VECTOR_LIST_SCAN_LIMIT": "bad",
    }

    with pytest.raises(ValueError):
        resolve_vector_list_limits(
            10,
            getenv_fn=lambda key, default: values.get(key, default),
        )
