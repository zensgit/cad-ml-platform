"""RFC 8785 boundary vectors used by ReviewReuse ledger digests."""

from __future__ import annotations

import pytest

from src.core.review_reuse.canonical import (
    CanonicalJSONError,
    canonical_json_v1,
    strict_json_loads,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (333333333.33333329, b"333333333.3333333"),
        (1e30, b"1e+30"),
        (4.50, b"4.5"),
        (2e-3, b"0.002"),
        (1e-27, b"1e-27"),
        (-0.0, b"0"),
        (1e20, b"100000000000000000000"),
        (1e21, b"1e+21"),
        (1e-6, b"0.000001"),
        (1e-7, b"1e-7"),
    ],
)
def test_canonical_number_vectors(value: float, expected: bytes) -> None:
    assert canonical_json_v1(value) == expected


def test_object_keys_use_utf16_sort_order() -> None:
    value = {"\ue000": "private", "😀": "astral"}
    assert canonical_json_v1(value) == (
        '{"😀":"astral","\ue000":"private"}'.encode("utf-8")
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_json_v1(value)


def test_non_i_json_values_are_rejected() -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_json_v1(9_007_199_254_740_992)
    with pytest.raises(CanonicalJSONError):
        canonical_json_v1("\ud800")


@pytest.mark.parametrize(
    "payload",
    [
        '{"task_id":"first","task_id":"second"}',
        '{"value":NaN}',
        '{"value":9007199254740992}',
    ],
)
def test_strict_json_parser_rejects_ambiguous_or_non_i_json(payload: str) -> None:
    with pytest.raises(CanonicalJSONError):
        strict_json_loads(payload)
