"""Unit + facade-compat tests for the extracted dedup 2D gating helpers."""

from __future__ import annotations

import pytest

import src.api.v1.dedup as dedup_router
from src.core.dedup_2d_gating import (
    _VERSION_GATE_MODES,
    _extract_file_stem_key,
    _extract_meta_drawing_key,
    _normalize_weights,
)


def test_helpers_reexported_from_router_module() -> None:
    assert dedup_router._VERSION_GATE_MODES is _VERSION_GATE_MODES
    assert dedup_router._normalize_weights is _normalize_weights
    assert dedup_router._extract_meta_drawing_key is _extract_meta_drawing_key
    assert dedup_router._extract_file_stem_key is _extract_file_stem_key


def test_version_gate_modes() -> None:
    assert _VERSION_GATE_MODES == {"off", "auto", "file_name", "meta"}


def test_normalize_weights() -> None:
    assert _normalize_weights(3, 1) == (0.75, 0.25)
    with pytest.raises(ValueError):
        _normalize_weights(0, 0)
    with pytest.raises(ValueError):
        _normalize_weights(-1, 1)


def test_extract_meta_drawing_key() -> None:
    assert _extract_meta_drawing_key({"meta": {"drawing_number": " D-100 "}}) == "D-100"
    assert _extract_meta_drawing_key({"meta": {"drawingNo": "X9"}}) == "X9"
    assert _extract_meta_drawing_key({"meta": {}}) is None
    assert _extract_meta_drawing_key({}) is None


def test_extract_file_stem_key() -> None:
    assert _extract_file_stem_key("part_v2.dxf") == "part"
    assert _extract_file_stem_key("part.dxf") is None  # no version suffix
    assert _extract_file_stem_key(None) is None
