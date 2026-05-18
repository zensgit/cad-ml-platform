from __future__ import annotations

import pytest

from src.api.v1 import vectors as vectors_module
from src.core.vector_layouts import VECTOR_LAYOUT_L3, VECTOR_LAYOUT_LEGACY
from src.core.vector_migration_upgrade import prepare_vector_for_upgrade


class _Extractor:
    def expected_dim(self, from_version: str) -> int:
        assert from_version == "v2"
        return 24

    def reorder_legacy_vector(self, vector: list[float], from_version: str) -> list[float]:
        assert from_version == "v2"
        return list(reversed(vector))


def test_prepare_vector_for_upgrade_reorders_legacy_layout() -> None:
    vector = [1.0, 2.0, 3.0]
    base_vector, l3_tail, layout = prepare_vector_for_upgrade(
        _Extractor(),
        vector,
        {"vector_layout": VECTOR_LAYOUT_LEGACY},
        "v2",
    )

    assert layout == VECTOR_LAYOUT_LEGACY
    assert l3_tail == []
    assert base_vector == [3.0, 2.0, 1.0]


def test_prepare_vector_for_upgrade_preserves_l3_tail() -> None:
    vector = [0.1] * 24 + [0.2, 0.3, 0.4]

    base_vector, l3_tail, layout = prepare_vector_for_upgrade(
        _Extractor(),
        vector,
        {"vector_layout": VECTOR_LAYOUT_L3, "l3_3d_dim": "3"},
        "v2",
    )

    assert layout == VECTOR_LAYOUT_L3
    assert base_vector == [0.1] * 24
    assert l3_tail == [0.2, 0.3, 0.4]


def test_prepare_vector_for_upgrade_infers_l3_tail_dimension() -> None:
    vector = [0.1] * 24 + [0.2, 0.3]

    base_vector, l3_tail, layout = prepare_vector_for_upgrade(
        _Extractor(),
        vector,
        {"vector_layout": VECTOR_LAYOUT_L3},
        "v2",
    )

    assert layout == VECTOR_LAYOUT_L3
    assert base_vector == [0.1] * 24
    assert l3_tail == [0.2, 0.3]


def test_prepare_vector_for_upgrade_rejects_invalid_l3_layout() -> None:
    with pytest.raises(ValueError, match="L3 layout length mismatch"):
        prepare_vector_for_upgrade(
            _Extractor(),
            [0.1] * 24,
            {"vector_layout": VECTOR_LAYOUT_L3, "l3_3d_dim": "3"},
            "v2",
        )


def test_vectors_facade_preserves_prepare_vector_for_upgrade_export() -> None:
    assert vectors_module._prepare_vector_for_upgrade is prepare_vector_for_upgrade
