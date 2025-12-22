from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.v1.vectors import (
    VectorMigrateRequest,
    migrate_vectors,
    preview_migration,
    _prepare_vector_for_upgrade,
)
from src.core.feature_extractor import FeatureExtractor
from src.core.vector_layouts import VECTOR_LAYOUT_L3, VECTOR_LAYOUT_LEGACY


def test_prepare_vector_reorders_legacy_layout():
    extractor = FeatureExtractor(feature_version="v3")
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    ext = [6.0, 7.0, 8.0, 9.0, 10.0]
    semantic = [99.0, 0.0]
    legacy = base + ext + semantic
    meta = {"vector_layout": VECTOR_LAYOUT_LEGACY}
    base_vector, l3_tail, layout = _prepare_vector_for_upgrade(extractor, legacy, meta, "v2")
    assert layout == VECTOR_LAYOUT_LEGACY
    assert l3_tail == []
    assert base_vector == base + semantic + ext


@pytest.mark.asyncio
async def test_preview_preserves_l3_tail_dimensions():
    vectors = {"vec1": [0.1] * 27}
    meta = {
        "vec1": {
            "feature_version": "v4",
            "vector_layout": VECTOR_LAYOUT_L3,
            "l3_3d_dim": "3",
        }
    }
    with (
        patch("src.core.similarity._VECTOR_STORE", vectors),
        patch("src.core.similarity._VECTOR_META", meta),
    ):
        response = await preview_migration(to_version="v3", limit=5, api_key="test")

    assert response.total_vectors == 1
    item = response.preview_items[0]
    assert item.dimension_before == 27
    assert item.dimension_after == 25


@pytest.mark.asyncio
async def test_migrate_preserves_l3_tail_and_updates_meta():
    vectors = {"vec1": [0.2] * 27}
    meta = {
        "vec1": {
            "feature_version": "v4",
            "vector_layout": VECTOR_LAYOUT_L3,
            "l3_3d_dim": "3",
        }
    }
    payload = VectorMigrateRequest(ids=["vec1"], to_version="v3", dry_run=False)
    with (
        patch("src.core.similarity._VECTOR_STORE", vectors),
        patch("src.core.similarity._VECTOR_META", meta),
    ):
        await migrate_vectors(payload, api_key="test")

    assert len(vectors["vec1"]) == 25
    assert meta["vec1"]["feature_version"] == "v3"
    assert meta["vec1"]["vector_layout"] == VECTOR_LAYOUT_L3
    assert meta["vec1"]["l3_3d_dim"] == "3"
    assert meta["vec1"]["total_dim"] == "25"
    assert meta["vec1"]["semantic_dim"] == "2"
    assert meta["vec1"]["geometric_dim"] == "20"
