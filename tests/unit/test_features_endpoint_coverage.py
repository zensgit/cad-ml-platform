"""Tests for src/api/v1/features.py to improve coverage.

Covers:
- features_diff endpoint with all status branches
- feature_slots endpoint
- feature_versions endpoint
- Error handling paths (not_found, dimension_mismatch)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


class TestFeaturesDiffSuccess:
    """Tests for features_diff success path."""

    @pytest.mark.asyncio
    async def test_features_diff_success(self):
        """Test successful features diff."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [1.0, 2.0, 3.0],
            "id_b": [1.5, 2.0, 3.5],
        }
        test_meta = {
            "id_a": {"feature_version": "v1"},
            "id_b": {"feature_version": "v1"},
        }

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                with patch("src.core.feature_extractor.FeatureExtractor") as mock_extractor_cls:
                    mock_extractor = MagicMock()
                    mock_extractor.slots.return_value = [
                        {"name": "feature_0"},
                        {"name": "feature_1"},
                        {"name": "feature_2"},
                    ]
                    mock_extractor_cls.return_value = mock_extractor

                    result = await features_diff("id_a", "id_b", api_key="test")

        assert result.status == "ok"
        assert result.id_a == "id_a"
        assert result.id_b == "id_b"
        assert result.dimension == 3
        assert len(result.diffs) == 3


class TestFeaturesDiffNotFound:
    """Tests for features_diff not_found path (lines 64-78)."""

    @pytest.mark.asyncio
    async def test_features_diff_first_vector_not_found(self):
        """Test features diff when first vector not found."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_b": [1.0, 2.0, 3.0],
        }
        test_meta = {}

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                result = await features_diff("id_a", "id_b", api_key="test")

        assert result.status == "not_found"
        assert result.error is not None
        assert result.dimension is None
        assert result.diffs == []

    @pytest.mark.asyncio
    async def test_features_diff_second_vector_not_found(self):
        """Test features diff when second vector not found."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [1.0, 2.0, 3.0],
        }
        test_meta = {}

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                result = await features_diff("id_a", "id_b", api_key="test")

        assert result.status == "not_found"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_features_diff_both_vectors_not_found(self):
        """Test features diff when both vectors not found."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store: Dict[str, List[float]] = {}
        test_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                result = await features_diff("id_a", "id_b", api_key="test")

        assert result.status == "not_found"


class TestFeaturesDiffDimensionMismatch:
    """Tests for features_diff dimension_mismatch path (lines 86-99)."""

    @pytest.mark.asyncio
    async def test_features_diff_dimension_mismatch(self):
        """Test features diff with dimension mismatch."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [1.0, 2.0, 3.0],  # 3 dimensions
            "id_b": [1.0, 2.0],  # 2 dimensions
        }
        test_meta = {}

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                result = await features_diff("id_a", "id_b", api_key="test")

        assert result.status == "dimension_mismatch"
        assert result.error is not None
        assert result.dimension is None
        assert result.diffs == []


class TestFeaturesDiffCalculation:
    """Tests for features diff calculation logic."""

    @pytest.mark.asyncio
    async def test_features_diff_relative_diff_calculation(self):
        """Test relative diff calculation."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [10.0, 0.0, 5.0],  # Second value is 0 for edge case
            "id_b": [15.0, 1.0, 5.0],
        }
        test_meta = {}

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                with patch("src.core.feature_extractor.FeatureExtractor") as mock_extractor_cls:
                    mock_extractor = MagicMock()
                    mock_extractor.slots.return_value = [
                        {"name": "f0"},
                        {"name": "f1"},
                        {"name": "f2"},
                    ]
                    mock_extractor_cls.return_value = mock_extractor

                    result = await features_diff("id_a", "id_b", api_key="test")

        assert result.status == "ok"
        # Should be sorted by abs_diff descending
        assert result.diffs[0].abs_diff >= result.diffs[1].abs_diff

    @pytest.mark.asyncio
    async def test_features_diff_zero_value_handling(self):
        """Test handling when value_a is zero (rel_diff should be None)."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [0.0, 1.0],
            "id_b": [1.0, 2.0],
        }
        test_meta = {}

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                with patch("src.core.feature_extractor.FeatureExtractor") as mock_extractor_cls:
                    mock_extractor = MagicMock()
                    mock_extractor.slots.return_value = [{"name": "f0"}, {"name": "f1"}]
                    mock_extractor_cls.return_value = mock_extractor

                    result = await features_diff("id_a", "id_b", api_key="test")

        # Find the diff where value_a was 0
        zero_diff = next((d for d in result.diffs if d.value_a == 0.0), None)
        assert zero_diff is not None
        assert zero_diff.rel_diff is None


class TestFeaturesDiffFeatureVersion:
    """Tests for feature version handling in features_diff."""

    @pytest.mark.asyncio
    async def test_features_diff_uses_meta_version(self):
        """Test features diff uses version from metadata."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [1.0, 2.0],
            "id_b": [1.5, 2.5],
        }
        test_meta = {
            "id_a": {"feature_version": "v2"},
            "id_b": {"feature_version": "v2"},
        }

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                with patch("src.core.feature_extractor.FeatureExtractor") as mock_extractor_cls:
                    mock_extractor = MagicMock()
                    mock_extractor.slots.return_value = [{"name": "f0"}, {"name": "f1"}]
                    mock_extractor_cls.return_value = mock_extractor

                    await features_diff("id_a", "id_b", api_key="test")

                    # Verify extractor was called with correct version
                    mock_extractor.slots.assert_called_with("v2")

    @pytest.mark.asyncio
    async def test_features_diff_fallback_to_env_version(self):
        """Test features diff falls back to env var for version."""
        from src.api.v1.features import features_diff
        from src.core import similarity

        test_store = {
            "id_a": [1.0, 2.0],
            "id_b": [1.5, 2.5],
        }
        test_meta: Dict[str, Any] = {}  # No version in metadata

        with patch.object(similarity, "_VECTOR_STORE", test_store):
            with patch.object(similarity, "_VECTOR_META", test_meta):
                with patch.dict("os.environ", {"FEATURE_VERSION": "v3"}):
                    with patch("src.core.feature_extractor.FeatureExtractor") as mock_extractor_cls:
                        mock_extractor = MagicMock()
                        mock_extractor.slots.return_value = [{"name": "f0"}, {"name": "f1"}]
                        mock_extractor_cls.return_value = mock_extractor

                        await features_diff("id_a", "id_b", api_key="test")

                        mock_extractor.slots.assert_called_with("v3")


class TestFeatureSlots:
    """Tests for feature_slots endpoint."""

    @pytest.mark.asyncio
    async def test_feature_slots_v1(self):
        """Test feature slots for v1."""
        from src.api.v1.features import feature_slots

        with patch("src.core.feature_extractor.FeatureExtractor") as mock_extractor_cls:
            mock_extractor = MagicMock()
            mock_extractor.slots.return_value = [
                {"name": "area"},
                {"name": "perimeter"},
            ]
            mock_extractor_cls.return_value = mock_extractor

            result = await feature_slots(version="v1", api_key="test")

        assert result.status == "ok"
        assert result.version == "v1"
        assert len(result.slots) == 2

    @pytest.mark.asyncio
    async def test_feature_slots_unsupported_version(self):
        """Test feature slots with unsupported version."""
        from fastapi import HTTPException
        from src.api.v1.features import feature_slots

        with pytest.raises(HTTPException) as exc_info:
            await feature_slots(version="v99", api_key="test")

        assert exc_info.value.status_code == 422


class TestFeatureVersions:
    """Tests for feature_versions endpoint."""

    @pytest.mark.asyncio
    async def test_feature_versions_returns_all(self):
        """Test feature versions returns all versions."""
        from src.api.v1.features import feature_versions

        result = await feature_versions(api_key="test")

        assert result.status == "ok"
        assert len(result.versions) == 4

        versions = [v["version"] for v in result.versions]
        assert "v1" in versions
        assert "v2" in versions
        assert "v3" in versions
        assert "v4" in versions

    @pytest.mark.asyncio
    async def test_feature_versions_stability_flags(self):
        """Test feature versions has correct stability flags."""
        from src.api.v1.features import feature_versions

        result = await feature_versions(api_key="test")

        v1 = next(v for v in result.versions if v["version"] == "v1")
        v4 = next(v for v in result.versions if v["version"] == "v4")

        assert v1["stable"] is True
        assert v1["experimental"] is False
        assert v4["stable"] is False
        assert v4["experimental"] is True


class TestFeatureSlotDiff:
    """Tests for FeatureSlotDiff model."""

    def test_feature_slot_diff_creation(self):
        """Test FeatureSlotDiff model creation."""
        from src.api.v1.features import FeatureSlotDiff

        diff = FeatureSlotDiff(
            index=0,
            name="area",
            value_a=1.0,
            value_b=2.0,
            abs_diff=1.0,
            rel_diff=100.0,
        )

        assert diff.index == 0
        assert diff.name == "area"
        assert diff.abs_diff == 1.0

    def test_feature_slot_diff_no_rel_diff(self):
        """Test FeatureSlotDiff with no relative diff."""
        from src.api.v1.features import FeatureSlotDiff

        diff = FeatureSlotDiff(
            index=0,
            name="slot",
            value_a=0.0,
            value_b=1.0,
            abs_diff=1.0,
            rel_diff=None,
        )

        assert diff.rel_diff is None


class TestFeaturesDiffResponse:
    """Tests for FeaturesDiffResponse model."""

    def test_features_diff_response_ok(self):
        """Test FeaturesDiffResponse for ok status."""
        from src.api.v1.features import FeaturesDiffResponse

        resp = FeaturesDiffResponse(
            id_a="a",
            id_b="b",
            dimension=3,
            diffs=[],
            status="ok",
        )

        assert resp.status == "ok"
        assert resp.error is None

    def test_features_diff_response_error(self):
        """Test FeaturesDiffResponse for error status."""
        from src.api.v1.features import FeaturesDiffResponse

        resp = FeaturesDiffResponse(
            id_a="a",
            id_b="b",
            dimension=None,
            diffs=[],
            status="not_found",
            error={"code": "DATA_NOT_FOUND"},
        )

        assert resp.status == "not_found"
        assert resp.error is not None


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_exists(self):
        """Test router is exported."""
        from src.api.v1.features import router

        assert router is not None

    def test_router_has_diff_route(self):
        """Test router has diff route."""
        from src.api.v1.features import router

        routes = [r.path for r in router.routes]
        assert "/diff" in routes

    def test_router_has_slots_route(self):
        """Test router has slots route."""
        from src.api.v1.features import router

        routes = [r.path for r in router.routes]
        assert "/slots" in routes

    def test_router_has_versions_route(self):
        """Test router has versions route."""
        from src.api.v1.features import router

        routes = [r.path for r in router.routes]
        assert "/versions" in routes
