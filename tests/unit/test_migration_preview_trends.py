"""Tests for migration preview and trends endpoints.

Verifies that:
1. Migration preview returns correct version distribution
2. Preview shows dimension changes accurately
3. Trends calculates correct rates and metrics
4. Time window filtering works
5. Edge cases are handled properly
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.api.v1.vectors import (
    VectorMigrateItem,
    VectorMigrationPreviewResponse,
    VectorMigrationTrendsResponse,
)


# Mock global history storage
def get_mock_history(entries: list[Dict[str, Any]] = None) -> list:
    """Create mock migration history."""
    if entries is None:
        return []
    return entries


class TestMigrationPreviewEndpoint:
    """Test migration preview endpoint."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store with test data."""
        vectors = {
            "vec1": [1.0] * 22,  # v3 vector (22 dimensions)
            "vec2": [2.0] * 22,  # v3 vector
            "vec3": [3.0] * 24,  # v4 vector (24 dimensions)
            "vec4": [4.0] * 7,  # v1 vector (7 dimensions)
            "vec5": [5.0] * 12,  # v2 vector (12 dimensions)
        }
        meta = {
            "vec1": {"feature_version": "v3"},
            "vec2": {"feature_version": "v3"},
            "vec3": {"feature_version": "v4"},
            "vec4": {"feature_version": "v1"},
            "vec5": {"feature_version": "v2"},
        }
        with patch("src.core.similarity._VECTOR_STORE", vectors):
            with patch("src.core.similarity._VECTOR_META", meta):
                yield vectors, meta

    @pytest.mark.asyncio
    async def test_preview_returns_version_distribution(self, mock_vector_store):
        """Preview should return accurate version distribution."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v4", limit=10, api_key="test")

        assert response.total_vectors == 5
        assert response.by_version == {"v3": 2, "v4": 1, "v1": 1, "v2": 1}

    @pytest.mark.asyncio
    async def test_preview_respects_limit(self, mock_vector_store):
        """Preview should respect limit parameter."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v4", limit=2, api_key="test")

        assert len(response.preview_items) <= 2

    @pytest.mark.asyncio
    async def test_preview_max_limit_enforced(self, mock_vector_store):
        """Preview should enforce max limit of 100."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            # Even with limit > 100, should cap at 100
            response = await preview_migration(to_version="v4", limit=500, api_key="test")

        # Should be capped but we only have 5 vectors
        assert len(response.preview_items) <= 5

    @pytest.mark.asyncio
    async def test_preview_skips_same_version(self, mock_vector_store):
        """Preview should skip vectors already at target version."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v4", limit=10, api_key="test")

        # vec3 is already v4, should be skipped
        skipped_items = [i for i in response.preview_items if i.status == "skipped"]
        assert len(skipped_items) >= 1

    @pytest.mark.asyncio
    async def test_preview_detects_upgrades(self, mock_vector_store):
        """Preview should detect upgrade operations."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v4", limit=10, api_key="test")

        upgrade_items = [i for i in response.preview_items if "upgrade" in i.status]
        # v1, v2, v3 vectors should be marked for upgrade
        assert len(upgrade_items) >= 1

    @pytest.mark.asyncio
    async def test_preview_detects_downgrades(self, mock_vector_store):
        """Preview should detect downgrade operations."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v2", limit=10, api_key="test")

        downgrade_items = [i for i in response.preview_items if "downgrade" in i.status]
        # v3, v4 vectors should be marked for downgrade
        assert len(downgrade_items) >= 1

    @pytest.mark.asyncio
    async def test_preview_dimension_changes_calculated(self, mock_vector_store):
        """Preview should calculate dimension changes."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v4", limit=10, api_key="test")

        assert "positive" in response.estimated_dimension_changes
        assert "negative" in response.estimated_dimension_changes
        assert "zero" in response.estimated_dimension_changes

    @pytest.mark.asyncio
    async def test_preview_feasibility_assessment(self, mock_vector_store):
        """Preview should assess migration feasibility."""
        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await preview_migration(to_version="v4", limit=10, api_key="test")

        assert isinstance(response.migration_feasible, bool)

    @pytest.mark.asyncio
    async def test_preview_invalid_version_rejected(self, mock_vector_store):
        """Preview should reject invalid target version."""
        from fastapi import HTTPException

        from src.api.v1.vectors import preview_migration

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            with pytest.raises(HTTPException) as exc_info:
                await preview_migration(to_version="v99", limit=10, api_key="test")

        assert exc_info.value.status_code == 422


class TestMigrationTrendsEndpoint:
    """Test migration trends endpoint."""

    @pytest.fixture
    def mock_empty_store(self):
        """Mock empty vector store."""
        with patch("src.core.similarity._VECTOR_STORE", {}):
            with patch("src.core.similarity._VECTOR_META", {}):
                yield

    @pytest.fixture
    def mock_store_with_versions(self):
        """Mock vector store with various versions."""
        vectors = {
            "vec1": [1.0] * 22,
            "vec2": [2.0] * 24,
            "vec3": [3.0] * 24,
            "vec4": [4.0] * 24,
            "vec5": [5.0] * 22,
        }
        meta = {
            "vec1": {"feature_version": "v3"},
            "vec2": {"feature_version": "v4"},
            "vec3": {"feature_version": "v4"},
            "vec4": {"feature_version": "v4"},
            "vec5": {"feature_version": "v3"},
        }
        with patch("src.core.similarity._VECTOR_STORE", vectors):
            with patch("src.core.similarity._VECTOR_META", meta):
                yield vectors, meta

    @pytest.mark.asyncio
    async def test_trends_empty_history(self, mock_empty_store):
        """Trends should handle empty history."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        # Clear history
        vectors_module._VECTOR_MIGRATION_HISTORY = []

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=24, api_key="test")

        assert response.total_migrations == 0
        assert response.success_rate == 0.0  # No attempts = 0 rate (calculated as 0/1)
        assert response.v4_adoption_rate == 0.0
        assert response.window_hours == 24

    @pytest.mark.asyncio
    async def test_trends_with_history(self, mock_store_with_versions):
        """Trends should calculate stats from history."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        # Set up history
        now = datetime.utcnow()
        vectors_module._VECTOR_MIGRATION_HISTORY = [
            {
                "migration_id": "test-1",
                "started_at": now.isoformat(),
                "finished_at": now.isoformat(),
                "total": 10,
                "migrated": 8,
                "skipped": 2,
                "counts": {
                    "migrated": 8,
                    "skipped": 2,
                    "downgraded": 0,
                    "error": 0,
                    "not_found": 0,
                },
            }
        ]

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=24, api_key="test")

        assert response.total_migrations == 10
        assert response.success_rate == 1.0  # 8 migrated, 0 errors

    @pytest.mark.asyncio
    async def test_trends_v4_adoption_rate(self, mock_store_with_versions):
        """Trends should calculate v4 adoption rate correctly."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        vectors_module._VECTOR_MIGRATION_HISTORY = []

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=24, api_key="test")

        # 3 out of 5 vectors are v4
        assert response.v4_adoption_rate == 0.6
        assert response.version_distribution == {"v3": 2, "v4": 3}

    @pytest.mark.asyncio
    async def test_trends_time_window_filtering(self, mock_store_with_versions):
        """Trends should filter by time window."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        now = datetime.utcnow()
        old_time = now - timedelta(hours=48)

        vectors_module._VECTOR_MIGRATION_HISTORY = [
            {
                "migration_id": "old",
                "started_at": old_time.isoformat(),
                "total": 100,
                "counts": {"migrated": 100},
            },
            {
                "migration_id": "new",
                "started_at": now.isoformat(),
                "total": 10,
                "counts": {"migrated": 10},
            },
        ]

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            # Only last 24 hours
            response = await migrate_trends(window_hours=24, api_key="test")

        # Should only include the new migration
        assert response.total_migrations == 10

    @pytest.mark.asyncio
    async def test_trends_error_rate(self, mock_store_with_versions):
        """Trends should calculate error rate correctly."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        now = datetime.utcnow()
        vectors_module._VECTOR_MIGRATION_HISTORY = [
            {
                "migration_id": "test",
                "started_at": now.isoformat(),
                "total": 10,
                "counts": {
                    "migrated": 6,
                    "skipped": 0,
                    "downgraded": 0,
                    "error": 2,
                    "not_found": 2,
                },
            }
        ]

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=24, api_key="test")

        # 4 errors out of 10 attempted
        assert response.error_rate == 0.4

    @pytest.mark.asyncio
    async def test_trends_downgrade_rate(self, mock_store_with_versions):
        """Trends should calculate downgrade rate correctly."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        now = datetime.utcnow()
        vectors_module._VECTOR_MIGRATION_HISTORY = [
            {
                "migration_id": "test",
                "started_at": now.isoformat(),
                "total": 10,
                "counts": {
                    "migrated": 5,
                    "skipped": 0,
                    "downgraded": 5,
                    "error": 0,
                    "not_found": 0,
                },
            }
        ]

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=24, api_key="test")

        # 5 downgrades out of 10 attempted
        assert response.downgrade_rate == 0.5

    @pytest.mark.asyncio
    async def test_trends_migration_velocity(self, mock_store_with_versions):
        """Trends should calculate migration velocity correctly."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        now = datetime.utcnow()
        vectors_module._VECTOR_MIGRATION_HISTORY = [
            {
                "migration_id": "test1",
                "started_at": now.isoformat(),
                "total": 24,
                "counts": {"migrated": 24},
            }
        ]

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=24, api_key="test")

        # 24 migrations / 24 hours = 1 per hour
        assert response.migration_velocity == 1.0

    @pytest.mark.asyncio
    async def test_trends_time_range_returned(self, mock_store_with_versions):
        """Trends should return time range."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        vectors_module._VECTOR_MIGRATION_HISTORY = []

        with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
            response = await migrate_trends(window_hours=12, api_key="test")

        assert response.window_hours == 12
        assert response.time_range is not None
        assert "start" in response.time_range
        assert "end" in response.time_range


class TestMigrationPreviewEdgeCases:
    """Test edge cases for migration preview."""

    @pytest.mark.asyncio
    async def test_preview_empty_store(self):
        """Preview should handle empty vector store."""
        from src.api.v1.vectors import preview_migration

        with patch("src.core.similarity._VECTOR_STORE", {}):
            with patch("src.core.similarity._VECTOR_META", {}):
                with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
                    response = await preview_migration(to_version="v4", limit=10, api_key="test")

        assert response.total_vectors == 0
        assert response.by_version == {}
        assert len(response.preview_items) == 0

    @pytest.mark.asyncio
    async def test_preview_all_same_version(self):
        """Preview should handle all vectors at target version."""
        from src.api.v1.vectors import preview_migration

        vectors = {"vec1": [1.0] * 24, "vec2": [2.0] * 24}
        meta = {"vec1": {"feature_version": "v4"}, "vec2": {"feature_version": "v4"}}

        with patch("src.core.similarity._VECTOR_STORE", vectors):
            with patch("src.core.similarity._VECTOR_META", meta):
                with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
                    response = await preview_migration(to_version="v4", limit=10, api_key="test")

        # All should be skipped
        assert all(item.status == "skipped" for item in response.preview_items)


class TestMigrationTrendsEdgeCases:
    """Test edge cases for migration trends."""

    @pytest.mark.asyncio
    async def test_trends_zero_window_hours(self):
        """Trends should handle zero window hours."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        vectors_module._VECTOR_MIGRATION_HISTORY = []

        with patch("src.core.similarity._VECTOR_STORE", {}):
            with patch("src.core.similarity._VECTOR_META", {}):
                with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
                    response = await migrate_trends(window_hours=0, api_key="test")

        assert response.window_hours == 0

    @pytest.mark.asyncio
    async def test_trends_invalid_history_entries(self):
        """Trends should handle invalid history entries."""
        import src.api.v1.vectors as vectors_module
        from src.api.v1.vectors import migrate_trends

        vectors_module._VECTOR_MIGRATION_HISTORY = [
            {"migration_id": "bad", "started_at": "invalid-date"},
            {"migration_id": "missing"},
        ]

        with patch("src.core.similarity._VECTOR_STORE", {}):
            with patch("src.core.similarity._VECTOR_META", {}):
                with patch("src.api.v1.vectors.get_api_key", return_value="test-key"):
                    # Should not raise
                    response = await migrate_trends(window_hours=24, api_key="test")

        assert response is not None
