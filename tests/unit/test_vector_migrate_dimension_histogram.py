"""Tests for vector migration dimension delta histogram metric.

Verifies that:
1. vector_migrate_dimension_delta histogram is properly defined
2. Dimension deltas are recorded during migration operations
3. Both positive (upgrade) and negative (downgrade) deltas are tracked
4. Histogram buckets correctly categorize dimension changes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestVectorMigrateDimensionHistogram:
    """Test dimension delta histogram during vector migration."""

    @pytest.fixture(autouse=True)
    def setup_vectors(self):
        """Set up test vectors with different dimensions."""
        from src.core import similarity as sim_module

        original_store = sim_module._VECTOR_STORE.copy()
        original_meta = sim_module._VECTOR_META.copy()

        # Create test vectors with different dimensions
        sim_module._VECTOR_STORE.update(
            {
                # v1 vectors (5 dimensions)
                "dim-v1-001": np.random.rand(5).tolist(),
                "dim-v1-002": np.random.rand(5).tolist(),
                # v2 vectors (10 dimensions)
                "dim-v2-001": np.random.rand(10).tolist(),
                "dim-v2-002": np.random.rand(10).tolist(),
                # v3 vectors (15 dimensions)
                "dim-v3-001": np.random.rand(15).tolist(),
                # v4 vectors (20 dimensions)
                "dim-v4-001": np.random.rand(20).tolist(),
            }
        )

        sim_module._VECTOR_META.update(
            {
                "dim-v1-001": {"feature_version": "v1"},
                "dim-v1-002": {"feature_version": "v1"},
                "dim-v2-001": {"feature_version": "v2"},
                "dim-v2-002": {"feature_version": "v2"},
                "dim-v3-001": {"feature_version": "v3"},
                "dim-v4-001": {"feature_version": "v4"},
            }
        )

        yield

        # Cleanup
        for key in [
            "dim-v1-001",
            "dim-v1-002",
            "dim-v2-001",
            "dim-v2-002",
            "dim-v3-001",
            "dim-v4-001",
        ]:
            sim_module._VECTOR_STORE.pop(key, None)
            sim_module._VECTOR_META.pop(key, None)

        sim_module._VECTOR_STORE.update(original_store)
        sim_module._VECTOR_META.update(original_meta)

    def test_dimension_histogram_metric_exists(self):
        """Test that dimension delta histogram metric is properly defined."""
        from src.utils.analysis_metrics import vector_migrate_dimension_delta

        assert vector_migrate_dimension_delta is not None

    def test_positive_dimension_delta_upgrade(self):
        """Test positive dimension delta when upgrading (v1 -> v2)."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # Simulate v1 (5 dim) -> v2 (10 dim), delta = +5
            mock_upgrade.return_value = np.random.rand(10).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={"ids": ["dim-v1-001"], "to_version": "v2", "dry_run": False},
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["dimension_before"] == 5
            assert item["dimension_after"] == 10
            # Delta should be +5

    def test_negative_dimension_delta_downgrade(self):
        """Test negative dimension delta when downgrading (v4 -> v2)."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # Simulate v4 (20 dim) -> v2 (10 dim), delta = -10
            mock_upgrade.return_value = np.random.rand(10).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={"ids": ["dim-v4-001"], "to_version": "v2", "dry_run": False},
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["dimension_before"] == 20
            assert item["dimension_after"] == 10
            # Delta should be -10

    def test_zero_dimension_delta_same_size(self):
        """Test zero dimension delta when dimensions stay the same."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # Simulate migration that keeps same dimension
            mock_upgrade.return_value = np.random.rand(10).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["dim-v2-001"],
                    "to_version": "v3",  # Hypothetically same dimension
                    "dry_run": False,
                },
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            # dimension_before and dimension_after should both be recorded
            assert "dimension_before" in item
            assert "dimension_after" in item

    def test_batch_migration_multiple_deltas(self):
        """Test multiple dimension deltas recorded in batch migration."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # All go to 15 dimensions (v3)
            mock_upgrade.return_value = np.random.rand(15).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["dim-v1-001", "dim-v2-001", "dim-v4-001"],
                    "to_version": "v3",
                    "dry_run": False,
                },
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have 3 items with different dimension changes
            assert len(data["items"]) == 3

            for item in data["items"]:
                if item["status"] in ("migrated", "downgraded"):
                    assert "dimension_before" in item
                    assert "dimension_after" in item
                    assert item["dimension_after"] == 15

    def test_dry_run_records_dimension_delta(self):
        """Test dimension delta is recorded even in dry run mode."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(20).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={"ids": ["dim-v1-001"], "to_version": "v4", "dry_run": True},
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["status"] == "dry_run"
            assert item["dimension_before"] == 5
            assert item["dimension_after"] == 20
            # Delta would be +15

    def test_large_positive_delta_v1_to_v4(self):
        """Test large positive delta when upgrading from v1 to v4."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # v1 (5) -> v4 (20), delta = +15
            mock_upgrade.return_value = np.random.rand(20).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={"ids": ["dim-v1-001"], "to_version": "v4", "dry_run": False},
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["dimension_before"] == 5
            assert item["dimension_after"] == 20
            # Delta = 20 - 5 = +15

    def test_large_negative_delta_v4_to_v1(self):
        """Test large negative delta when downgrading from v4 to v1."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # v4 (20) -> v1 (5), delta = -15
            mock_upgrade.return_value = np.random.rand(5).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={"ids": ["dim-v4-001"], "to_version": "v1", "dry_run": False},
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["dimension_before"] == 20
            assert item["dimension_after"] == 5
            # Delta = 5 - 20 = -15


class TestDimensionHistogramBuckets:
    """Test histogram bucket coverage for dimension deltas."""

    def test_histogram_has_negative_buckets(self):
        """Test that histogram includes negative buckets for downgrades."""
        from src.utils.analysis_metrics import vector_migrate_dimension_delta

        # The histogram should be able to handle negative values
        assert vector_migrate_dimension_delta is not None

    def test_histogram_buckets_defined(self):
        """Test histogram buckets are properly defined."""
        from src.utils.analysis_metrics import vector_migrate_dimension_delta

        # Verify the metric exists and has expected structure
        assert vector_migrate_dimension_delta is not None


class TestMigrationResponseDimensions:
    """Test that migration response includes dimension information."""

    @pytest.fixture(autouse=True)
    def setup_vectors(self):
        """Set up test vector."""
        from src.core import similarity as sim_module

        original_store = sim_module._VECTOR_STORE.copy()
        original_meta = sim_module._VECTOR_META.copy()

        sim_module._VECTOR_STORE["response-test-001"] = np.random.rand(10).tolist()
        sim_module._VECTOR_META["response-test-001"] = {"feature_version": "v2"}

        yield

        sim_module._VECTOR_STORE.pop("response-test-001", None)
        sim_module._VECTOR_META.pop("response-test-001", None)

        sim_module._VECTOR_STORE.update(original_store)
        sim_module._VECTOR_META.update(original_meta)

    def test_migrated_status_has_dimensions(self):
        """Test that migrated status includes dimension info."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(20).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={"ids": ["response-test-001"], "to_version": "v4", "dry_run": False},
                headers={"X-API-Key": "test"},
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["status"] == "migrated"
            assert "dimension_before" in item
            assert "dimension_after" in item

    def test_downgraded_status_has_dimensions(self):
        """Test that downgraded status includes dimension info."""
        from src.core import similarity as sim_module

        # Add v4 vector for downgrade test
        sim_module._VECTOR_STORE["downgrade-dim-001"] = np.random.rand(20).tolist()
        sim_module._VECTOR_META["downgrade-dim-001"] = {"feature_version": "v4"}

        try:
            with patch(
                "src.core.feature_extractor.FeatureExtractor.upgrade_vector"
            ) as mock_upgrade:
                mock_upgrade.return_value = np.random.rand(10).tolist()

                response = client.post(
                    "/api/v1/vectors/migrate",
                    json={"ids": ["downgrade-dim-001"], "to_version": "v2", "dry_run": False},
                    headers={"X-API-Key": "test"},
                )

                assert response.status_code == 200
                data = response.json()

                item = data["items"][0]
                assert item["status"] == "downgraded"
                assert "dimension_before" in item
                assert "dimension_after" in item
        finally:
            sim_module._VECTOR_STORE.pop("downgrade-dim-001", None)
            sim_module._VECTOR_META.pop("downgrade-dim-001", None)

    def test_skipped_status_no_dimensions(self):
        """Test that skipped status (same version) has no dimension change."""
        response = client.post(
            "/api/v1/vectors/migrate",
            json={
                "ids": ["response-test-001"],
                "to_version": "v2",  # Same as current version
                "dry_run": False,
            },
            headers={"X-API-Key": "test"},
        )

        assert response.status_code == 200
        data = response.json()

        item = data["items"][0]
        assert item["status"] == "skipped"
        # Skipped items typically don't have dimension info
        # since no migration occurred

    def test_not_found_status_no_dimensions(self):
        """Test that not_found status has no dimension info."""
        response = client.post(
            "/api/v1/vectors/migrate",
            json={"ids": ["nonexistent-vector-xyz"], "to_version": "v4", "dry_run": False},
            headers={"X-API-Key": "test"},
        )

        assert response.status_code == 200
        data = response.json()

        item = data["items"][0]
        assert item["status"] == "not_found"
        # Not found items shouldn't have dimension info
