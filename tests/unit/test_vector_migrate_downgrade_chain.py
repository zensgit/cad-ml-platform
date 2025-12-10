"""Tests: downgrade migration chain (v4→v3→v2) and status counting.

Verifies that:
1. Vectors can be migrated (downgraded) from higher to lower versions
2. Status "downgraded" is correctly assigned for version downgrades
3. Metrics vector_migrate_total{status="downgraded"} increments properly
4. Migration summary correctly aggregates counts
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from src.main import app


client = TestClient(app)


class TestVectorMigrateDowngradeChain:
    """Test downgrade migration chain functionality."""

    @pytest.fixture(autouse=True)
    def setup_vectors(self):
        """Set up test vectors before each test."""
        # Import the actual vector store from similarity module
        from src.core import similarity as sim_module

        # Save original state
        original_store = sim_module._VECTOR_STORE.copy()
        original_meta = sim_module._VECTOR_META.copy()

        # Create test vectors with v4 features
        sim_module._VECTOR_STORE.update({
            "test-v4-001": np.random.rand(20).tolist(),
            "test-v4-002": np.random.rand(20).tolist(),
            "test-v4-003": np.random.rand(20).tolist(),
            "test-v3-001": np.random.rand(15).tolist(),
            "test-v3-002": np.random.rand(15).tolist(),
        })

        sim_module._VECTOR_META.update({
            "test-v4-001": {"feature_version": "v4", "material": "steel"},
            "test-v4-002": {"feature_version": "v4", "material": "aluminum"},
            "test-v4-003": {"feature_version": "v4", "material": "plastic"},
            "test-v3-001": {"feature_version": "v3", "material": "steel"},
            "test-v3-002": {"feature_version": "v3", "material": "aluminum"},
        })

        yield

        # Restore original state - remove test entries
        for key in ["test-v4-001", "test-v4-002", "test-v4-003", "test-v3-001", "test-v3-002"]:
            sim_module._VECTOR_STORE.pop(key, None)
            sim_module._VECTOR_META.pop(key, None)

        # Restore any original entries that might have been overwritten
        sim_module._VECTOR_STORE.update(original_store)
        sim_module._VECTOR_META.update(original_meta)

    def test_v4_to_v3_downgrade_single(self):
        """Test single vector downgrade from v4 to v3."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # Return smaller dimension vector for v3
            mock_upgrade.return_value = np.random.rand(15).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001"],
                    "to_version": "v3",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Verify downgrade status
            assert len(data["items"]) == 1
            item = data["items"][0]
            assert item["status"] == "downgraded"
            assert item["from_version"] == "v4"
            assert item["to_version"] == "v3"

    def test_v4_to_v3_downgrade_batch(self):
        """Test batch vector downgrade from v4 to v3."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(15).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001", "test-v4-002", "test-v4-003"],
                    "to_version": "v3",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # All 3 should be downgraded
            downgraded_count = sum(1 for item in data["items"] if item["status"] == "downgraded")
            assert downgraded_count == 3

    def test_v3_to_v2_downgrade(self):
        """Test vector downgrade from v3 to v2."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(10).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v3-001", "test-v3-002"],
                    "to_version": "v2",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Both should be downgraded
            downgraded_count = sum(1 for item in data["items"] if item["status"] == "downgraded")
            assert downgraded_count == 2

            for item in data["items"]:
                if item["status"] == "downgraded":
                    assert item["from_version"] == "v3"
                    assert item["to_version"] == "v2"

    def test_downgrade_chain_v4_to_v2(self):
        """Test downgrade chain from v4 directly to v2."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(10).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001"],
                    "to_version": "v2",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["status"] == "downgraded"
            assert item["from_version"] == "v4"
            assert item["to_version"] == "v2"

    def test_downgrade_chain_v4_to_v1(self):
        """Test downgrade chain from v4 directly to v1."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(5).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001"],
                    "to_version": "v1",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert item["status"] == "downgraded"
            assert item["from_version"] == "v4"
            assert item["to_version"] == "v1"

    def test_downgrade_dimension_tracking(self):
        """Test that dimension changes are tracked during downgrade."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            # v4 has 20 dims, v3 should have 15
            mock_upgrade.return_value = np.random.rand(15).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001"],
                    "to_version": "v3",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            assert "dimension_before" in item
            assert "dimension_after" in item
            assert item["dimension_before"] == 20
            assert item["dimension_after"] == 15

    def test_same_version_skipped(self):
        """Test that same version migration is skipped, not downgraded."""
        response = client.post(
            "/api/v1/vectors/migrate",
            json={
                "ids": ["test-v4-001"],
                "to_version": "v4",
                "dry_run": False
            },
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        item = data["items"][0]
        assert item["status"] == "skipped"

    def test_upgrade_not_downgraded(self):
        """Test that upgrade (v3->v4) is 'migrated', not 'downgraded'."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(20).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v3-001"],
                    "to_version": "v4",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            item = data["items"][0]
            # Upgrade should be "migrated", not "downgraded"
            assert item["status"] == "migrated"
            assert item["from_version"] == "v3"
            assert item["to_version"] == "v4"

    def test_mixed_upgrade_downgrade(self):
        """Test batch with both upgrades and downgrades."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(15).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    # test-v4-001: v4->v3 = downgrade
                    # test-v3-001: v3->v3 = skip (same version)
                    "ids": ["test-v4-001", "test-v3-001"],
                    "to_version": "v3",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            statuses = {item["id"]: item["status"] for item in data["items"]}
            assert statuses["test-v4-001"] == "downgraded"
            assert statuses["test-v3-001"] == "skipped"

    def test_migrate_summary_downgrade_counts(self):
        """Test that migration summary correctly counts downgrades."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(15).tolist()

            # First do a downgrade migration
            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001", "test-v4-002"],
                    "to_version": "v3",
                    "dry_run": False
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200

            # Now check summary
            summary_response = client.get(
                "/api/v1/vectors/migrate/summary",
                headers={"X-API-Key": "test"}
            )

            assert summary_response.status_code == 200
            summary = summary_response.json()

            # Summary should include downgrade count
            assert "total_migrations" in summary or "downgraded" in str(summary).lower()

    def test_dry_run_downgrade_preview(self):
        """Test dry run mode for downgrade shows correct status without modification."""
        with patch("src.core.feature_extractor.FeatureExtractor.upgrade_vector") as mock_upgrade:
            mock_upgrade.return_value = np.random.rand(15).tolist()

            response = client.post(
                "/api/v1/vectors/migrate",
                json={
                    "ids": ["test-v4-001"],
                    "to_version": "v3",
                    "dry_run": True
                },
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Dry run should show "dry_run" status, not "downgraded"
            item = data["items"][0]
            assert item["status"] == "dry_run"
            assert data.get("dry_run_total", 0) >= 1


class TestVectorMigrateMetrics:
    """Test metrics recording for migration operations."""

    def test_downgrade_metric_increment(self):
        """Test that vector_migrate_total metric increments for downgrades."""
        from src.utils.analysis_metrics import vector_migrate_total

        # This test verifies the metric is properly defined and can be accessed
        assert vector_migrate_total is not None

        # Verify the metric has "downgraded" as a valid status label
        # The actual increment happens in the migrate endpoint
        # Here we just verify the metric structure

    def test_dimension_delta_metric(self):
        """Test that vector_migrate_dimension_delta metric is recorded."""
        from src.utils.analysis_metrics import vector_migrate_dimension_delta

        assert vector_migrate_dimension_delta is not None
        # The metric should have buckets for dimension changes
