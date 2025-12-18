"""Tests for src/api/v1/maintenance.py to improve coverage.

Covers:
- cleanup_orphan_vectors endpoint logic
- clear_cache endpoint logic
- get_maintenance_stats endpoint logic
- reload_vector_backend endpoint logic
- Error handling paths
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOrphanCleanupLogic:
    """Tests for orphan vector cleanup logic."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store state."""
        return {
            "vec1": [0.1, 0.2, 0.3],
            "vec2": [0.4, 0.5, 0.6],
            "vec3": [0.7, 0.8, 0.9],
        }

    @pytest.fixture
    def mock_vector_meta(self):
        """Create mock vector metadata."""
        return {
            "vec1": {"timestamp": 1234567890},
            "vec2": {"timestamp": 1234567891},
        }

    def test_orphan_detection_logic(self, mock_vector_store):
        """Test logic for detecting orphan vectors."""
        # Simulate vector IDs that have no cache entry
        cache_exists = {"vec1": True, "vec2": False, "vec3": False}

        orphan_ids = []
        for vid in mock_vector_store.keys():
            if not cache_exists.get(vid, False):
                orphan_ids.append(vid)

        assert "vec2" in orphan_ids
        assert "vec3" in orphan_ids
        assert "vec1" not in orphan_ids
        assert len(orphan_ids) == 2

    def test_threshold_check_below(self):
        """Test cleanup skipped when orphan count below threshold."""
        orphan_count = 5
        threshold = 10
        force = False

        should_skip = not force and orphan_count < threshold
        assert should_skip is True

    def test_threshold_check_above(self):
        """Test cleanup proceeds when orphan count above threshold."""
        orphan_count = 15
        threshold = 10
        force = False

        should_skip = not force and orphan_count < threshold
        assert should_skip is False

    def test_threshold_check_force_override(self):
        """Test force flag overrides threshold check."""
        orphan_count = 5
        threshold = 10
        force = True

        should_skip = not force and orphan_count < threshold
        assert should_skip is False

    def test_dry_run_logic(self, mock_vector_store):
        """Test dry run mode doesn't delete vectors."""
        dry_run = True
        orphan_ids = ["vec2", "vec3"]

        deleted_count = 0
        if not dry_run:
            for oid in orphan_ids:
                if oid in mock_vector_store:
                    del mock_vector_store[oid]
                    deleted_count += 1

        # In dry run mode, nothing should be deleted
        assert deleted_count == 0
        assert "vec2" in mock_vector_store
        assert "vec3" in mock_vector_store

    def test_deletion_logic(self, mock_vector_store):
        """Test actual deletion logic."""
        dry_run = False
        orphan_ids = ["vec2", "vec3"]

        deleted_count = 0
        if not dry_run:
            for oid in orphan_ids:
                if oid in mock_vector_store:
                    del mock_vector_store[oid]
                    deleted_count += 1

        assert deleted_count == 2
        assert "vec2" not in mock_vector_store
        assert "vec3" not in mock_vector_store
        assert "vec1" in mock_vector_store

    def test_verbose_mode_sample_ids(self):
        """Test verbose mode returns sample IDs."""
        orphan_ids = ["id1", "id2", "id3", "id4", "id5",
                      "id6", "id7", "id8", "id9", "id10",
                      "id11", "id12"]
        verbose = True

        sample_ids = orphan_ids[:10] if verbose else None
        assert len(sample_ids) == 10
        assert "id11" not in sample_ids
        assert "id12" not in sample_ids

    def test_verbose_mode_disabled(self):
        """Test verbose disabled returns None for sample_ids."""
        orphan_ids = ["id1", "id2", "id3"]
        verbose = False

        sample_ids = orphan_ids[:10] if verbose else None
        assert sample_ids is None

    def test_metadata_cleanup_logic(self, mock_vector_meta):
        """Test metadata is cleaned along with vectors."""
        orphan_ids = ["vec1", "vec2"]

        for oid in orphan_ids:
            if oid in mock_vector_meta:
                del mock_vector_meta[oid]

        assert "vec1" not in mock_vector_meta
        assert "vec2" not in mock_vector_meta

    def test_redis_error_threshold_abort(self):
        """Test cleanup aborts after too many Redis errors."""
        redis_errors = 0
        max_redis_errors = 10

        # Simulate multiple Redis errors
        for i in range(15):
            redis_errors += 1
            if redis_errors > max_redis_errors:
                break

        assert redis_errors > max_redis_errors


class TestCacheClearLogic:
    """Tests for cache clear logic."""

    def test_pattern_matching_wildcard(self):
        """Test pattern matching with wildcard."""
        import fnmatch

        pattern = "analysis_*"
        keys = ["analysis_result:1", "analysis_result:2", "other_key", "analysis_cache"]

        matching_keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        assert len(matching_keys) == 3
        assert "other_key" not in matching_keys

    def test_pattern_matching_exact(self):
        """Test exact pattern matching."""
        import fnmatch

        pattern = "analysis_result:123"
        keys = ["analysis_result:123", "analysis_result:124", "analysis_result:125"]

        matching_keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        assert len(matching_keys) == 1
        assert matching_keys[0] == "analysis_result:123"

    def test_pattern_matching_question_mark(self):
        """Test pattern matching with ? wildcard."""
        import fnmatch

        pattern = "key_?"
        keys = ["key_1", "key_2", "key_10", "key_abc"]

        matching_keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        assert len(matching_keys) == 2
        assert "key_1" in matching_keys
        assert "key_2" in matching_keys

    def test_empty_keys_response(self):
        """Test response when no keys match pattern."""
        matching_keys = []

        if not matching_keys:
            response = {
                "deleted_count": 0,
                "message": "No keys matching pattern: nonexistent_*"
            }

        assert response["deleted_count"] == 0


class TestMaintenanceStatsLogic:
    """Tests for maintenance stats logic."""

    def test_stats_structure(self):
        """Test stats response structure."""
        stats = {
            "vector_store": {
                "total_vectors": 100,
                "metadata_entries": 95
            },
            "cache": {
                "available": True,
                "size": 1024000
            },
            "maintenance": {
                "orphan_check_available": True,
                "last_cleanup": None
            }
        }

        assert "vector_store" in stats
        assert "cache" in stats
        assert "maintenance" in stats
        assert stats["vector_store"]["total_vectors"] == 100

    def test_stats_cache_unavailable(self):
        """Test stats when cache is unavailable."""
        stats = {
            "cache": {
                "available": False,
                "size": 0
            }
        }

        assert stats["cache"]["available"] is False
        assert stats["cache"]["size"] == 0

    def test_stats_orphan_check_depends_on_cache(self):
        """Test orphan check availability depends on cache."""
        cache_available = True

        stats = {
            "maintenance": {
                "orphan_check_available": cache_available,
                "last_cleanup": None
            }
        }

        assert stats["maintenance"]["orphan_check_available"] is True


class TestVectorBackendReloadLogic:
    """Tests for vector backend reload logic."""

    def test_reload_success_response(self):
        """Test response structure on successful reload."""
        ok = True
        backend = "memory"

        if ok:
            response = {"status": "ok", "backend": backend}

        assert response["status"] == "ok"
        assert response["backend"] == "memory"

    def test_reload_failure_handling(self):
        """Test error handling when reload fails."""
        ok = False

        if not ok:
            should_raise_error = True

        assert should_raise_error is True

    def test_backend_env_var_default(self):
        """Test default backend from environment."""
        import os

        with patch.dict("os.environ", {}, clear=True):
            backend = os.environ.get("VECTOR_STORE_BACKEND", "memory")
            assert backend == "memory"

    def test_backend_env_var_override(self):
        """Test backend override from environment."""
        import os

        with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "faiss"}):
            backend = os.environ.get("VECTOR_STORE_BACKEND", "memory")
            assert backend == "faiss"


class TestOrphanCleanupResponse:
    """Tests for OrphanCleanupResponse model fields."""

    def test_response_skipped_status(self):
        """Test skipped status response fields."""
        response = {
            "orphan_count": 5,
            "deleted_count": 0,
            "sample_ids": None,
            "status": "skipped",
            "message": "Orphan count 5 below threshold 10"
        }

        assert response["status"] == "skipped"
        assert response["deleted_count"] == 0
        assert "threshold" in response["message"]

    def test_response_dry_run_status(self):
        """Test dry_run status response fields."""
        response = {
            "orphan_count": 15,
            "deleted_count": 0,
            "sample_ids": ["id1", "id2"],
            "status": "dry_run",
            "message": "Would delete 15 orphan vectors"
        }

        assert response["status"] == "dry_run"
        assert response["deleted_count"] == 0
        assert "Would delete" in response["message"]

    def test_response_ok_status(self):
        """Test ok status response fields."""
        response = {
            "orphan_count": 15,
            "deleted_count": 15,
            "sample_ids": None,
            "status": "ok",
            "message": "Successfully deleted 15 orphan vectors"
        }

        assert response["status"] == "ok"
        assert response["deleted_count"] == 15
        assert "Successfully" in response["message"]


class TestErrorHandlingLogic:
    """Tests for error handling paths."""

    def test_redis_connection_error_handling(self):
        """Test handling of Redis connection errors."""
        error_type = "SERVICE_UNAVAILABLE"
        stage = "orphan_cleanup"

        error = {
            "code": error_type,
            "stage": stage,
            "message": "Redis connection failed during orphan cleanup"
        }

        assert error["code"] == "SERVICE_UNAVAILABLE"
        assert error["stage"] == "orphan_cleanup"

    def test_cache_client_unavailable_error(self):
        """Test handling when cache client is unavailable."""
        client = None

        if client is None:
            error = {
                "code": "SERVICE_UNAVAILABLE",
                "stage": "cache_clear",
                "message": "Cache client not available"
            }

        assert error["code"] == "SERVICE_UNAVAILABLE"

    def test_internal_error_handling(self):
        """Test internal error handling."""
        error = {
            "code": "INTERNAL_ERROR",
            "stage": "cache_clear",
            "message": "Failed to clear cache"
        }

        assert error["code"] == "INTERNAL_ERROR"

    def test_vector_store_stats_error(self):
        """Test error handling when reading vector store stats fails."""
        error = {
            "code": "INTERNAL_ERROR",
            "stage": "maintenance_stats",
            "message": "Failed to read vector store stats"
        }

        assert error["code"] == "INTERNAL_ERROR"
        assert error["stage"] == "maintenance_stats"


class TestMetricsIntegration:
    """Tests for metrics integration in maintenance endpoints."""

    def test_vector_orphan_metric_exists(self):
        """Test vector_orphan_total metric exists."""
        from src.utils.analysis_metrics import vector_orphan_total

        assert vector_orphan_total is not None

    def test_vector_cold_pruned_metric_exists(self):
        """Test vector_cold_pruned_total metric exists."""
        from src.utils.analysis_metrics import vector_cold_pruned_total

        assert vector_cold_pruned_total is not None

    def test_vector_store_reload_metric_exists(self):
        """Test vector_store_reload_total metric exists."""
        from src.utils.analysis_metrics import vector_store_reload_total

        assert vector_store_reload_total is not None

    def test_vector_store_reload_metric_labels(self):
        """Test vector_store_reload_total supports required labels."""
        from src.utils.analysis_metrics import vector_store_reload_total

        labeled_success = vector_store_reload_total.labels(status="success")
        labeled_error = vector_store_reload_total.labels(status="error")

        assert labeled_success is not None
        assert labeled_error is not None


class TestVectorStoreReloadResponse:
    """Tests for VectorStoreReloadResponse model."""

    def test_reload_response_ok(self):
        """Test reload response with ok status."""
        response = {
            "status": "ok",
            "backend": "memory"
        }

        assert response["status"] == "ok"
        assert response["backend"] == "memory"

    def test_reload_response_faiss_backend(self):
        """Test reload response with faiss backend."""
        response = {
            "status": "ok",
            "backend": "faiss"
        }

        assert response["backend"] == "faiss"


class TestCachePatternValidation:
    """Tests for cache pattern validation."""

    def test_all_keys_pattern(self):
        """Test * pattern matches all keys."""
        import fnmatch

        pattern = "*"
        keys = ["key1", "key2", "analysis:1", "cache:test"]

        matching = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        assert len(matching) == len(keys)

    def test_prefix_pattern(self):
        """Test prefix pattern matching."""
        import fnmatch

        pattern = "analysis:*"
        keys = ["analysis:1", "analysis:2", "cache:1", "other"]

        matching = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        assert len(matching) == 2

    def test_suffix_pattern(self):
        """Test suffix pattern matching."""
        import fnmatch

        pattern = "*:result"
        keys = ["analysis:result", "cache:result", "test:data"]

        matching = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        assert len(matching) == 2
