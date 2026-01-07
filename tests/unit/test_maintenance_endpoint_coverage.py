"""Tests for src/api/v1/maintenance.py endpoint functions to improve coverage.

Covers:
- cleanup_orphan_vectors endpoint
- clear_cache endpoint
- get_maintenance_stats endpoint
- reload_vector_backend endpoint
- Error handling paths
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _assert_error_context(exc: HTTPException, operation: str, resource_id: str) -> None:
    detail = exc.detail
    assert isinstance(detail, dict)
    context = detail.get("context")
    assert isinstance(context, dict)
    assert context.get("operation") == operation
    assert context.get("resource_id") == resource_id
    assert "suggestion" in context


class TestCleanupOrphanVectorsEndpoint:
    """Tests for cleanup_orphan_vectors endpoint."""

    @pytest.mark.asyncio
    async def test_cleanup_orphans_redis_connection_failed(self):
        """Test cleanup handles Redis connection failure."""
        from src.api.v1.maintenance import cleanup_orphan_vectors

        with patch("src.utils.cache.get_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Connection refused")

            with pytest.raises(HTTPException) as exc_info:
                await cleanup_orphan_vectors(api_key="test")

            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_cleanup_orphans_no_cache_client(self):
        """Test cleanup when no cache client available (all vectors are orphans)."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        # Use real lock object
        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1, 2, 3], "v2": [4, 5, 6]}
        test_vector_meta: Dict[str, Any] = {}

        with patch("src.utils.cache.get_client", return_value=None):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                        result = await cleanup_orphan_vectors(
                            threshold=0, force=True, dry_run=False, verbose=True, api_key="test"
                        )

                        assert result.status == "ok"
                        assert result.orphan_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_orphans_skipped_below_threshold(self):
        """Test cleanup skipped when orphan count below threshold."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)  # All vectors are orphans

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1, 2, 3]}

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    result = await cleanup_orphan_vectors(
                        threshold=10,  # Higher than orphan count
                        force=False,
                        dry_run=False,
                        verbose=True,
                        api_key="test",
                    )

                    assert result.status == "skipped"
                    assert "below threshold" in result.message

    @pytest.mark.asyncio
    async def test_cleanup_orphans_dry_run(self):
        """Test cleanup in dry run mode."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)  # All vectors are orphans

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1, 2, 3], "v2": [4, 5, 6]}

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    result = await cleanup_orphan_vectors(
                        threshold=0, force=True, dry_run=True, verbose=True, api_key="test"
                    )

                    assert result.status == "dry_run"
                    assert result.deleted_count == 0
                    assert "Would delete" in result.message
                    assert result.sample_ids is not None

    @pytest.mark.asyncio
    async def test_cleanup_orphans_successful_deletion(self):
        """Test successful orphan vector deletion."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)  # All vectors are orphans

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1, 2, 3], "v2": [4, 5, 6]}
        test_vector_meta: Dict[str, Any] = {"v1": {"created": 123}, "v2": {"created": 456}}

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                        result = await cleanup_orphan_vectors(
                            threshold=0, force=True, dry_run=False, verbose=False, api_key="test"
                        )

                        assert result.status == "ok"
                        assert result.deleted_count == 2
                        assert result.sample_ids is None  # verbose=False

    @pytest.mark.asyncio
    async def test_cleanup_orphans_cache_has_entries(self):
        """Test cleanup when some vectors have cache entries (not orphans)."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        mock_client = AsyncMock()

        # v1 has cache entry, v2 doesn't (orphan)
        async def mock_get(key):
            if "v1" in key:
                return b"cached_data"
            return None

        mock_client.get = mock_get

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1, 2, 3], "v2": [4, 5, 6]}
        test_vector_meta: Dict[str, Any] = {}

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                        result = await cleanup_orphan_vectors(
                            threshold=0, force=True, dry_run=False, verbose=True, api_key="test"
                        )

                        # Only v2 should be orphan
                        assert result.orphan_count == 1
                        assert result.deleted_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_orphans_redis_timeout_during_check(self):
        """Test cleanup handles Redis timeout during cache check."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=TimeoutError("Connection timeout"))

        # Create a large store to trigger the error threshold
        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {f"v{i}": [i] for i in range(20)}

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with pytest.raises(HTTPException) as exc_info:
                        await cleanup_orphan_vectors(
                            threshold=0, force=True, dry_run=False, verbose=False, api_key="test"
                        )

                    assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_cleanup_orphans_unexpected_error_during_check(self):
        """Test cleanup handles unexpected error during cache check."""
        from src.api.v1.maintenance import cleanup_orphan_vectors
        from src.core import similarity

        mock_client = AsyncMock()
        call_count = [0]

        async def mock_get(key):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Unexpected error")
            return None

        mock_client.get = mock_get

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1, 2, 3], "v2": [4, 5, 6]}
        test_vector_meta: Dict[str, Any] = {}

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                        result = await cleanup_orphan_vectors(
                            threshold=0, force=True, dry_run=False, verbose=False, api_key="test"
                        )

                        # Should continue and handle 1 orphan (v2)
                        assert result.orphan_count == 1


class TestClearCacheEndpoint:
    """Tests for clear_cache endpoint."""

    @pytest.mark.asyncio
    async def test_clear_cache_no_client(self):
        """Test clear_cache when no cache client available."""
        from src.api.v1.maintenance import clear_cache

        with patch("src.utils.cache.get_client", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await clear_cache(pattern="*", api_key="test")

            assert exc_info.value.status_code == 503
            _assert_error_context(exc_info.value, "clear_cache", "cache")

    @pytest.mark.asyncio
    async def test_clear_cache_no_matching_keys(self):
        """Test clear_cache when no keys match pattern."""
        from src.api.v1.maintenance import clear_cache

        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(return_value=[])

        with patch("src.utils.cache.get_client", return_value=mock_client):
            result = await clear_cache(pattern="nonexistent:*", api_key="test")

            assert result["deleted_count"] == 0
            assert "No keys matching" in result["message"]

    @pytest.mark.asyncio
    async def test_clear_cache_successful(self):
        """Test successful cache clearing."""
        from src.api.v1.maintenance import clear_cache

        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(return_value=["key1", "key2", "key3"])
        mock_client.delete = AsyncMock(return_value=3)

        with patch("src.utils.cache.get_client", return_value=mock_client):
            result = await clear_cache(pattern="analysis_result:*", api_key="test")

            assert result["deleted_count"] == 3
            assert "Successfully deleted" in result["message"]

    @pytest.mark.asyncio
    async def test_clear_cache_exception(self):
        """Test clear_cache handles exceptions."""
        from src.api.v1.maintenance import clear_cache

        mock_client = AsyncMock()
        mock_client.keys = AsyncMock(side_effect=Exception("Redis error"))

        with patch("src.utils.cache.get_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await clear_cache(pattern="*", api_key="test")

            assert exc_info.value.status_code == 500


class TestGetMaintenanceStatsEndpoint:
    """Tests for get_maintenance_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_basic(self):
        """Test get_maintenance_stats returns basic stats."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {"v1": [1], "v2": [2]}
        test_vector_meta: Dict[str, Any] = {"v1": {"meta": 1}}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch("src.utils.cache.get_client", return_value=None):
                        result = await get_maintenance_stats(api_key="test")

                        assert result["vector_store"]["total_vectors"] == 2
                        assert result["vector_store"]["metadata_entries"] == 1
                        assert result["cache"]["available"] is False

    @pytest.mark.asyncio
    async def test_get_stats_with_cache(self):
        """Test get_maintenance_stats with cache available."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.info = AsyncMock(return_value={"used_memory": 1024000})

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {}
        test_vector_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch("src.utils.cache.get_client", return_value=mock_client):
                        result = await get_maintenance_stats(api_key="test")

                        assert result["cache"]["available"] is True
                        assert result["cache"]["size"] == 1024000
                        assert result["maintenance"]["orphan_check_available"] is True

    @pytest.mark.asyncio
    async def test_get_stats_cache_connection_error(self):
        """Test get_maintenance_stats handles cache connection error."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.info = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {}
        test_vector_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch("src.utils.cache.get_client", return_value=mock_client):
                        result = await get_maintenance_stats(api_key="test")

                        # Should not fail, just mark cache unavailable
                        assert result["cache"]["available"] is False

    @pytest.mark.asyncio
    async def test_get_stats_cache_timeout(self):
        """Test get_maintenance_stats handles cache timeout."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.info = AsyncMock(side_effect=TimeoutError("Timeout"))

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {}
        test_vector_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch("src.utils.cache.get_client", return_value=mock_client):
                        result = await get_maintenance_stats(api_key="test")

                        assert result["cache"]["available"] is False

    @pytest.mark.asyncio
    async def test_get_stats_cache_generic_exception(self):
        """Test get_maintenance_stats handles generic cache exception."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        mock_client = AsyncMock()
        mock_client.info = AsyncMock(side_effect=Exception("Unknown error"))

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {}
        test_vector_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch("src.utils.cache.get_client", return_value=mock_client):
                        result = await get_maintenance_stats(api_key="test")

                        # Should handle gracefully
                        assert "vector_store" in result

    @pytest.mark.asyncio
    async def test_get_stats_get_client_fails(self):
        """Test get_maintenance_stats handles get_client failure."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        test_lock = threading.RLock()
        test_vector_store: Dict[str, Any] = {}
        test_vector_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", test_lock):
                    with patch("src.utils.cache.get_client", side_effect=Exception("Init failed")):
                        result = await get_maintenance_stats(api_key="test")

                        # Should handle gracefully
                        assert result["cache"]["available"] is False

    @pytest.mark.asyncio
    async def test_get_stats_vector_store_exception(self):
        """Test get_maintenance_stats handles vector store read exception."""
        from src.api.v1.maintenance import get_maintenance_stats
        from src.core import similarity

        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(side_effect=Exception("Lock error"))
        mock_lock.__exit__ = MagicMock()

        test_vector_store: Dict[str, Any] = {}
        test_vector_meta: Dict[str, Any] = {}

        with patch.object(similarity, "_VECTOR_STORE", test_vector_store):
            with patch.object(similarity, "_VECTOR_META", test_vector_meta):
                with patch.object(similarity, "_VECTOR_LOCK", mock_lock):
                    with pytest.raises(HTTPException) as exc_info:
                        await get_maintenance_stats(api_key="test")

                    assert exc_info.value.status_code == 500
                    _assert_error_context(exc_info.value, "get_maintenance_stats", "vector_store")


class TestReloadVectorBackendEndpoint:
    """Tests for reload_vector_backend endpoint."""

    @pytest.mark.asyncio
    async def test_reload_backend_successful(self):
        """Test successful backend reload."""
        from src.api.v1.maintenance import reload_vector_backend

        with patch("src.core.similarity.reload_vector_store_backend", return_value=True):
            with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "memory"}):
                result = await reload_vector_backend(api_key="test")

                assert result.status == "ok"
                assert result.backend == "memory"

    @pytest.mark.asyncio
    async def test_reload_backend_failed(self):
        """Test backend reload failure (returns False)."""
        from src.api.v1.maintenance import reload_vector_backend

        with patch("src.core.similarity.reload_vector_store_backend", return_value=False):
            with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "faiss"}):
                with pytest.raises(HTTPException) as exc_info:
                    await reload_vector_backend(api_key="test")

                assert exc_info.value.status_code == 500
                _assert_error_context(exc_info.value, "reload_vector_backend", "vector_store")

    @pytest.mark.asyncio
    async def test_reload_backend_exception(self):
        """Test backend reload throws exception."""
        from src.api.v1.maintenance import reload_vector_backend

        with patch(
            "src.core.similarity.reload_vector_store_backend", side_effect=Exception("Reload error")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reload_vector_backend(api_key="test")

            assert exc_info.value.status_code == 500
            _assert_error_context(exc_info.value, "reload_vector_backend", "vector_store")


class TestOrphanCleanupResponseModel:
    """Tests for OrphanCleanupResponse model."""

    def test_model_creation(self):
        """Test OrphanCleanupResponse model creation."""
        from src.api.v1.maintenance import OrphanCleanupResponse

        response = OrphanCleanupResponse(
            orphan_count=10,
            deleted_count=8,
            sample_ids=["v1", "v2"],
            status="ok",
            message="Cleaned successfully",
        )

        assert response.orphan_count == 10
        assert response.deleted_count == 8
        assert response.sample_ids == ["v1", "v2"]
        assert response.status == "ok"

    def test_model_with_none_sample_ids(self):
        """Test OrphanCleanupResponse with None sample_ids."""
        from src.api.v1.maintenance import OrphanCleanupResponse

        response = OrphanCleanupResponse(
            orphan_count=5, deleted_count=5, sample_ids=None, status="ok", message="Done"
        )

        assert response.sample_ids is None


class TestCleanupAnalysisResultStoreEndpoint:
    """Tests for cleanup_analysis_result_store endpoint."""

    @pytest.mark.asyncio
    async def test_cleanup_analysis_results_disabled(self, monkeypatch: pytest.MonkeyPatch):
        from src.api.v1.maintenance import cleanup_analysis_result_store

        monkeypatch.delenv("ANALYSIS_RESULT_STORE_DIR", raising=False)
        result = await cleanup_analysis_result_store(api_key="test")
        assert result.status == "disabled"

    @pytest.mark.asyncio
    async def test_cleanup_analysis_results_threshold_skip(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        from src.api.v1.maintenance import cleanup_analysis_result_store
        from src.utils.analysis_result_store import store_analysis_result

        monkeypatch.setenv("ANALYSIS_RESULT_STORE_DIR", str(tmp_path))
        await store_analysis_result("old", {"status": "old"})
        old_path = tmp_path / "old.json"
        now = time.time()
        os.utime(old_path, (now - 3600, now - 3600))

        result = await cleanup_analysis_result_store(
            max_age_seconds=60,
            threshold=2,
            dry_run=False,
            verbose=False,
            api_key="test",
        )

        assert result.status == "skipped"
        assert result.deleted_count == 0
        assert old_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_analysis_results_error_context(self):
        from src.api.v1.maintenance import cleanup_analysis_result_store

        with patch(
            "src.api.v1.maintenance.cleanup_analysis_results",
            new=AsyncMock(side_effect=Exception("cleanup failed")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await cleanup_analysis_result_store(api_key="test")

            assert exc_info.value.status_code == 500
            _assert_error_context(
                exc_info.value, "cleanup_analysis_result_store", "analysis_result_store"
            )


class TestVectorStoreReloadResponseModel:
    """Tests for VectorStoreReloadResponse model."""

    def test_model_creation(self):
        """Test VectorStoreReloadResponse model creation."""
        from src.api.v1.maintenance import VectorStoreReloadResponse

        response = VectorStoreReloadResponse(status="ok", backend="memory")

        assert response.status == "ok"
        assert response.backend == "memory"

    def test_model_with_none_backend(self):
        """Test VectorStoreReloadResponse with None backend."""
        from src.api.v1.maintenance import VectorStoreReloadResponse

        response = VectorStoreReloadResponse(status="error", backend=None)

        assert response.backend is None


class TestRouterExport:
    """Tests for router export."""

    def test_router_exported(self):
        """Test router is exported from module."""
        from src.api.v1.maintenance import router

        assert router is not None
