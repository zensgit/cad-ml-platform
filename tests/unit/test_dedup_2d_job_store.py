"""Unit tests for the Dedup2D async job store with tenant isolation.

Phase 1 features:
- Tenant isolation: jobs are scoped by tenant_id
- Queue capacity protection: JobQueueFullError when max_jobs exceeded
- Multi-worker detection: startup warning for uvicorn --workers > 1
"""

from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any, Dict
from unittest import mock

import pytest

from src.core.dedupcad_2d_jobs import (
    Dedup2DJob,
    Dedup2DJobStatus,
    Dedup2DJobStore,
    JobForbiddenError,
    JobNotFoundError,
    JobQueueFullError,
    _check_multi_worker_warning,
    _MULTI_WORKER_WARNING_ISSUED,
    get_dedup2d_job_store,
    reset_dedup2d_job_store,
)


@pytest.fixture
def job_store():
    """Create a fresh job store for each test."""
    store = Dedup2DJobStore(max_concurrency=2, max_jobs=5, ttl_seconds=60)
    yield store
    store.reset()


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestDedup2DJobStoreTenantIsolation:
    """Test tenant isolation in the job store."""

    @pytest.mark.asyncio
    async def test_submit_creates_job_with_tenant_id(self, job_store):
        """submit() should create a job with the provided tenant_id."""

        async def runner() -> Dict[str, Any]:
            return {"success": True}

        job = await job_store.submit(runner, tenant_id="tenant_A")
        assert job.tenant_id == "tenant_A"
        assert job.status == Dedup2DJobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_for_tenant_allows_same_tenant(self, job_store):
        """get_for_tenant() should allow access by the owning tenant."""

        async def runner() -> Dict[str, Any]:
            return {"success": True}

        job = await job_store.submit(runner, tenant_id="tenant_A")
        result = await job_store.get_for_tenant(job.job_id, "tenant_A")
        assert result.job_id == job.job_id
        assert result.tenant_id == "tenant_A"

    @pytest.mark.asyncio
    async def test_get_for_tenant_denies_different_tenant(self, job_store):
        """get_for_tenant() should raise JobForbiddenError for different tenant."""

        async def runner() -> Dict[str, Any]:
            return {"success": True}

        job = await job_store.submit(runner, tenant_id="tenant_A")

        with pytest.raises(JobForbiddenError) as exc_info:
            await job_store.get_for_tenant(job.job_id, "tenant_B")

        assert exc_info.value.job_id == job.job_id
        assert exc_info.value.tenant_id == "tenant_B"

    @pytest.mark.asyncio
    async def test_get_for_tenant_raises_not_found_for_missing_job(self, job_store):
        """get_for_tenant() should raise JobNotFoundError for non-existent job."""
        with pytest.raises(JobNotFoundError) as exc_info:
            await job_store.get_for_tenant("nonexistent-job-id", "any_tenant")

        assert exc_info.value.job_id == "nonexistent-job-id"

    @pytest.mark.asyncio
    async def test_cancel_for_tenant_allows_same_tenant(self, job_store):
        """cancel_for_tenant() should allow cancellation by the owning tenant."""

        async def slow_runner() -> Dict[str, Any]:
            await asyncio.sleep(1.0)
            return {"success": True}

        job = await job_store.submit(slow_runner, tenant_id="tenant_A")
        result = await job_store.cancel_for_tenant(job.job_id, "tenant_A")
        assert result is True

        # Verify job is canceled
        job_status = await job_store.get_for_tenant(job.job_id, "tenant_A")
        assert job_status.status == Dedup2DJobStatus.CANCELED

    @pytest.mark.asyncio
    async def test_cancel_for_tenant_denies_different_tenant(self, job_store):
        """cancel_for_tenant() should raise JobForbiddenError for different tenant."""

        async def slow_runner() -> Dict[str, Any]:
            await asyncio.sleep(1.0)
            return {"success": True}

        job = await job_store.submit(slow_runner, tenant_id="tenant_A")

        with pytest.raises(JobForbiddenError) as exc_info:
            await job_store.cancel_for_tenant(job.job_id, "tenant_B")

        assert exc_info.value.job_id == job.job_id
        assert exc_info.value.tenant_id == "tenant_B"

    @pytest.mark.asyncio
    async def test_cancel_for_tenant_raises_not_found_for_missing_job(self, job_store):
        """cancel_for_tenant() should raise JobNotFoundError for non-existent job."""
        with pytest.raises(JobNotFoundError) as exc_info:
            await job_store.cancel_for_tenant("nonexistent-job-id", "any_tenant")

        assert exc_info.value.job_id == "nonexistent-job-id"

    @pytest.mark.asyncio
    async def test_list_for_tenant_returns_only_tenant_jobs(self, job_store):
        """list_for_tenant() should only return jobs for the specified tenant."""

        async def runner() -> Dict[str, Any]:
            return {"success": True}

        # Create jobs for different tenants
        await job_store.submit(runner, tenant_id="tenant_A")
        await job_store.submit(runner, tenant_id="tenant_A")
        await job_store.submit(runner, tenant_id="tenant_B")

        # Wait for jobs to complete
        await asyncio.sleep(0.1)

        # List tenant A's jobs
        tenant_a_jobs = await job_store.list_for_tenant("tenant_A")
        assert len(tenant_a_jobs) == 2
        assert all(j.tenant_id == "tenant_A" for j in tenant_a_jobs)

        # List tenant B's jobs
        tenant_b_jobs = await job_store.list_for_tenant("tenant_B")
        assert len(tenant_b_jobs) == 1
        assert all(j.tenant_id == "tenant_B" for j in tenant_b_jobs)

    @pytest.mark.asyncio
    async def test_list_for_tenant_with_status_filter(self, job_store):
        """list_for_tenant() should filter by status when provided."""

        async def runner() -> Dict[str, Any]:
            return {"success": True}

        async def slow_runner() -> Dict[str, Any]:
            await asyncio.sleep(5.0)
            return {"success": True}

        # Create a fast job that completes
        await job_store.submit(runner, tenant_id="tenant_A")
        await asyncio.sleep(0.1)  # Let it complete

        # Create a slow job that stays pending/in_progress
        await job_store.submit(slow_runner, tenant_id="tenant_A")

        completed_jobs = await job_store.list_for_tenant(
            "tenant_A", status=Dedup2DJobStatus.COMPLETED
        )
        assert len(completed_jobs) >= 1
        assert all(j.status == Dedup2DJobStatus.COMPLETED for j in completed_jobs)


class TestDedup2DJobStoreQueueCapacity:
    """Test queue capacity protection."""

    @pytest.mark.asyncio
    async def test_queue_full_raises_error(self):
        """Should raise JobQueueFullError when queue is at capacity."""
        # Create a store with a very small max_jobs limit
        store = Dedup2DJobStore(max_concurrency=1, max_jobs=2, ttl_seconds=60)

        try:
            async def slow_runner() -> Dict[str, Any]:
                await asyncio.sleep(10.0)
                return {"success": True}

            # Submit jobs up to capacity
            await store.submit(slow_runner, tenant_id="tenant_A")
            await store.submit(slow_runner, tenant_id="tenant_A")

            # Third job should raise JobQueueFullError
            with pytest.raises(JobQueueFullError) as exc_info:
                await store.submit(slow_runner, tenant_id="tenant_A")

            assert exc_info.value.max_jobs == 2
            assert exc_info.value.current_jobs == 2
        finally:
            store.reset()

    @pytest.mark.asyncio
    async def test_get_queue_depth(self, job_store):
        """get_queue_depth() should return the current number of pending/running jobs."""

        async def slow_runner() -> Dict[str, Any]:
            await asyncio.sleep(1.0)
            return {"success": True}

        assert job_store.get_queue_depth() == 0

        await job_store.submit(slow_runner, tenant_id="tenant_A")
        assert job_store.get_queue_depth() >= 1

        await job_store.submit(slow_runner, tenant_id="tenant_A")
        assert job_store.get_queue_depth() >= 2


class TestDedup2DJobStoreMultiWorkerWarning:
    """Test multi-worker warning detection."""

    def test_multi_worker_warning_issued_for_web_concurrency(self):
        """Warning should be issued when WEB_CONCURRENCY > 1."""
        import src.core.dedupcad_2d_jobs as jobs_module

        # Reset the warning flag
        jobs_module._MULTI_WORKER_WARNING_ISSUED = False

        with mock.patch.dict(os.environ, {"WEB_CONCURRENCY": "2", "UVICORN_WORKERS": "1"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_multi_worker_warning()

                # Check that a warning was issued
                assert len(w) >= 1
                assert any("Multi-worker mode detected" in str(warning.message) for warning in w)

        # Reset the flag for other tests
        jobs_module._MULTI_WORKER_WARNING_ISSUED = False

    def test_multi_worker_warning_issued_for_uvicorn_workers(self):
        """Warning should be issued when UVICORN_WORKERS > 1."""
        import src.core.dedupcad_2d_jobs as jobs_module

        # Reset the warning flag
        jobs_module._MULTI_WORKER_WARNING_ISSUED = False

        with mock.patch.dict(os.environ, {"WEB_CONCURRENCY": "1", "UVICORN_WORKERS": "4"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_multi_worker_warning()

                # Check that a warning was issued
                assert len(w) >= 1
                assert any("Multi-worker mode detected" in str(warning.message) for warning in w)

        # Reset the flag for other tests
        jobs_module._MULTI_WORKER_WARNING_ISSUED = False

    def test_no_warning_for_single_worker(self):
        """No warning should be issued for single worker mode."""
        import src.core.dedupcad_2d_jobs as jobs_module

        # Reset the warning flag
        jobs_module._MULTI_WORKER_WARNING_ISSUED = False

        with mock.patch.dict(os.environ, {"WEB_CONCURRENCY": "1", "UVICORN_WORKERS": "1"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_multi_worker_warning()

                # Check that no multi-worker warning was issued
                multi_worker_warnings = [
                    warning for warning in w if "Multi-worker mode detected" in str(warning.message)
                ]
                assert len(multi_worker_warnings) == 0

        # Reset the flag for other tests
        jobs_module._MULTI_WORKER_WARNING_ISSUED = False


class TestDedup2DJobStoreExceptions:
    """Test exception classes."""

    def test_job_queue_full_error_message(self):
        """JobQueueFullError should have descriptive message."""
        error = JobQueueFullError(max_jobs=100, current_jobs=100)
        assert str(error) == "Job queue full: 100/100 jobs"
        assert error.max_jobs == 100
        assert error.current_jobs == 100

    def test_job_not_found_error_message(self):
        """JobNotFoundError should have descriptive message."""
        error = JobNotFoundError(job_id="test-job-id")
        assert str(error) == "Job not found: test-job-id"
        assert error.job_id == "test-job-id"

    def test_job_forbidden_error_message(self):
        """JobForbiddenError should have descriptive message."""
        error = JobForbiddenError(job_id="test-job-id", tenant_id="tenant_B")
        assert str(error) == "Access denied to job test-job-id for tenant tenant_B"
        assert error.job_id == "test-job-id"
        assert error.tenant_id == "tenant_B"


class TestDedup2DJobDataclass:
    """Test the Dedup2DJob dataclass."""

    def test_job_is_finished_for_terminal_states(self):
        """is_finished() should return True for COMPLETED, FAILED, CANCELED."""
        for status in [
            Dedup2DJobStatus.COMPLETED,
            Dedup2DJobStatus.FAILED,
            Dedup2DJobStatus.CANCELED,
        ]:
            job = Dedup2DJob(
                job_id="test",
                tenant_id="tenant",
                status=status,
            )
            assert job.is_finished() is True

    def test_job_is_not_finished_for_active_states(self):
        """is_finished() should return False for PENDING, IN_PROGRESS."""
        for status in [Dedup2DJobStatus.PENDING, Dedup2DJobStatus.IN_PROGRESS]:
            job = Dedup2DJob(
                job_id="test",
                tenant_id="tenant",
                status=status,
            )
            assert job.is_finished() is False
