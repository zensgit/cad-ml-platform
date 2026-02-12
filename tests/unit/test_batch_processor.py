"""Tests for batch processing."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBatchJobStatus:
    """Tests for BatchJobStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        from src.core.batch.processor import BatchJobStatus

        assert BatchJobStatus.PENDING.value == "pending"
        assert BatchJobStatus.RUNNING.value == "running"
        assert BatchJobStatus.COMPLETED.value == "completed"
        assert BatchJobStatus.FAILED.value == "failed"
        assert BatchJobStatus.CANCELLED.value == "cancelled"
        assert BatchJobStatus.PARTIAL.value == "partial"


class TestBatchItemStatus:
    """Tests for BatchItemStatus enum."""

    def test_item_status_values(self):
        """Test item status enum values."""
        from src.core.batch.processor import BatchItemStatus

        assert BatchItemStatus.PENDING.value == "pending"
        assert BatchItemStatus.RUNNING.value == "running"
        assert BatchItemStatus.COMPLETED.value == "completed"
        assert BatchItemStatus.FAILED.value == "failed"
        assert BatchItemStatus.SKIPPED.value == "skipped"


class TestBatchItem:
    """Tests for BatchItem dataclass."""

    def test_default_values(self):
        """Test default item values."""
        from src.core.batch.processor import BatchItem, BatchItemStatus

        item = BatchItem(id="item-1", data={"key": "value"})

        assert item.id == "item-1"
        assert item.data == {"key": "value"}
        assert item.status == BatchItemStatus.PENDING
        assert item.result is None
        assert item.error is None
        assert item.retries == 0


class TestBatchJob:
    """Tests for BatchJob dataclass."""

    def test_default_values(self):
        """Test default job values."""
        from src.core.batch.processor import BatchItem, BatchJob, BatchJobStatus

        items = [BatchItem(id="1", data={}), BatchItem(id="2", data={})]
        job = BatchJob(id="job-1", operation="test", items=items)

        assert job.id == "job-1"
        assert job.operation == "test"
        assert job.status == BatchJobStatus.PENDING
        assert job.priority == 5
        assert job.total_items == 2

    def test_progress_calculation(self):
        """Test progress calculation."""
        from src.core.batch.processor import BatchItem, BatchItemStatus, BatchJob

        items = [
            BatchItem(id="1", data={}, status=BatchItemStatus.COMPLETED),
            BatchItem(id="2", data={}, status=BatchItemStatus.PENDING),
            BatchItem(id="3", data={}, status=BatchItemStatus.FAILED),
            BatchItem(id="4", data={}, status=BatchItemStatus.PENDING),
        ]
        job = BatchJob(id="job-1", operation="test", items=items)

        assert job.completed_items == 2
        assert job.successful_items == 1
        assert job.failed_items == 1
        assert job.progress == 50.0

    def test_to_summary(self):
        """Test summary generation."""
        from src.core.batch.processor import BatchItem, BatchJob

        items = [BatchItem(id="1", data={})]
        job = BatchJob(
            id="job-1",
            operation="test",
            items=items,
            user_id="user-1",
            tenant_id="tenant-1",
        )

        summary = job.to_summary()

        assert summary["job_id"] == "job-1"
        assert summary["operation"] == "test"
        assert summary["status"] == "pending"
        assert summary["total_items"] == 1
        assert summary["user_id"] == "user-1"
        assert summary["tenant_id"] == "tenant-1"


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.mark.asyncio
    async def test_register_handler(self):
        """Test handler registration."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {"result": "ok"}

        processor.register_handler("test", test_handler)

        assert "test" in processor._handlers

    @pytest.mark.asyncio
    async def test_submit_job_creates_job(self):
        """Test job submission."""
        from src.core.batch.processor import BatchJobStatus, BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {"result": "ok"}

        processor.register_handler("test", test_handler)

        job = await processor.submit_job(
            operation="test",
            items=[{"id": 1}, {"id": 2}],
            user_id="user-1",
        )

        assert job.id is not None
        assert job.operation == "test"
        assert job.total_items == 2
        assert job.user_id == "user-1"
        assert job.status == BatchJobStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_job_unknown_operation(self):
        """Test submission fails for unknown operation."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()

        with pytest.raises(ValueError, match="Unknown operation"):
            await processor.submit_job(
                operation="unknown",
                items=[{"id": 1}],
            )

    @pytest.mark.asyncio
    async def test_submit_job_too_many_items(self):
        """Test submission fails for too many items."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor(max_items_per_job=5)

        async def test_handler(data):
            return {}

        processor.register_handler("test", test_handler)

        with pytest.raises(ValueError, match="Too many items"):
            await processor.submit_job(
                operation="test",
                items=[{"id": i} for i in range(10)],
            )

    @pytest.mark.asyncio
    async def test_get_job(self):
        """Test getting a job by ID."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {}

        processor.register_handler("test", test_handler)

        job = await processor.submit_job("test", [{"id": 1}])
        retrieved = await processor.get_job(job.id)

        assert retrieved is not None
        assert retrieved.id == job.id

    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        """Test getting a non-existent job."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()
        result = await processor.get_job("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_job_status(self):
        """Test getting job status."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {}

        processor.register_handler("test", test_handler)

        job = await processor.submit_job("test", [{"id": 1}])
        status = await processor.get_job_status(job.id)

        assert status is not None
        assert status["job_id"] == job.id
        assert status["status"] == "pending"

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test cancelling a job."""
        from src.core.batch.processor import BatchJobStatus, BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {}

        processor.register_handler("test", test_handler)

        job = await processor.submit_job("test", [{"id": 1}])
        result = await processor.cancel_job(job.id)

        assert result is True
        assert job.status == BatchJobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self):
        """Test cancelling a non-existent job."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()
        result = await processor.cancel_job("non-existent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        """Test listing jobs."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {}

        processor.register_handler("test", test_handler)

        await processor.submit_job("test", [{"id": 1}], user_id="user-1")
        await processor.submit_job("test", [{"id": 2}], user_id="user-2")

        all_jobs = await processor.list_jobs()
        user1_jobs = await processor.list_jobs(user_id="user-1")

        assert len(all_jobs) == 2
        assert len(user1_jobs) == 1
        assert user1_jobs[0]["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting metrics."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {}

        processor.register_handler("test", test_handler)

        await processor.submit_job("test", [{"id": 1}])

        metrics = processor.get_metrics()

        assert metrics["jobs_submitted"] == 1
        assert metrics["total_jobs"] == 1


class TestBatchProcessorProcessing:
    """Tests for batch processor job processing."""

    @pytest.mark.asyncio
    async def test_process_job_success(self):
        """Test successful job processing."""
        from src.core.batch.processor import BatchJobStatus, BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            return {"processed": data["id"]}

        processor.register_handler("test", test_handler)
        await processor.start()

        try:
            job = await processor.submit_job("test", [{"id": 1}, {"id": 2}])

            # Wait for processing
            for _ in range(50):
                await asyncio.sleep(0.1)
                if job.status in (BatchJobStatus.COMPLETED, BatchJobStatus.FAILED):
                    break

            assert job.status == BatchJobStatus.COMPLETED
            assert job.successful_items == 2
            assert job.failed_items == 0

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_process_job_with_failures(self):
        """Test job processing with some failures."""
        from src.core.batch.processor import BatchJobStatus, BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            if data.get("fail"):
                raise ValueError("Intentional failure")
            return {"processed": True}

        processor.register_handler("test", test_handler)
        await processor.start()

        try:
            job = await processor.submit_job(
                "test",
                [{"id": 1}, {"id": 2, "fail": True}],
                max_retries=0,
            )

            # Wait for processing
            for _ in range(50):
                await asyncio.sleep(0.1)
                if job.status in (
                    BatchJobStatus.COMPLETED,
                    BatchJobStatus.FAILED,
                    BatchJobStatus.PARTIAL,
                ):
                    break

            assert job.status == BatchJobStatus.PARTIAL
            assert job.successful_items == 1
            assert job.failed_items == 1

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_process_job_all_failures(self):
        """Test job processing with all failures."""
        from src.core.batch.processor import BatchJobStatus, BatchProcessor

        processor = BatchProcessor()

        async def test_handler(data):
            raise ValueError("Always fail")

        processor.register_handler("test", test_handler)
        await processor.start()

        try:
            job = await processor.submit_job(
                "test",
                [{"id": 1}, {"id": 2}],
                max_retries=0,
            )

            # Wait for processing
            for _ in range(50):
                await asyncio.sleep(0.1)
                if job.status in (BatchJobStatus.COMPLETED, BatchJobStatus.FAILED):
                    break

            assert job.status == BatchJobStatus.FAILED
            assert job.successful_items == 0
            assert job.failed_items == 2

        finally:
            await processor.stop()


class TestBatchProcessorLifecycle:
    """Tests for batch processor start/stop."""

    @pytest.mark.asyncio
    async def test_start_creates_worker(self):
        """Test starting creates worker task."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()
        await processor.start()

        assert processor._running is True
        assert processor._worker_task is not None

        await processor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_worker(self):
        """Test stopping cancels worker."""
        from src.core.batch.processor import BatchProcessor

        processor = BatchProcessor()
        await processor.start()
        await processor.stop()

        assert processor._running is False


class TestGetBatchProcessor:
    """Tests for get_batch_processor function."""

    @pytest.mark.asyncio
    async def test_returns_singleton(self):
        """Test returns singleton instance."""
        from src.core.batch import processor as batch_module

        # Reset global
        batch_module._processor = None

        proc1 = batch_module.get_batch_processor()
        proc2 = batch_module.get_batch_processor()

        assert proc1 is proc2

        # Cleanup
        batch_module._processor = None
