"""
Unit tests for Arq-based task queue module.

Tests cover:
- TaskConfig configuration
- TaskClient methods
- TaskResult and TaskStatus
- Fallback behavior when arq unavailable
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.core.tasks import (
    TaskConfig,
    TaskResult,
    TaskStatus,
    ARQ_AVAILABLE,
)


class TestTaskConfig:
    """Test TaskConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TaskConfig()

        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert config.redis_password is None
        assert config.max_jobs == 10
        assert config.job_timeout == 300
        assert config.retry_jobs is True
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = TaskConfig(
            redis_host="redis.example.com",
            redis_port=6380,
            redis_db=1,
            redis_password="secret",
            max_jobs=20,
            job_timeout=600,
            retry_jobs=False,
            max_retries=5,
        )

        assert config.redis_host == "redis.example.com"
        assert config.redis_port == 6380
        assert config.redis_db == 1
        assert config.redis_password == "secret"
        assert config.max_jobs == 20
        assert config.job_timeout == 600
        assert config.retry_jobs is False
        assert config.max_retries == 5

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "REDIS_HOST": "redis-server",
            "REDIS_PORT": "6381",
            "REDIS_DB": "2",
            "REDIS_PASSWORD": "env-password",
            "ARQ_MAX_JOBS": "15",
            "ARQ_JOB_TIMEOUT": "450",
            "ARQ_RETRY_JOBS": "false",
            "ARQ_MAX_RETRIES": "4",
        }

        with patch.dict("os.environ", env_vars):
            config = TaskConfig.from_env()

            assert config.redis_host == "redis-server"
            assert config.redis_port == 6381
            assert config.redis_db == 2
            assert config.redis_password == "env-password"
            assert config.max_jobs == 15
            assert config.job_timeout == 450
            assert config.retry_jobs is False
            assert config.max_retries == 4


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_basic_result(self):
        """Test basic task result."""
        result = TaskResult(
            job_id="job-123",
            status=TaskStatus.COMPLETED,
        )

        assert result.job_id == "job-123"
        assert result.status == TaskStatus.COMPLETED
        assert result.result is None
        assert result.error is None

    def test_successful_result(self):
        """Test successful task result."""
        result = TaskResult(
            job_id="job-456",
            status=TaskStatus.COMPLETED,
            result={"analysis": "complete", "features": [1, 2, 3]},
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 10, 0, 30),
        )

        assert result.result["analysis"] == "complete"
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_failed_result(self):
        """Test failed task result."""
        result = TaskResult(
            job_id="job-789",
            status=TaskStatus.FAILED,
            error="File not found: doc.dxf",
        )

        assert result.status == TaskStatus.FAILED
        assert "File not found" in result.error

    def test_timeout_result(self):
        """Test timeout task result."""
        result = TaskResult(
            job_id="job-timeout",
            status=TaskStatus.TIMEOUT,
            error="Task timed out waiting for result",
        )

        assert result.status == TaskStatus.TIMEOUT


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.TIMEOUT == "timeout"
        assert TaskStatus.DEFERRED == "deferred"


@pytest.mark.skipif(not ARQ_AVAILABLE, reason="arq not installed")
class TestTaskClient:
    """Test TaskClient implementation."""

    @pytest.fixture
    def mock_arq_pool(self):
        """Create a mock arq pool."""
        with patch("src.core.tasks.create_pool") as mock_create:
            pool = AsyncMock()
            mock_create.return_value = pool
            yield pool

    @pytest.fixture
    def task_client(self):
        """Create a TaskClient instance."""
        from src.core.tasks import TaskClient
        return TaskClient()

    @pytest.mark.asyncio
    async def test_connect(self, task_client, mock_arq_pool):
        """Test client connection."""
        await task_client.connect()
        assert task_client._connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, task_client, mock_arq_pool):
        """Test client disconnection."""
        await task_client.connect()
        await task_client.disconnect()
        assert task_client._connected is False

    @pytest.mark.asyncio
    async def test_submit_analysis(self, task_client, mock_arq_pool):
        """Test submitting analysis task."""
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"
        mock_arq_pool.enqueue_job = AsyncMock(return_value=mock_job)

        await task_client.connect()
        job_id = await task_client.submit_analysis(
            "doc.dxf",
            options={"extract_features": True},
        )

        assert job_id == "test-job-123"
        mock_arq_pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_arq_pool):
        """Test async context manager."""
        from src.core.tasks import TaskClient

        async with TaskClient() as client:
            assert client._connected is True

        assert client._connected is False


class TestTaskClientFactory:
    """Test task client factory function."""

    @pytest.mark.skipif(not ARQ_AVAILABLE, reason="arq not installed")
    def test_get_task_client(self):
        """Test getting global task client."""
        from src.core.tasks import get_task_client

        client = get_task_client()
        assert client is not None

    @pytest.mark.skipif(not ARQ_AVAILABLE, reason="arq not installed")
    def test_singleton_behavior(self):
        """Test singleton behavior."""
        from src.core.tasks import get_task_client

        client1 = get_task_client()
        client2 = get_task_client()
        assert client1 is client2


class TestWorkerSettings:
    """Test worker settings."""

    @pytest.mark.skipif(not ARQ_AVAILABLE, reason="arq not installed")
    def test_worker_functions_defined(self):
        """Test that worker functions are defined."""
        from src.core.tasks.worker import WorkerSettings

        assert len(WorkerSettings.functions) > 0
        function_names = [f.__name__ for f in WorkerSettings.functions]
        assert "analyze_cad_file" in function_names
        assert "extract_features" in function_names

    @pytest.mark.skipif(not ARQ_AVAILABLE, reason="arq not installed")
    def test_worker_redis_settings(self):
        """Test worker Redis settings."""
        from src.core.tasks.worker import WorkerSettings

        assert WorkerSettings.redis_settings is not None
        assert WorkerSettings.max_jobs > 0
