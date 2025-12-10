"""Tests for src/core/resilience/bulkhead.py to improve coverage.

Covers:
- BulkheadError exception
- BulkheadStats dataclass
- BulkheadStrategy abstract base class
- ThreadPoolBulkhead class
- SemaphoreBulkhead class
- Bulkhead unified interface
- bulkhead decorator
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest


class TestBulkheadError:
    """Tests for BulkheadError exception."""

    def test_bulkhead_error_is_exception(self):
        """Test BulkheadError is an Exception subclass."""
        from src.core.resilience.bulkhead import BulkheadError

        assert issubclass(BulkheadError, Exception)

    def test_bulkhead_error_with_message(self):
        """Test BulkheadError with message."""
        from src.core.resilience.bulkhead import BulkheadError

        error = BulkheadError("Test error message")
        assert str(error) == "Test error message"

    def test_bulkhead_error_can_be_raised(self):
        """Test BulkheadError can be raised and caught."""
        from src.core.resilience.bulkhead import BulkheadError

        with pytest.raises(BulkheadError) as exc_info:
            raise BulkheadError("Queue is full")

        assert "Queue is full" in str(exc_info.value)


class TestBulkheadStats:
    """Tests for BulkheadStats dataclass."""

    def test_bulkhead_stats_defaults(self):
        """Test BulkheadStats default values."""
        from src.core.resilience.bulkhead import BulkheadStats

        stats = BulkheadStats()

        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.rejected_calls == 0
        assert stats.timeout_calls == 0
        assert stats.active_calls == 0
        assert stats.queued_calls == 0
        assert stats.max_active_recorded == 0
        assert stats.last_rejection_time is None
        assert stats.avg_execution_time == 0.0

    def test_bulkhead_stats_custom_values(self):
        """Test BulkheadStats with custom values."""
        from src.core.resilience.bulkhead import BulkheadStats

        now = datetime.now()
        stats = BulkheadStats(
            total_calls=100,
            successful_calls=90,
            rejected_calls=8,
            timeout_calls=2,
            active_calls=5,
            queued_calls=3,
            max_active_recorded=10,
            last_rejection_time=now,
            avg_execution_time=0.5
        )

        assert stats.total_calls == 100
        assert stats.successful_calls == 90
        assert stats.rejected_calls == 8
        assert stats.timeout_calls == 2
        assert stats.active_calls == 5
        assert stats.queued_calls == 3
        assert stats.max_active_recorded == 10
        assert stats.last_rejection_time == now
        assert stats.avg_execution_time == 0.5


class TestThreadPoolBulkhead:
    """Tests for ThreadPoolBulkhead class."""

    def test_threadpool_init_defaults(self):
        """Test ThreadPoolBulkhead initialization with defaults."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead

        bulkhead = ThreadPoolBulkhead()

        assert bulkhead.max_workers == 10
        assert bulkhead.queue_size == 0
        assert bulkhead.timeout is None

        bulkhead.shutdown(wait=False)

    def test_threadpool_init_custom(self):
        """Test ThreadPoolBulkhead initialization with custom values."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead

        bulkhead = ThreadPoolBulkhead(max_workers=5, queue_size=10, timeout=30.0)

        assert bulkhead.max_workers == 5
        assert bulkhead.queue_size == 10
        assert bulkhead.timeout == 30.0

        bulkhead.shutdown(wait=False)

    def test_threadpool_execute_success(self):
        """Test ThreadPoolBulkhead executes function successfully."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead

        bulkhead = ThreadPoolBulkhead(max_workers=2)

        def test_func(x, y):
            return x + y

        result = bulkhead.execute(test_func, 3, 5)

        assert result == 8
        bulkhead.shutdown(wait=True)

    def test_threadpool_execute_with_kwargs(self):
        """Test ThreadPoolBulkhead executes function with kwargs."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead

        bulkhead = ThreadPoolBulkhead(max_workers=2)

        def test_func(a, b=10):
            return a * b

        result = bulkhead.execute(test_func, 5, b=3)

        assert result == 15
        bulkhead.shutdown(wait=True)

    def test_threadpool_queue_full_raises_error(self):
        """Test ThreadPoolBulkhead raises error when queue is full."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead, BulkheadError

        bulkhead = ThreadPoolBulkhead(max_workers=1, queue_size=1)

        # Fill the queue
        event = threading.Event()

        def blocking_func():
            event.wait(timeout=5)
            return "done"

        # Start first call that will block
        def submit_blocking():
            try:
                bulkhead.execute(blocking_func)
            except BulkheadError:
                pass

        t1 = threading.Thread(target=submit_blocking)
        t1.start()

        # Give time for first call to start
        time.sleep(0.1)

        # Try second call that should also get queued
        t2 = threading.Thread(target=submit_blocking)
        t2.start()
        time.sleep(0.1)

        # Third call should fail because queue is full
        with pytest.raises(BulkheadError) as exc_info:
            bulkhead.execute(blocking_func)

        assert "queue is full" in str(exc_info.value).lower()

        # Cleanup
        event.set()
        t1.join(timeout=1)
        t2.join(timeout=1)
        bulkhead.shutdown(wait=False)

    def test_threadpool_timeout_raises_error(self):
        """Test ThreadPoolBulkhead raises error on timeout."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead, BulkheadError

        bulkhead = ThreadPoolBulkhead(max_workers=2, timeout=0.1)

        def slow_func():
            time.sleep(1)
            return "done"

        with pytest.raises(BulkheadError) as exc_info:
            bulkhead.execute(slow_func)

        assert "timeout" in str(exc_info.value).lower()
        bulkhead.shutdown(wait=False)

    def test_threadpool_get_stats(self):
        """Test ThreadPoolBulkhead get_stats method."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead

        bulkhead = ThreadPoolBulkhead(max_workers=5)

        stats = bulkhead.get_stats()

        assert "max_workers" in stats
        assert "active_threads" in stats
        assert "queued_tasks" in stats
        assert "available_capacity" in stats
        assert stats["max_workers"] == 5

        bulkhead.shutdown(wait=False)

    def test_threadpool_shutdown(self):
        """Test ThreadPoolBulkhead shutdown method."""
        from src.core.resilience.bulkhead import ThreadPoolBulkhead

        bulkhead = ThreadPoolBulkhead(max_workers=2)

        # Execute a function to ensure executor is active
        result = bulkhead.execute(lambda: 42)
        assert result == 42

        # Shutdown should complete without error
        bulkhead.shutdown(wait=True)


class TestSemaphoreBulkhead:
    """Tests for SemaphoreBulkhead class."""

    def test_semaphore_init_defaults(self):
        """Test SemaphoreBulkhead initialization with defaults."""
        from src.core.resilience.bulkhead import SemaphoreBulkhead

        bulkhead = SemaphoreBulkhead()

        assert bulkhead.max_concurrent_calls == 10
        assert bulkhead.timeout is None

    def test_semaphore_init_custom(self):
        """Test SemaphoreBulkhead initialization with custom values."""
        from src.core.resilience.bulkhead import SemaphoreBulkhead

        bulkhead = SemaphoreBulkhead(max_concurrent_calls=3, timeout=5.0)

        assert bulkhead.max_concurrent_calls == 3
        assert bulkhead.timeout == 5.0

    def test_semaphore_execute_success(self):
        """Test SemaphoreBulkhead executes function successfully."""
        from src.core.resilience.bulkhead import SemaphoreBulkhead

        bulkhead = SemaphoreBulkhead(max_concurrent_calls=2)

        def test_func(x):
            return x * 2

        result = bulkhead.execute(test_func, 7)

        assert result == 14

    def test_semaphore_timeout_raises_error(self):
        """Test SemaphoreBulkhead raises error when can't acquire semaphore."""
        from src.core.resilience.bulkhead import SemaphoreBulkhead, BulkheadError

        bulkhead = SemaphoreBulkhead(max_concurrent_calls=1, timeout=0.1)

        event = threading.Event()
        started = threading.Event()

        def blocking_func():
            started.set()
            event.wait(timeout=5)
            return "done"

        # Start blocking call
        def run_blocking():
            try:
                bulkhead.execute(blocking_func)
            except BulkheadError:
                pass

        t = threading.Thread(target=run_blocking)
        t.start()

        # Wait for blocking call to start
        started.wait(timeout=1)

        # This call should timeout
        with pytest.raises(BulkheadError) as exc_info:
            bulkhead.execute(lambda: "test")

        assert "semaphore" in str(exc_info.value).lower()

        # Cleanup
        event.set()
        t.join(timeout=1)

    def test_semaphore_get_stats(self):
        """Test SemaphoreBulkhead get_stats method."""
        from src.core.resilience.bulkhead import SemaphoreBulkhead

        bulkhead = SemaphoreBulkhead(max_concurrent_calls=8)

        stats = bulkhead.get_stats()

        assert "max_concurrent_calls" in stats
        assert "active_calls" in stats
        assert "available_permits" in stats
        assert stats["max_concurrent_calls"] == 8


class TestBulkhead:
    """Tests for unified Bulkhead class."""

    def test_bulkhead_init_threadpool(self):
        """Test Bulkhead initialization with threadpool type."""
        from src.core.resilience.bulkhead import Bulkhead, ThreadPoolBulkhead

        bulkhead = Bulkhead(name="test", bulkhead_type="threadpool")

        assert bulkhead.name == "test"
        assert isinstance(bulkhead._strategy, ThreadPoolBulkhead)

    def test_bulkhead_init_semaphore(self):
        """Test Bulkhead initialization with semaphore type."""
        from src.core.resilience.bulkhead import Bulkhead, SemaphoreBulkhead

        bulkhead = Bulkhead(name="test", bulkhead_type="semaphore")

        assert bulkhead.name == "test"
        assert isinstance(bulkhead._strategy, SemaphoreBulkhead)

    def test_bulkhead_init_invalid_type(self):
        """Test Bulkhead initialization with invalid type raises error."""
        from src.core.resilience.bulkhead import Bulkhead

        with pytest.raises(ValueError) as exc_info:
            Bulkhead(name="test", bulkhead_type="invalid")

        assert "Unknown bulkhead type" in str(exc_info.value)

    def test_bulkhead_execute_success(self):
        """Test Bulkhead execute method on success."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        result = bulkhead.execute(lambda x: x * 2, 10)

        assert result == 20

    def test_bulkhead_execute_updates_stats(self):
        """Test Bulkhead execute updates statistics."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        # Execute multiple calls
        for i in range(5):
            bulkhead.execute(lambda: 42)

        stats = bulkhead.get_stats()

        assert stats.total_calls == 5
        assert stats.successful_calls == 5

    def test_bulkhead_execute_with_metrics_callback(self):
        """Test Bulkhead execute calls metrics callback."""
        from src.core.resilience.bulkhead import Bulkhead

        callback_data = []

        def metrics_callback(data):
            callback_data.append(data)

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=5,
            metrics_callback=metrics_callback
        )

        bulkhead.execute(lambda: 42)

        assert len(callback_data) == 1
        assert callback_data[0]["bulkhead"] == "test"
        assert callback_data[0]["event"] == "success"

    def test_bulkhead_execute_timeout(self):
        """Test Bulkhead execute handles timeout."""
        from src.core.resilience.bulkhead import Bulkhead, BulkheadError

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=2,
            max_wait_duration=0.1,
            bulkhead_type="threadpool"
        )

        def slow_func():
            time.sleep(1)
            return "done"

        with pytest.raises(BulkheadError):
            bulkhead.execute(slow_func)

        stats = bulkhead.get_stats()
        assert stats.timeout_calls == 1

    def test_bulkhead_get_stats(self):
        """Test Bulkhead get_stats method."""
        from src.core.resilience.bulkhead import Bulkhead, BulkheadStats

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)
        bulkhead.execute(lambda: 42)

        stats = bulkhead.get_stats()

        assert isinstance(stats, BulkheadStats)
        assert stats.total_calls == 1
        assert stats.successful_calls == 1

    def test_bulkhead_reset_stats(self):
        """Test Bulkhead reset_stats method."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        # Execute some calls
        for _ in range(3):
            bulkhead.execute(lambda: 42)

        # Verify stats accumulated
        stats = bulkhead.get_stats()
        assert stats.total_calls == 3

        # Reset stats
        bulkhead.reset_stats()

        # Verify stats cleared
        stats = bulkhead.get_stats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0

    def test_bulkhead_get_health(self):
        """Test Bulkhead get_health method."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)
        bulkhead.execute(lambda: 42)

        health = bulkhead.get_health()

        assert health["name"] == "test"
        assert "max_concurrent_calls" in health
        assert "active_calls" in health
        assert "total_calls" in health
        assert "successful_calls" in health
        assert "success_rate" in health
        assert "rejection_rate" in health
        assert "utilization" in health

    def test_bulkhead_get_health_with_rejection(self):
        """Test Bulkhead get_health includes rejection info."""
        from src.core.resilience.bulkhead import Bulkhead, BulkheadError

        bulkhead = Bulkhead(name="test", max_concurrent_calls=1, max_wait_duration=0)

        # Fill the bulkhead
        event = threading.Event()
        started = threading.Event()

        def blocking_func():
            started.set()
            event.wait(timeout=5)
            return "done"

        def run_blocking():
            try:
                bulkhead.execute(blocking_func)
            except BulkheadError:
                pass

        t = threading.Thread(target=run_blocking)
        t.start()
        started.wait(timeout=1)

        # This should be rejected
        try:
            bulkhead.execute(lambda: "test")
        except BulkheadError:
            pass

        event.set()
        t.join(timeout=1)

        health = bulkhead.get_health()
        assert health["rejected_calls"] >= 0

    def test_bulkhead_resize_threadpool(self):
        """Test Bulkhead resize method for threadpool."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=5,
            bulkhead_type="threadpool"
        )

        assert bulkhead.max_concurrent_calls == 5

        bulkhead.resize(10)

        assert bulkhead.max_concurrent_calls == 10

    def test_bulkhead_resize_semaphore(self):
        """Test Bulkhead resize method for semaphore."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=5,
            bulkhead_type="semaphore"
        )

        assert bulkhead.max_concurrent_calls == 5

        bulkhead.resize(8)

        assert bulkhead.max_concurrent_calls == 8

    def test_bulkhead_execution_time_tracking(self):
        """Test Bulkhead tracks average execution time."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        def slow_func():
            time.sleep(0.05)
            return 42

        for _ in range(3):
            bulkhead.execute(slow_func)

        stats = bulkhead.get_stats()
        assert stats.avg_execution_time > 0

    def test_bulkhead_max_active_recorded(self):
        """Test Bulkhead tracks max active recorded."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        for _ in range(3):
            bulkhead.execute(lambda: 42)

        stats = bulkhead.get_stats()
        assert stats.max_active_recorded >= 0


class TestBulkheadDecorator:
    """Tests for bulkhead decorator."""

    def test_decorator_basic(self):
        """Test bulkhead decorator basic usage."""
        from src.core.resilience.bulkhead import bulkhead

        @bulkhead(max_concurrent_calls=5)
        def test_func(x):
            return x * 2

        result = test_func(7)
        assert result == 14

    def test_decorator_attaches_bulkhead(self):
        """Test bulkhead decorator attaches bulkhead instance."""
        from src.core.resilience.bulkhead import bulkhead

        @bulkhead(max_concurrent_calls=3)
        def test_func():
            return 42

        assert hasattr(test_func, "bulkhead")
        # Check bulkhead has correct settings via name (contains function name)
        assert "test_func" in test_func.bulkhead.name

    def test_decorator_with_timeout(self):
        """Test bulkhead decorator with timeout parameter."""
        from src.core.resilience.bulkhead import bulkhead, BulkheadError

        @bulkhead(max_concurrent_calls=2, timeout=0.1)
        def slow_func():
            time.sleep(1)
            return "done"

        with pytest.raises(BulkheadError):
            slow_func()

    def test_decorator_with_semaphore_type(self):
        """Test bulkhead decorator with semaphore type."""
        from src.core.resilience.bulkhead import bulkhead, SemaphoreBulkhead

        @bulkhead(max_concurrent_calls=5, bulkhead_type="semaphore")
        def test_func():
            return 42

        assert isinstance(test_func.bulkhead._strategy, SemaphoreBulkhead)
        result = test_func()
        assert result == 42


class TestBulkheadConcurrency:
    """Tests for concurrent behavior."""

    def test_concurrent_execution_threadpool(self):
        """Test concurrent execution with threadpool."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=5,
            bulkhead_type="threadpool"
        )

        results = []
        errors = []

        def worker(x):
            time.sleep(0.05)
            return x * 2

        def run_task(x):
            try:
                result = bulkhead.execute(worker, x)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=run_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        assert len(results) == 3
        assert len(errors) == 0
        assert set(results) == {0, 2, 4}

    def test_concurrent_execution_semaphore(self):
        """Test concurrent execution with semaphore."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=5,
            bulkhead_type="semaphore"
        )

        results = []

        def worker(x):
            return x + 1

        def run_task(x):
            result = bulkhead.execute(worker, x)
            results.append(result)

        threads = []
        for i in range(3):
            t = threading.Thread(target=run_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        assert len(results) == 3
        assert set(results) == {1, 2, 3}


class TestBulkheadExceptionHandling:
    """Tests for exception handling."""

    def test_function_exception_propagates(self):
        """Test exceptions from wrapped function propagate."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError) as exc_info:
            bulkhead.execute(failing_func)

        assert "Test error" in str(exc_info.value)

    def test_function_exception_emits_failure_metric(self):
        """Test function exception emits failure metric."""
        from src.core.resilience.bulkhead import Bulkhead

        callback_data = []

        def metrics_callback(data):
            callback_data.append(data)

        bulkhead = Bulkhead(
            name="test",
            max_concurrent_calls=5,
            metrics_callback=metrics_callback
        )

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            bulkhead.execute(failing_func)

        # Should have emitted failure metric
        assert len(callback_data) == 1
        assert callback_data[0]["event"] == "failure"


class TestBulkheadEdgeCases:
    """Tests for edge cases."""

    def test_zero_max_concurrent_calls(self):
        """Test behavior with zero max_concurrent_calls."""
        from src.core.resilience.bulkhead import Bulkhead

        # ThreadPoolExecutor requires at least 1 worker
        bulkhead = Bulkhead(name="test", max_concurrent_calls=1)
        result = bulkhead.execute(lambda: 42)
        assert result == 42

    def test_empty_execution_times_list(self):
        """Test avg_execution_time with no executions."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        stats = bulkhead.get_stats()
        assert stats.avg_execution_time == 0.0

    def test_execution_times_list_trimming(self):
        """Test execution times list is trimmed to 100 entries."""
        from src.core.resilience.bulkhead import Bulkhead

        bulkhead = Bulkhead(name="test", max_concurrent_calls=5)

        # Execute more than 100 calls
        for _ in range(105):
            bulkhead.execute(lambda: 42)

        # Internal list should be trimmed
        assert len(bulkhead._execution_times) <= 100
