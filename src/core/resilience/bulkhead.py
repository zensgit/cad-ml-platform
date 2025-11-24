"""
Bulkhead Pattern Implementation
隔板模式 - 资源隔离和故障限制
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BulkheadError(Exception):
    """隔板异常"""
    pass


@dataclass
class BulkheadStats:
    """隔板统计信息"""
    total_calls: int = 0
    successful_calls: int = 0
    rejected_calls: int = 0
    timeout_calls: int = 0
    active_calls: int = 0
    queued_calls: int = 0
    max_active_recorded: int = 0
    last_rejection_time: Optional[datetime] = None
    avg_execution_time: float = 0.0


class BulkheadStrategy(ABC):
    """隔板策略抽象基类"""

    @abstractmethod
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行函数调用"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class ThreadPoolBulkhead(BulkheadStrategy):
    """
    线程池隔板实现

    使用独立的线程池隔离资源
    """

    def __init__(
        self,
        max_workers: int = 10,
        queue_size: int = 0,
        timeout: Optional[float] = None
    ):
        """
        初始化线程池隔板

        Args:
            max_workers: 最大工作线程数
            queue_size: 队列大小（0表示无限制）
            timeout: 执行超时时间（秒）
        """
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_count = 0
        self._queue_count = 0
        self._lock = threading.Lock()

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """通过线程池执行函数"""
        with self._lock:
            if self.queue_size > 0 and self._queue_count >= self.queue_size:
                raise BulkheadError(
                    f"Thread pool queue is full (size: {self.queue_size})"
                )
            self._queue_count += 1

        try:
            future = self.executor.submit(self._wrapped_func, func, *args, **kwargs)

            if self.timeout:
                result = future.result(timeout=self.timeout)
            else:
                result = future.result()

            return result

        except FutureTimeoutError:
            raise BulkheadError(f"Execution timeout after {self.timeout} seconds")
        finally:
            with self._lock:
                self._queue_count = max(0, self._queue_count - 1)

    def _wrapped_func(self, func: Callable, *args, **kwargs) -> Any:
        """包装函数以跟踪活跃调用"""
        with self._lock:
            self._active_count += 1
        try:
            return func(*args, **kwargs)
        finally:
            with self._lock:
                self._active_count -= 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "active_threads": self._active_count,
                "queued_tasks": self._queue_count,
                "available_capacity": self.max_workers - self._active_count
            }

    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)


class SemaphoreBulkhead(BulkheadStrategy):
    """
    信号量隔板实现

    使用信号量限制并发访问
    """

    def __init__(
        self,
        max_concurrent_calls: int = 10,
        timeout: Optional[float] = None
    ):
        """
        初始化信号量隔板

        Args:
            max_concurrent_calls: 最大并发调用数
            timeout: 获取信号量的超时时间
        """
        self.max_concurrent_calls = max_concurrent_calls
        self.timeout = timeout
        self.semaphore = threading.BoundedSemaphore(max_concurrent_calls)
        self._active_count = 0
        self._lock = threading.Lock()

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """通过信号量控制执行函数"""
        acquired = self.semaphore.acquire(timeout=self.timeout)
        if not acquired:
            raise BulkheadError(
                f"Failed to acquire semaphore within {self.timeout} seconds"
            )

        with self._lock:
            self._active_count += 1

        try:
            return func(*args, **kwargs)
        finally:
            with self._lock:
                self._active_count -= 1
            self.semaphore.release()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "max_concurrent_calls": self.max_concurrent_calls,
                "active_calls": self._active_count,
                "available_permits": self.max_concurrent_calls - self._active_count
            }


class Bulkhead:
    """
    统一的隔板接口

    功能:
    - 资源隔离
    - 故障限制
    - 统计收集
    - 动态调整
    """

    def __init__(
        self,
        name: str,
        max_concurrent_calls: int = 10,
        max_wait_duration: float = 0,
        bulkhead_type: str = "threadpool",
        metrics_callback: Optional[Callable] = None
    ):
        """
        初始化隔板

        Args:
            name: 隔板名称
            max_concurrent_calls: 最大并发调用数
            max_wait_duration: 最大等待时间（0表示不等待）
            bulkhead_type: 隔板类型（threadpool, semaphore）
            metrics_callback: 指标收集回调函数
        """
        self.name = name
        self.max_concurrent_calls = max_concurrent_calls
        self.max_wait_duration = max_wait_duration
        self.metrics_callback = metrics_callback
        self._stats = BulkheadStats()
        self._execution_times = []
        self._lock = threading.Lock()

        # 初始化隔板策略
        if bulkhead_type == "threadpool":
            self._strategy = ThreadPoolBulkhead(
                max_workers=max_concurrent_calls,
                timeout=max_wait_duration if max_wait_duration > 0 else None
            )
        elif bulkhead_type == "semaphore":
            self._strategy = SemaphoreBulkhead(
                max_concurrent_calls=max_concurrent_calls,
                timeout=max_wait_duration if max_wait_duration > 0 else None
            )
        else:
            raise ValueError(f"Unknown bulkhead type: {bulkhead_type}")

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过隔板执行函数调用

        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果

        Raises:
            BulkheadError: 资源不可用或超时
        """
        start_time = time.time()

        with self._lock:
            self._stats.total_calls += 1
            strategy_stats = self._strategy.get_stats()

            # 检查是否应该拒绝
            if "active_calls" in strategy_stats:
                active = strategy_stats["active_calls"]
                if active >= self.max_concurrent_calls and self.max_wait_duration == 0:
                    self._stats.rejected_calls += 1
                    self._stats.last_rejection_time = datetime.now()
                    self._emit_metrics("rejected", 0)
                    raise BulkheadError(
                        f"Bulkhead '{self.name}' is full "
                        f"({active}/{self.max_concurrent_calls})"
                    )

                self._stats.active_calls = active
                self._stats.max_active_recorded = max(
                    self._stats.max_active_recorded, active
                )

        try:
            # 执行实际调用
            result = self._strategy.execute(func, *args, **kwargs)
            elapsed = time.time() - start_time

            with self._lock:
                self._stats.successful_calls += 1
                self._execution_times.append(elapsed)
                if len(self._execution_times) > 100:
                    self._execution_times = self._execution_times[-100:]
                self._stats.avg_execution_time = sum(self._execution_times) / len(self._execution_times)

            self._emit_metrics("success", elapsed)
            return result

        except BulkheadError as e:
            if "timeout" in str(e).lower():
                with self._lock:
                    self._stats.timeout_calls += 1
                self._emit_metrics("timeout", time.time() - start_time)
            else:
                with self._lock:
                    self._stats.rejected_calls += 1
                self._emit_metrics("rejected", time.time() - start_time)
            raise

        except Exception as e:
            self._emit_metrics("failure", time.time() - start_time)
            raise

    def get_stats(self) -> BulkheadStats:
        """获取统计信息"""
        with self._lock:
            strategy_stats = self._strategy.get_stats()
            if "active_calls" in strategy_stats:
                self._stats.active_calls = strategy_stats["active_calls"]
            if "queued_tasks" in strategy_stats:
                self._stats.queued_calls = strategy_stats["queued_tasks"]
            return self._stats

    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._stats = BulkheadStats()
            self._execution_times = []

    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        stats = self.get_stats()
        strategy_stats = self._strategy.get_stats()

        return {
            "name": self.name,
            "type": type(self._strategy).__name__,
            "max_concurrent_calls": self.max_concurrent_calls,
            "active_calls": stats.active_calls,
            "queued_calls": stats.queued_calls,
            "total_calls": stats.total_calls,
            "successful_calls": stats.successful_calls,
            "rejected_calls": stats.rejected_calls,
            "timeout_calls": stats.timeout_calls,
            "rejection_rate": (
                stats.rejected_calls / stats.total_calls
                if stats.total_calls > 0 else 0
            ),
            "success_rate": (
                stats.successful_calls / stats.total_calls
                if stats.total_calls > 0 else 0
            ),
            "avg_execution_time": stats.avg_execution_time,
            "max_active_recorded": stats.max_active_recorded,
            "utilization": (
                stats.active_calls / self.max_concurrent_calls
                if self.max_concurrent_calls > 0 else 0
            ),
            "last_rejection": (
                stats.last_rejection_time.isoformat()
                if stats.last_rejection_time else None
            ),
            **strategy_stats
        }

    def resize(self, new_max_concurrent_calls: int):
        """动态调整隔板大小"""
        with self._lock:
            old_size = self.max_concurrent_calls
            self.max_concurrent_calls = new_max_concurrent_calls

            # 重新创建策略
            if isinstance(self._strategy, ThreadPoolBulkhead):
                self._strategy.shutdown(wait=False)
                self._strategy = ThreadPoolBulkhead(
                    max_workers=new_max_concurrent_calls,
                    timeout=self.max_wait_duration if self.max_wait_duration > 0 else None
                )
            elif isinstance(self._strategy, SemaphoreBulkhead):
                self._strategy = SemaphoreBulkhead(
                    max_concurrent_calls=new_max_concurrent_calls,
                    timeout=self.max_wait_duration if self.max_wait_duration > 0 else None
                )

            logger.info(
                f"Bulkhead '{self.name}' resized from {old_size} to "
                f"{new_max_concurrent_calls}"
            )

    def _emit_metrics(self, event_type: str, duration: float):
        """发送指标"""
        if self.metrics_callback:
            self.metrics_callback({
                "bulkhead": self.name,
                "event": event_type,
                "duration": duration,
                "active_calls": self._stats.active_calls,
                "timestamp": datetime.now().isoformat()
            })


def bulkhead(
    max_concurrent_calls: int = 10,
    bulkhead_type: str = "threadpool",
    timeout: float = 0
):
    """
    隔板装饰器

    Usage:
        @bulkhead(max_concurrent_calls=5)
        def resource_intensive_operation():
            # Operation logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        bulk = Bulkhead(
            name=f"{func.__module__}.{func.__name__}",
            max_concurrent_calls=max_concurrent_calls,
            max_wait_duration=timeout,
            bulkhead_type=bulkhead_type
        )

        def wrapper(*args, **kwargs):
            return bulk.execute(func, *args, **kwargs)

        wrapper.bulkhead = bulk
        return wrapper

    return decorator