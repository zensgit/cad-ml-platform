"""
Retry Policy Pattern Implementation
重试策略模式 - 处理瞬时故障和提高可靠性
"""

import time
import random
import logging
from typing import Callable, Optional, List, Type, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """重试异常"""
    pass


@dataclass
class RetryStats:
    """重试统计信息"""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time: float = 0.0
    last_retry_time: Optional[datetime] = None
    error_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.error_distribution is None:
            self.error_distribution = {}


class RetryStrategy(ABC):
    """重试策略抽象基类"""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """获取下次重试的延迟时间"""
        pass


class FixedDelay(RetryStrategy):
    """固定延迟策略"""

    def __init__(self, delay: float):
        self.delay = delay

    def get_delay(self, attempt: int) -> float:
        return self.delay


class LinearBackoff(RetryStrategy):
    """线性退避策略"""

    def __init__(self, initial_delay: float, increment: float, max_delay: float = 300):
        self.initial_delay = initial_delay
        self.increment = increment
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay + (attempt - 1) * self.increment
        return min(delay, self.max_delay)


class ExponentialBackoff(RetryStrategy):
    """指数退避策略"""

    def __init__(
        self,
        base_delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 300.0,
        jitter: bool = True
    ):
        self.base_delay = base_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            # 添加随机抖动以避免雷鸣群效应
            delay = delay * (0.5 + random.random())

        return delay


class FibonacciBackoff(RetryStrategy):
    """斐波那契退避策略"""

    def __init__(self, initial_delay: float = 1.0, max_delay: float = 300.0):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._fib_cache = {0: 0, 1: 1}

    def _fibonacci(self, n: int) -> int:
        if n in self._fib_cache:
            return self._fib_cache[n]
        self._fib_cache[n] = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        return self._fib_cache[n]

    def get_delay(self, attempt: int) -> float:
        fib_value = self._fibonacci(attempt)
        delay = self.initial_delay * fib_value
        return min(delay, self.max_delay)


class RetryPolicy:
    """
    重试策略实现

    功能:
    - 多种重试策略支持
    - 可配置的重试条件
    - 统计信息收集
    - 自定义异常处理
    """

    def __init__(
        self,
        name: str,
        max_attempts: int = 3,
        strategy: Optional[RetryStrategy] = None,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
        metrics_callback: Optional[Callable] = None
    ):
        """
        初始化重试策略

        Args:
            name: 策略名称
            max_attempts: 最大重试次数
            strategy: 重试策略实例
            retryable_exceptions: 可重试的异常类型列表
            non_retryable_exceptions: 不可重试的异常类型列表
            on_retry: 重试时的回调函数
            metrics_callback: 指标收集回调函数
        """
        self.name = name
        self.max_attempts = max_attempts
        self.strategy = strategy or ExponentialBackoff()
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.non_retryable_exceptions = non_retryable_exceptions or []
        self.on_retry = on_retry
        self.metrics_callback = metrics_callback
        self._stats = RetryStats()

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行带重试的函数调用

        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果

        Raises:
            RetryError: 重试次数用尽
        """
        last_exception = None
        total_delay = 0.0

        for attempt in range(1, self.max_attempts + 1):
            try:
                # 执行函数
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # 记录成功
                if attempt > 1:
                    self._stats.successful_retries += 1
                    logger.info(
                        f"Retry successful for '{self.name}' "
                        f"after {attempt} attempts"
                    )

                self._emit_metrics("success", attempt, elapsed, total_delay)
                return result

            except Exception as e:
                last_exception = e
                self._stats.total_attempts += 1

                # 记录错误分布
                error_type = type(e).__name__
                self._stats.error_distribution[error_type] = \
                    self._stats.error_distribution.get(error_type, 0) + 1

                # 检查是否应该重试
                if not self._should_retry(e):
                    logger.error(
                        f"Non-retryable exception for '{self.name}': {e}"
                    )
                    self._emit_metrics("non_retryable", attempt, 0, total_delay)
                    raise

                if attempt == self.max_attempts:
                    logger.error(
                        f"Max retry attempts ({self.max_attempts}) reached "
                        f"for '{self.name}'"
                    )
                    self._stats.failed_retries += 1
                    self._emit_metrics("exhausted", attempt, 0, total_delay)
                    raise RetryError(
                        f"Operation failed after {self.max_attempts} attempts"
                    ) from last_exception

                # 计算延迟
                delay = self.strategy.get_delay(attempt)
                total_delay += delay
                self._stats.total_delay_time += delay
                self._stats.last_retry_time = datetime.now()

                logger.warning(
                    f"Attempt {attempt} failed for '{self.name}': {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )

                # 执行重试回调
                if self.on_retry:
                    self.on_retry(e, attempt)

                self._emit_metrics("retry", attempt, 0, delay)

                # 等待
                time.sleep(delay)

    def _should_retry(self, exception: Exception) -> bool:
        """判断异常是否应该重试"""
        # 检查不可重试异常
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False

        # 检查可重试异常
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True

        return False

    def get_stats(self) -> RetryStats:
        """获取统计信息"""
        return self._stats

    def reset_stats(self):
        """重置统计信息"""
        self._stats = RetryStats()

    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "name": self.name,
            "max_attempts": self.max_attempts,
            "total_attempts": self._stats.total_attempts,
            "successful_retries": self._stats.successful_retries,
            "failed_retries": self._stats.failed_retries,
            "success_rate": (
                self._stats.successful_retries / self._stats.total_attempts
                if self._stats.total_attempts > 0 else 0
            ),
            "total_delay_time": self._stats.total_delay_time,
            "avg_delay_time": (
                self._stats.total_delay_time / self._stats.total_attempts
                if self._stats.total_attempts > 0 else 0
            ),
            "last_retry": (
                self._stats.last_retry_time.isoformat()
                if self._stats.last_retry_time else None
            ),
            "error_distribution": self._stats.error_distribution
        }

    def _emit_metrics(
        self,
        event_type: str,
        attempt: int,
        duration: float,
        delay: float
    ):
        """发送指标"""
        if self.metrics_callback:
            self.metrics_callback({
                "retry_policy": self.name,
                "event": event_type,
                "attempt": attempt,
                "duration": duration,
                "delay": delay,
                "timestamp": datetime.now().isoformat()
            })


def retry(
    max_attempts: int = 3,
    strategy: Optional[RetryStrategy] = None,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable] = None
):
    """
    重试装饰器

    Usage:
        @retry(max_attempts=3, strategy=ExponentialBackoff())
        def unreliable_operation():
            # Operation that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        policy = RetryPolicy(
            name=f"{func.__module__}.{func.__name__}",
            max_attempts=max_attempts,
            strategy=strategy,
            retryable_exceptions=retryable_exceptions,
            on_retry=on_retry
        )

        def wrapper(*args, **kwargs):
            return policy.execute(func, *args, **kwargs)

        wrapper.retry_policy = policy
        return wrapper

    return decorator


class AdaptiveRetry(RetryPolicy):
    """
    自适应重试策略

    根据历史成功率动态调整重试参数
    """

    def __init__(
        self,
        name: str,
        initial_max_attempts: int = 3,
        min_success_rate: float = 0.5,
        adjustment_window: int = 100,
        **kwargs
    ):
        super().__init__(name, initial_max_attempts, **kwargs)
        self.min_success_rate = min_success_rate
        self.adjustment_window = adjustment_window
        self._window_counter = 0

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        result = super().execute(func, *args, **kwargs)

        # 自适应调整
        self._window_counter += 1
        if self._window_counter >= self.adjustment_window:
            self._adjust_parameters()
            self._window_counter = 0

        return result

    def _adjust_parameters(self):
        """根据成功率调整参数"""
        stats = self.get_stats()
        if stats.total_attempts == 0:
            return

        success_rate = stats.successful_retries / stats.total_attempts

        if success_rate < self.min_success_rate:
            # 降低成功率，增加重试次数
            self.max_attempts = min(self.max_attempts + 1, 10)
            logger.info(
                f"Adaptive retry '{self.name}': "
                f"Increasing max_attempts to {self.max_attempts}"
            )
        elif success_rate > 0.8:
            # 高成功率，减少重试次数
            self.max_attempts = max(self.max_attempts - 1, 1)
            logger.info(
                f"Adaptive retry '{self.name}': "
                f"Decreasing max_attempts to {self.max_attempts}"
            )

        # 重置统计窗口
        self.reset_stats()