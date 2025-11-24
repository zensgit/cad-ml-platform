"""
Rate Limiter Pattern Implementation
限流器模式 - 防止系统过载和资源耗尽
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """限流异常"""
    pass


@dataclass
class RateLimiterStats:
    """限流器统计信息"""
    allowed_count: int = 0
    rejected_count: int = 0
    total_requests: int = 0
    last_rejection_time: Optional[datetime] = None
    current_rate: float = 0.0


class RateLimiterAlgorithm(ABC):
    """限流算法抽象基类"""

    @abstractmethod
    def allow_request(self) -> bool:
        """判断请求是否允许通过"""
        pass

    @abstractmethod
    def get_wait_time(self) -> float:
        """获取下次允许请求的等待时间"""
        pass


class TokenBucket(RateLimiterAlgorithm):
    """
    令牌桶算法实现

    特点:
    - 固定速率生成令牌
    - 允许突发流量（桶容量）
    - 平滑限流
    """

    def __init__(self, rate: float, capacity: int):
        """
        初始化令牌桶

        Args:
            rate: 每秒生成的令牌数
            capacity: 桶的最大容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """尝试获取令牌"""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def get_wait_time(self) -> float:
        """计算获取下一个令牌的等待时间"""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                return 0
            return (1 - self.tokens) / self.rate

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now


class SlidingWindowLog(RateLimiterAlgorithm):
    """
    滑动窗口日志算法

    特点:
    - 精确计数
    - 无突发流量
    - 内存消耗较大
    """

    def __init__(self, rate: int, window_size: int = 1):
        """
        初始化滑动窗口

        Args:
            rate: 窗口内允许的最大请求数
            window_size: 窗口大小（秒）
        """
        self.rate = rate
        self.window_size = window_size
        self.requests = []
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """检查请求是否允许"""
        with self.lock:
            now = time.time()
            self._clean_old_requests(now)

            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True
            return False

    def get_wait_time(self) -> float:
        """计算下次请求的等待时间"""
        with self.lock:
            now = time.time()
            self._clean_old_requests(now)

            if len(self.requests) < self.rate:
                return 0

            oldest_request = self.requests[0]
            return max(0, self.window_size - (now - oldest_request))

    def _clean_old_requests(self, now: float):
        """清理过期的请求记录"""
        cutoff = now - self.window_size
        self.requests = [t for t in self.requests if t > cutoff]


class LeakyBucket(RateLimiterAlgorithm):
    """
    漏桶算法实现

    特点:
    - 固定速率处理
    - 缓冲突发流量
    - 流量整形
    """

    def __init__(self, rate: float, capacity: int):
        """
        初始化漏桶

        Args:
            rate: 每秒漏出的请求数
            capacity: 桶的最大容量
        """
        self.rate = rate
        self.capacity = capacity
        self.queue_size = 0
        self.last_leak = time.time()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """尝试将请求加入队列"""
        with self.lock:
            self._leak()
            if self.queue_size < self.capacity:
                self.queue_size += 1
                return True
            return False

    def get_wait_time(self) -> float:
        """计算队列清空的等待时间"""
        with self.lock:
            self._leak()
            if self.queue_size < self.capacity:
                return 0
            return self.queue_size / self.rate

    def _leak(self):
        """漏出请求"""
        now = time.time()
        elapsed = now - self.last_leak
        leaked = elapsed * self.rate
        self.queue_size = max(0, self.queue_size - leaked)
        self.last_leak = now


class RateLimiter:
    """
    统一的限流器接口

    功能:
    - 多种限流算法支持
    - 统计信息收集
    - 指标发送
    - 动态调整
    """

    def __init__(
        self,
        name: str,
        rate: float,
        burst: Optional[int] = None,
        algorithm: str = "token_bucket",
        metrics_callback: Optional[Callable] = None
    ):
        """
        初始化限流器

        Args:
            name: 限流器名称
            rate: 限流速率（请求/秒）
            burst: 突发容量（仅对支持的算法有效）
            algorithm: 限流算法（token_bucket, sliding_window, leaky_bucket）
            metrics_callback: 指标收集回调函数
        """
        self.name = name
        self.rate = rate
        self.burst = burst or int(rate * 1.5)
        self.metrics_callback = metrics_callback
        self._stats = RateLimiterStats()

        # 初始化限流算法
        if algorithm == "token_bucket":
            self._algorithm = TokenBucket(rate, self.burst)
        elif algorithm == "sliding_window":
            self._algorithm = SlidingWindowLog(int(rate))
        elif algorithm == "leaky_bucket":
            self._algorithm = LeakyBucket(rate, self.burst)
        else:
            raise ValueError(f"Unknown rate limiting algorithm: {algorithm}")

        self._lock = threading.Lock()

    def allow_request(self, identifier: Optional[str] = None) -> bool:
        """
        检查请求是否允许通过

        Args:
            identifier: 请求标识符（用于分组限流）

        Returns:
            是否允许请求
        """
        with self._lock:
            self._stats.total_requests += 1

            if self._algorithm.allow_request():
                self._stats.allowed_count += 1
                self._emit_metrics("allowed", identifier)
                return True
            else:
                self._stats.rejected_count += 1
                self._stats.last_rejection_time = datetime.now()
                self._emit_metrics("rejected", identifier)
                return False

    def acquire(self, identifier: Optional[str] = None, timeout: float = 0) -> bool:
        """
        获取许可（阻塞式）

        Args:
            identifier: 请求标识符
            timeout: 最大等待时间（秒），0表示不等待

        Returns:
            是否成功获取许可
        """
        if self.allow_request(identifier):
            return True

        if timeout <= 0:
            return False

        wait_time = self._algorithm.get_wait_time()
        if wait_time > timeout:
            return False

        time.sleep(wait_time)
        return self.allow_request(identifier)

    def get_wait_time(self) -> float:
        """获取下次允许请求的等待时间"""
        return self._algorithm.get_wait_time()

    def get_stats(self) -> RateLimiterStats:
        """获取统计信息"""
        with self._lock:
            # 计算当前速率
            if self._stats.total_requests > 0:
                self._stats.current_rate = (
                    self._stats.allowed_count / self._stats.total_requests
                ) * self.rate
            return self._stats

    def reset(self):
        """重置限流器"""
        with self._lock:
            self._stats = RateLimiterStats()
            # 重新初始化算法
            if isinstance(self._algorithm, TokenBucket):
                self._algorithm = TokenBucket(self.rate, self.burst)
            elif isinstance(self._algorithm, SlidingWindowLog):
                self._algorithm = SlidingWindowLog(int(self.rate))
            elif isinstance(self._algorithm, LeakyBucket):
                self._algorithm = LeakyBucket(self.rate, self.burst)

    def update_rate(self, new_rate: float, new_burst: Optional[int] = None):
        """动态更新限流速率"""
        with self._lock:
            self.rate = new_rate
            self.burst = new_burst or int(new_rate * 1.5)

            # 更新算法参数
            if isinstance(self._algorithm, TokenBucket):
                self._algorithm = TokenBucket(new_rate, self.burst)
            elif isinstance(self._algorithm, SlidingWindowLog):
                self._algorithm = SlidingWindowLog(int(new_rate))
            elif isinstance(self._algorithm, LeakyBucket):
                self._algorithm = LeakyBucket(new_rate, self.burst)

    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        stats = self.get_stats()
        return {
            "name": self.name,
            "rate": self.rate,
            "burst": self.burst,
            "allowed_count": stats.allowed_count,
            "rejected_count": stats.rejected_count,
            "total_requests": stats.total_requests,
            "rejection_rate": (
                stats.rejected_count / stats.total_requests
                if stats.total_requests > 0 else 0
            ),
            "current_rate": stats.current_rate,
            "last_rejection": (
                stats.last_rejection_time.isoformat()
                if stats.last_rejection_time else None
            )
        }

    def _emit_metrics(self, event_type: str, identifier: Optional[str] = None):
        """发送指标"""
        if self.metrics_callback:
            self.metrics_callback({
                "rate_limiter": self.name,
                "event": event_type,
                "identifier": identifier or "default",
                "timestamp": datetime.now().isoformat()
            })


def rate_limit(
    rate: float,
    burst: Optional[int] = None,
    algorithm: str = "token_bucket",
    raise_on_limit: bool = True
):
    """
    限流装饰器

    Usage:
        @rate_limit(rate=10, burst=15)
        def api_call():
            # API logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter(
            name=f"{func.__module__}.{func.__name__}",
            rate=rate,
            burst=burst,
            algorithm=algorithm
        )

        def wrapper(*args, **kwargs):
            if not limiter.allow_request():
                if raise_on_limit:
                    wait_time = limiter.get_wait_time()
                    raise RateLimitError(
                        f"Rate limit exceeded. Try again in {wait_time:.2f} seconds"
                    )
                return None
            return func(*args, **kwargs)

        wrapper.rate_limiter = limiter
        return wrapper

    return decorator