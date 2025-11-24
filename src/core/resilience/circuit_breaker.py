"""
Circuit Breaker Pattern Implementation
熔断器模式 - 防止级联故障和资源耗尽
"""

import time
import threading
from enum import Enum
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 正常状态，允许请求通过
    OPEN = "open"           # 熔断状态，阻止请求
    HALF_OPEN = "half_open" # 半开状态，允许有限请求进行测试


class CircuitBreakerError(Exception):
    """熔断器异常"""
    pass


@dataclass
class CircuitBreakerStats:
    """熔断器统计信息"""
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_calls: int = 0
    state_transitions: list = field(default_factory=list)
    error_distribution: Dict[str, int] = field(default_factory=dict)


class CircuitBreaker:
    """
    熔断器实现

    功能:
    - 监控调用失败率
    - 自动熔断故障服务
    - 定时恢复测试
    - 统计和指标收集
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        metrics_callback: Optional[Callable] = None
    ):
        """
        初始化熔断器

        Args:
            name: 熔断器名称
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断后恢复测试的等待时间（秒）
            expected_exception: 需要捕获的异常类型
            half_open_max_calls: 半开状态允许的最大测试调用数
            success_threshold: 半开状态恢复到关闭状态所需的成功次数
            metrics_callback: 指标收集回调函数
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.metrics_callback = metrics_callback

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_state_change = datetime.now()
        self._lock = threading.RLock()
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """获取统计信息"""
        return self._stats

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器执行函数调用

        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果

        Raises:
            CircuitBreakerError: 熔断器开启时
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                self._check_state_transition()
                if self._state == CircuitState.OPEN:
                    self._record_rejection()
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Recovery attempt in {self._time_until_recovery()} seconds"
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._record_rejection()
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN. "
                        f"Max test calls ({self.half_open_max_calls}) reached"
                    )
                self._half_open_calls += 1

        try:
            # 执行实际调用
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            with self._lock:
                self._on_success(elapsed)

            return result

        except self.expected_exception as e:
            with self._lock:
                self._on_failure(e)
            raise

    def _check_state_transition(self):
        """检查并执行状态转换"""
        if self._state == CircuitState.OPEN:
            if self._time_until_recovery() <= 0:
                self._transition_to_half_open()

    def _transition_to_half_open(self):
        """转换到半开状态"""
        logger.info(f"Circuit breaker '{self.name}' transitioning OPEN -> HALF_OPEN")
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._stats.consecutive_failures = 0
        self._last_state_change = datetime.now()
        self._record_state_transition(CircuitState.OPEN, CircuitState.HALF_OPEN)

    def _transition_to_open(self):
        """转换到开启状态"""
        logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN")
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now()
        self._record_state_transition(self._state, CircuitState.OPEN)

    def _transition_to_closed(self):
        """转换到关闭状态"""
        logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
        prev_state = self._state
        self._state = CircuitState.CLOSED
        self._stats.consecutive_failures = 0
        self._half_open_calls = 0
        self._last_state_change = datetime.now()
        self._record_state_transition(prev_state, CircuitState.CLOSED)

    def _on_success(self, elapsed_time: float):
        """处理成功调用"""
        self._stats.success_count += 1
        self._stats.total_calls += 1
        self._stats.last_success_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.success_count >= self.success_threshold:
                self._transition_to_closed()
        else:
            self._stats.consecutive_failures = 0

        self._emit_metrics("success", elapsed_time)

    def _on_failure(self, exception: Exception):
        """处理失败调用"""
        self._stats.failure_count += 1
        self._stats.total_calls += 1
        self._stats.consecutive_failures += 1
        self._stats.last_failure_time = datetime.now()

        # 记录错误分布
        error_type = type(exception).__name__
        self._stats.error_distribution[error_type] = \
            self._stats.error_distribution.get(error_type, 0) + 1

        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._transition_to_open()
        elif self._state == CircuitState.HALF_OPEN:
            self._transition_to_open()

        self._emit_metrics("failure", 0, error_type)

    def _record_rejection(self):
        """记录请求拒绝"""
        self._emit_metrics("rejected", 0)

    def _record_state_transition(self, from_state: CircuitState, to_state: CircuitState):
        """记录状态转换"""
        transition = {
            "timestamp": datetime.now().isoformat(),
            "from": from_state.value,
            "to": to_state.value
        }
        self._stats.state_transitions.append(transition)
        self._emit_metrics("state_change", 0, f"{from_state.value}_to_{to_state.value}")

    def _time_until_recovery(self) -> float:
        """计算到恢复测试的剩余时间"""
        elapsed = (datetime.now() - self._last_state_change).total_seconds()
        return max(0, self.recovery_timeout - elapsed)

    def _emit_metrics(self, event_type: str, duration: float = 0, detail: str = ""):
        """发送指标"""
        if self.metrics_callback:
            self.metrics_callback({
                "circuit_breaker": self.name,
                "state": self._state.value,
                "event": event_type,
                "duration": duration,
                "detail": detail,
                "timestamp": datetime.now().isoformat()
            })

    def reset(self):
        """手动重置熔断器"""
        with self._lock:
            self._stats = CircuitBreakerStats()
            self._transition_to_closed()

    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "success_count": self._stats.success_count,
                "failure_count": self._stats.failure_count,
                "consecutive_failures": self._stats.consecutive_failures,
                "total_calls": self._stats.total_calls,
                "failure_rate": (
                    self._stats.failure_count / self._stats.total_calls
                    if self._stats.total_calls > 0 else 0
                ),
                "last_failure": (
                    self._stats.last_failure_time.isoformat()
                    if self._stats.last_failure_time else None
                ),
                "last_success": (
                    self._stats.last_success_time.isoformat()
                    if self._stats.last_success_time else None
                ),
                "error_distribution": self._stats.error_distribution,
                "recent_transitions": self._stats.state_transitions[-5:]
            }


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
):
    """
    熔断器装饰器

    Usage:
        @circuit_breaker(name="external_api", failure_threshold=3)
        def call_external_api():
            # API call logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )

        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator