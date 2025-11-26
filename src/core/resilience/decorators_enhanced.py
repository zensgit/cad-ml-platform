"""
Enhanced Resilience Decorators
增强的韧性装饰器 - 用于主路径的全量接入
"""

import functools
import asyncio
import time
from typing import Callable, Any, Optional, Dict, TypeVar
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge
import logging

from src.api.health_resilience import resilience_collector, CircuitState
from src.core.resilience.circuit_breaker import CircuitBreaker
from src.core.resilience.rate_limiter import TokenBucketRateLimiter
from src.core.resilience.retry_policy import ExponentialBackoff, RetryPolicy
from src.core.resilience.bulkhead import SemaphoreBulkhead

logger = logging.getLogger(__name__)

# Prometheus 指标
resilience_circuit_open_total = Counter(
    'resilience_circuit_open_total',
    'Total number of circuit breaker opens',
    ['service', 'endpoint']
)

resilience_circuit_state = Gauge(
    'resilience_circuit_state',
    'Current circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['service', 'endpoint']
)

resilience_rate_limit_rejected = Counter(
    'resilience_rate_limit_rejected',
    'Total number of rate limited requests',
    ['service', 'endpoint']
)

resilience_rate_limit_tokens = Gauge(
    'resilience_rate_limit_tokens',
    'Current available tokens in rate limiter',
    ['service', 'endpoint']
)

resilience_retry_attempts = Counter(
    'resilience_retry_attempts',
    'Total number of retry attempts',
    ['service', 'endpoint']
)

resilience_bulkhead_rejected = Counter(
    'resilience_bulkhead_rejected',
    'Total number of bulkhead rejected requests',
    ['service', 'endpoint']
)

resilience_request_duration = Histogram(
    'resilience_request_duration_seconds',
    'Request duration with resilience overhead',
    ['service', 'endpoint', 'status']
)


T = TypeVar('T')


@dataclass
class ResilienceConfig:
    """韧性配置"""
    # Circuit Breaker
    cb_enabled: bool = True
    cb_failure_threshold: int = 5
    cb_recovery_timeout: int = 60
    cb_half_open_calls: int = 3

    # Rate Limiter
    rl_enabled: bool = True
    rl_rate: int = 100
    rl_burst: int = 150
    rl_algorithm: str = "token_bucket"

    # Retry Policy
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0

    # Bulkhead
    bh_enabled: bool = False  # 默认关闭，避免过度资源隔离
    bh_max_concurrent: int = 10
    bh_timeout: float = 30.0

    # Metrics
    metrics_enabled: bool = True

    # Service identification
    service_name: str = "unknown"
    endpoint_name: str = "unknown"


class ResilienceManager:
    """韧性管理器 - 管理所有韧性组件实例"""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.bulkheads: Dict[str, SemaphoreBulkhead] = {}

    def get_or_create_circuit_breaker(
        self,
        key: str,
        config: ResilienceConfig
    ) -> CircuitBreaker:
        """获取或创建熔断器"""
        if key not in self.circuit_breakers:
            cb = CircuitBreaker(
                failure_threshold=config.cb_failure_threshold,
                recovery_timeout=config.cb_recovery_timeout,
                half_open_max_calls=config.cb_half_open_calls
            )
            self.circuit_breakers[key] = cb

            # 注册到健康收集器
            resilience_collector.register_circuit_breaker(key, cb)

        return self.circuit_breakers[key]

    def get_or_create_rate_limiter(
        self,
        key: str,
        config: ResilienceConfig
    ) -> TokenBucketRateLimiter:
        """获取或创建限流器"""
        if key not in self.rate_limiters:
            rl = TokenBucketRateLimiter(
                rate=config.rl_rate,
                burst=config.rl_burst
            )
            self.rate_limiters[key] = rl

            # 注册到健康收集器
            resilience_collector.register_rate_limiter(key, rl)

        return self.rate_limiters[key]

    def get_or_create_retry_policy(
        self,
        key: str,
        config: ResilienceConfig
    ) -> RetryPolicy:
        """获取或创建重试策略"""
        if key not in self.retry_policies:
            self.retry_policies[key] = ExponentialBackoff(
                initial_delay=config.retry_initial_delay,
                max_delay=config.retry_max_delay,
                multiplier=2.0
            )
        return self.retry_policies[key]

    def get_or_create_bulkhead(
        self,
        key: str,
        config: ResilienceConfig
    ) -> SemaphoreBulkhead:
        """获取或创建隔离舱"""
        if key not in self.bulkheads:
            self.bulkheads[key] = SemaphoreBulkhead(
                max_concurrent_calls=config.bh_max_concurrent,
                timeout=config.bh_timeout
            )
        return self.bulkheads[key]

    def update_metrics(self, config: ResilienceConfig):
        """更新 Prometheus 指标"""
        if not config.metrics_enabled:
            return

        labels = {
            'service': config.service_name,
            'endpoint': config.endpoint_name
        }

        # 更新熔断器状态
        key = f"{config.service_name}:{config.endpoint_name}"
        if key in self.circuit_breakers:
            cb = self.circuit_breakers[key]
            state_value = {
                CircuitState.CLOSED: 0,
                CircuitState.OPEN: 1,
                CircuitState.HALF_OPEN: 2
            }.get(cb.state, 0)
            resilience_circuit_state.labels(**labels).set(state_value)

        # 更新限流器令牌
        if key in self.rate_limiters:
            rl = self.rate_limiters[key]
            resilience_rate_limit_tokens.labels(**labels).set(rl.tokens)


# 全局管理器实例
_resilience_manager = ResilienceManager()


def with_resilience(
    config: Optional[ResilienceConfig] = None,
    **kwargs
) -> Callable:
    """
    主韧性装饰器 - 应用所有启用的韧性组件

    使用示例:
    @with_resilience(
        service_name="ocr",
        endpoint_name="process",
        cb_failure_threshold=3
    )
    def process_ocr(image):
        return ocr_provider.process(image)
    """
    if config is None:
        config = ResilienceConfig(**kwargs)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        key = f"{config.service_name}:{config.endpoint_name}"

        # 获取或创建组件
        circuit_breaker = None
        rate_limiter = None
        retry_policy = None
        bulkhead = None

        if config.cb_enabled:
            circuit_breaker = _resilience_manager.get_or_create_circuit_breaker(key, config)

        if config.rl_enabled:
            rate_limiter = _resilience_manager.get_or_create_rate_limiter(key, config)

        if config.retry_enabled:
            retry_policy = _resilience_manager.get_or_create_retry_policy(key, config)

        if config.bh_enabled:
            bulkhead = _resilience_manager.get_or_create_bulkhead(key, config)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            status = "success"

            try:
                # 1. Rate Limiting
                if rate_limiter and not rate_limiter.acquire():
                    resilience_rate_limit_rejected.labels(
                        service=config.service_name,
                        endpoint=config.endpoint_name
                    ).inc()
                    raise Exception("Rate limit exceeded")

                # 2. Bulkhead (如果启用)
                if bulkhead:
                    with bulkhead:
                        return _execute_with_circuit_breaker_and_retry(
                            func, args, kwargs,
                            circuit_breaker, retry_policy, config
                        )
                else:
                    return _execute_with_circuit_breaker_and_retry(
                        func, args, kwargs,
                        circuit_breaker, retry_policy, config
                    )

            except Exception as e:
                status = "error"
                logger.error(f"Resilience wrapped call failed: {e}")
                raise
            finally:
                # 记录持续时间
                duration = time.time() - start_time
                if config.metrics_enabled:
                    resilience_request_duration.labels(
                        service=config.service_name,
                        endpoint=config.endpoint_name,
                        status=status
                    ).observe(duration)

                # 更新指标
                _resilience_manager.update_metrics(config)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            status = "success"

            try:
                # 异步版本的实现
                if rate_limiter and not rate_limiter.acquire():
                    resilience_rate_limit_rejected.labels(
                        service=config.service_name,
                        endpoint=config.endpoint_name
                    ).inc()
                    raise Exception("Rate limit exceeded")

                # 异步执行
                return await _execute_async_with_circuit_breaker_and_retry(
                    func, args, kwargs,
                    circuit_breaker, retry_policy, config
                )

            except Exception as e:
                status = "error"
                logger.error(f"Async resilience call failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                if config.metrics_enabled:
                    resilience_request_duration.labels(
                        service=config.service_name,
                        endpoint=config.endpoint_name,
                        status=status
                    ).observe(duration)

                _resilience_manager.update_metrics(config)

        # 返回合适的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _execute_with_circuit_breaker_and_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    circuit_breaker: Optional[CircuitBreaker],
    retry_policy: Optional[RetryPolicy],
    config: ResilienceConfig
) -> Any:
    """执行带熔断器和重试的函数"""

    def execute():
        if circuit_breaker:
            return circuit_breaker.call(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    if retry_policy and config.retry_enabled:
        attempt = 0
        last_exception = None

        for attempt in range(config.retry_max_attempts):
            try:
                result = execute()

                # 成功后记录
                if attempt > 0:
                    resilience_retry_attempts.labels(
                        service=config.service_name,
                        endpoint=config.endpoint_name
                    ).inc(attempt)

                return result

            except Exception as e:
                last_exception = e

                if attempt < config.retry_max_attempts - 1:
                    delay = retry_policy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise

        raise last_exception
    else:
        return execute()


async def _execute_async_with_circuit_breaker_and_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    circuit_breaker: Optional[CircuitBreaker],
    retry_policy: Optional[RetryPolicy],
    config: ResilienceConfig
) -> Any:
    """异步执行带熔断器和重试的函数"""

    async def execute():
        if circuit_breaker:
            # 需要异步版本的熔断器
            return await func(*args, **kwargs)
        else:
            return await func(*args, **kwargs)

    if retry_policy and config.retry_enabled:
        attempt = 0
        last_exception = None

        for attempt in range(config.retry_max_attempts):
            try:
                result = await execute()

                if attempt > 0:
                    resilience_retry_attempts.labels(
                        service=config.service_name,
                        endpoint=config.endpoint_name
                    ).inc(attempt)

                return result

            except Exception as e:
                last_exception = e

                if attempt < config.retry_max_attempts - 1:
                    delay = retry_policy.get_delay(attempt)
                    logger.warning(
                        f"Async attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_exception
    else:
        return await execute()


# 便捷装饰器
def resilient_ocr(func: Callable) -> Callable:
    """OCR 服务专用韧性装饰器"""
    return with_resilience(
        service_name="ocr",
        endpoint_name=func.__name__,
        cb_failure_threshold=3,
        rl_rate=50,
        retry_max_attempts=2
    )(func)


def resilient_vision(func: Callable) -> Callable:
    """Vision 服务专用韧性装饰器"""
    return with_resilience(
        service_name="vision",
        endpoint_name=func.__name__,
        cb_failure_threshold=5,
        rl_rate=100,
        retry_max_attempts=3
    )(func)


def resilient_api(func: Callable) -> Callable:
    """API 端点专用韧性装饰器"""
    return with_resilience(
        service_name="api",
        endpoint_name=func.__name__,
        cb_failure_threshold=10,
        rl_rate=200,
        retry_enabled=False  # API 层不重试
    )(func)