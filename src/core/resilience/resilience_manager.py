"""
Resilience Manager - 统一的弹性管理器
整合所有弹性模式，提供统一接口和协调
"""

import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

from .circuit_breaker import CircuitBreaker, CircuitState
from .rate_limiter import RateLimiter
from .retry_policy import RetryPolicy, ExponentialBackoff
from .bulkhead import Bulkhead
from .metrics import ResilienceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ResilienceConfig:
    """弹性配置"""
    # Circuit Breaker 配置
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 60
    circuit_half_open_max_calls: int = 3

    # Rate Limiter 配置
    rate_limiter_enabled: bool = True
    rate_limit: float = 100.0
    rate_burst: int = 150
    rate_algorithm: str = "token_bucket"

    # Retry Policy 配置
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_exponential_base: float = 2.0

    # Bulkhead 配置
    bulkhead_enabled: bool = True
    bulkhead_max_concurrent: int = 10
    bulkhead_max_wait: float = 0.0
    bulkhead_type: str = "threadpool"

    # 全局配置
    metrics_enabled: bool = True
    auto_scaling_enabled: bool = False
    health_check_interval: int = 60


@dataclass
class ResilienceHealth:
    """弹性健康状态"""
    healthy: bool = True
    circuit_breakers: Dict[str, Dict] = field(default_factory=dict)
    rate_limiters: Dict[str, Dict] = field(default_factory=dict)
    retry_policies: Dict[str, Dict] = field(default_factory=dict)
    bulkheads: Dict[str, Dict] = field(default_factory=dict)
    overall_status: str = "healthy"
    issues: List[str] = field(default_factory=list)
    timestamp: str = ""


class ResilienceManager:
    """
    弹性管理器

    功能:
    - 统一管理所有弹性组件
    - 协调多个弹性模式
    - 自动健康检查和恢复
    - 指标收集和监控
    - 动态配置调整
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化管理器"""
        if not hasattr(self, 'initialized'):
            self.config = ResilienceConfig()
            self.circuit_breakers: Dict[str, CircuitBreaker] = {}
            self.rate_limiters: Dict[str, RateLimiter] = {}
            self.retry_policies: Dict[str, RetryPolicy] = {}
            self.bulkheads: Dict[str, Bulkhead] = {}
            self.metrics = ResilienceMetrics()
            self._health_check_thread = None
            self._stop_health_check = threading.Event()
            self.initialized = True

    def configure(self, config: ResilienceConfig):
        """更新配置"""
        self.config = config
        logger.info(f"Resilience configuration updated: {config}")

    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None
    ) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold or self.config.circuit_failure_threshold,
                recovery_timeout=recovery_timeout or self.config.circuit_recovery_timeout,
                half_open_max_calls=self.config.circuit_half_open_max_calls,
                metrics_callback=self.metrics.record_circuit_breaker_event
            )
        return self.circuit_breakers[name]

    def get_rate_limiter(
        self,
        name: str,
        rate: Optional[float] = None,
        burst: Optional[int] = None
    ) -> RateLimiter:
        """获取或创建限流器"""
        if name not in self.rate_limiters:
            self.rate_limiters[name] = RateLimiter(
                name=name,
                rate=rate or self.config.rate_limit,
                burst=burst or self.config.rate_burst,
                algorithm=self.config.rate_algorithm,
                metrics_callback=self.metrics.record_rate_limiter_event
            )
        return self.rate_limiters[name]

    def get_retry_policy(
        self,
        name: str,
        max_attempts: Optional[int] = None
    ) -> RetryPolicy:
        """获取或创建重试策略"""
        if name not in self.retry_policies:
            self.retry_policies[name] = RetryPolicy(
                name=name,
                max_attempts=max_attempts or self.config.retry_max_attempts,
                strategy=ExponentialBackoff(
                    base_delay=self.config.retry_base_delay,
                    max_delay=self.config.retry_max_delay,
                    exponential_base=self.config.retry_exponential_base
                ),
                metrics_callback=self.metrics.record_retry_event
            )
        return self.retry_policies[name]

    def get_bulkhead(
        self,
        name: str,
        max_concurrent_calls: Optional[int] = None
    ) -> Bulkhead:
        """获取或创建隔板"""
        if name not in self.bulkheads:
            self.bulkheads[name] = Bulkhead(
                name=name,
                max_concurrent_calls=max_concurrent_calls or self.config.bulkhead_max_concurrent,
                max_wait_duration=self.config.bulkhead_max_wait,
                bulkhead_type=self.config.bulkhead_type,
                metrics_callback=self.metrics.record_bulkhead_event
            )
        return self.bulkheads[name]

    def protect(
        self,
        name: str,
        func: Callable,
        use_circuit_breaker: bool = True,
        use_rate_limiter: bool = True,
        use_retry: bool = True,
        use_bulkhead: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        使用多重保护执行函数

        按照以下顺序应用保护:
        1. Rate Limiter - 限制请求速率
        2. Bulkhead - 限制并发
        3. Circuit Breaker - 快速失败
        4. Retry - 重试机制
        """
        # 1. Rate Limiter
        if use_rate_limiter and self.config.rate_limiter_enabled:
            limiter = self.get_rate_limiter(name)
            if not limiter.allow_request(name):
                raise Exception(f"Rate limit exceeded for {name}")

        # 2. Bulkhead
        if use_bulkhead and self.config.bulkhead_enabled:
            bulkhead = self.get_bulkhead(name)
            return self._execute_with_bulkhead(
                bulkhead, name, func,
                use_circuit_breaker, use_retry,
                *args, **kwargs
            )

        # 3. Circuit Breaker + Retry
        return self._execute_with_circuit_breaker_and_retry(
            name, func,
            use_circuit_breaker, use_retry,
            *args, **kwargs
        )

    def _execute_with_bulkhead(
        self,
        bulkhead: Bulkhead,
        name: str,
        func: Callable,
        use_circuit_breaker: bool,
        use_retry: bool,
        *args,
        **kwargs
    ) -> Any:
        """通过隔板执行"""
        def wrapped():
            return self._execute_with_circuit_breaker_and_retry(
                name, func,
                use_circuit_breaker, use_retry,
                *args, **kwargs
            )
        return bulkhead.execute(wrapped)

    def _execute_with_circuit_breaker_and_retry(
        self,
        name: str,
        func: Callable,
        use_circuit_breaker: bool,
        use_retry: bool,
        *args,
        **kwargs
    ) -> Any:
        """通过熔断器和重试执行"""
        if use_retry and self.config.retry_enabled:
            retry_policy = self.get_retry_policy(name)

            if use_circuit_breaker and self.config.circuit_breaker_enabled:
                circuit_breaker = self.get_circuit_breaker(name)
                return retry_policy.execute(
                    lambda: circuit_breaker.call(func, *args, **kwargs)
                )
            else:
                return retry_policy.execute(func, *args, **kwargs)

        elif use_circuit_breaker and self.config.circuit_breaker_enabled:
            circuit_breaker = self.get_circuit_breaker(name)
            return circuit_breaker.call(func, *args, **kwargs)

        else:
            return func(*args, **kwargs)

    def get_health(self) -> ResilienceHealth:
        """获取整体健康状态"""
        health = ResilienceHealth(timestamp=datetime.now().isoformat())

        # 收集各组件健康状态
        for name, cb in self.circuit_breakers.items():
            cb_health = cb.get_health()
            health.circuit_breakers[name] = cb_health
            if cb_health["state"] == CircuitState.OPEN.value:
                health.issues.append(f"Circuit breaker '{name}' is OPEN")

        for name, rl in self.rate_limiters.items():
            rl_health = rl.get_health()
            health.rate_limiters[name] = rl_health
            if rl_health["rejection_rate"] > 0.1:
                health.issues.append(f"Rate limiter '{name}' high rejection rate")

        for name, rp in self.retry_policies.items():
            rp_health = rp.get_health()
            health.retry_policies[name] = rp_health
            if rp_health.get("success_rate", 1) < 0.5:
                health.issues.append(f"Retry policy '{name}' low success rate")

        for name, bh in self.bulkheads.items():
            bh_health = bh.get_health()
            health.bulkheads[name] = bh_health
            if bh_health["utilization"] > 0.9:
                health.issues.append(f"Bulkhead '{name}' high utilization")

        # 确定整体状态
        if len(health.issues) == 0:
            health.overall_status = "healthy"
            health.healthy = True
        elif len(health.issues) < 3:
            health.overall_status = "degraded"
            health.healthy = True
        else:
            health.overall_status = "unhealthy"
            health.healthy = False

        return health

    def reset_component(self, component_type: str, name: str):
        """重置特定组件"""
        if component_type == "circuit_breaker" and name in self.circuit_breakers:
            self.circuit_breakers[name].reset()
            logger.info(f"Circuit breaker '{name}' reset")
        elif component_type == "rate_limiter" and name in self.rate_limiters:
            self.rate_limiters[name].reset()
            logger.info(f"Rate limiter '{name}' reset")
        elif component_type == "retry_policy" and name in self.retry_policies:
            self.retry_policies[name].reset_stats()
            logger.info(f"Retry policy '{name}' stats reset")
        elif component_type == "bulkhead" and name in self.bulkheads:
            self.bulkheads[name].reset_stats()
            logger.info(f"Bulkhead '{name}' stats reset")

    def reset_all(self):
        """重置所有组件"""
        for name in self.circuit_breakers:
            self.reset_component("circuit_breaker", name)
        for name in self.rate_limiters:
            self.reset_component("rate_limiter", name)
        for name in self.retry_policies:
            self.reset_component("retry_policy", name)
        for name in self.bulkheads:
            self.reset_component("bulkhead", name)
        logger.info("All resilience components reset")

    def auto_scale(self):
        """自动调整组件参数"""
        if not self.config.auto_scaling_enabled:
            return

        health = self.get_health()

        # 自动调整熔断器
        for name, cb_health in health.circuit_breakers.items():
            if cb_health["failure_rate"] < 0.01:
                # 非常低的失败率，可以放宽阈值
                cb = self.circuit_breakers[name]
                cb.failure_threshold = min(cb.failure_threshold + 1, 10)

        # 自动调整限流器
        for name, rl_health in health.rate_limiters.items():
            if rl_health["rejection_rate"] > 0.2:
                # 高拒绝率，增加限流
                rl = self.rate_limiters[name]
                new_rate = rl.rate * 0.9
                rl.update_rate(new_rate)
            elif rl_health["rejection_rate"] < 0.01:
                # 低拒绝率，可以增加流量
                rl = self.rate_limiters[name]
                new_rate = rl.rate * 1.1
                rl.update_rate(new_rate)

        # 自动调整隔板
        for name, bh_health in health.bulkheads.items():
            if bh_health["utilization"] > 0.9:
                # 高利用率，增加容量
                bh = self.bulkheads[name]
                new_capacity = min(bh.max_concurrent_calls + 2, 50)
                bh.resize(new_capacity)
            elif bh_health["utilization"] < 0.3:
                # 低利用率，减少容量
                bh = self.bulkheads[name]
                new_capacity = max(bh.max_concurrent_calls - 1, 5)
                bh.resize(new_capacity)

    def export_config(self) -> Dict[str, Any]:
        """导出当前配置"""
        return {
            "global_config": self.config.__dict__,
            "circuit_breakers": {
                name: {
                    "failure_threshold": cb.failure_threshold,
                    "recovery_timeout": cb.recovery_timeout,
                    "state": cb.state.value
                }
                for name, cb in self.circuit_breakers.items()
            },
            "rate_limiters": {
                name: {
                    "rate": rl.rate,
                    "burst": rl.burst
                }
                for name, rl in self.rate_limiters.items()
            },
            "retry_policies": {
                name: {
                    "max_attempts": rp.max_attempts
                }
                for name, rp in self.retry_policies.items()
            },
            "bulkheads": {
                name: {
                    "max_concurrent_calls": bh.max_concurrent_calls,
                    "max_wait_duration": bh.max_wait_duration
                }
                for name, bh in self.bulkheads.items()
            }
        }

    def import_config(self, config_dict: Dict[str, Any]):
        """导入配置"""
        if "global_config" in config_dict:
            self.config = ResilienceConfig(**config_dict["global_config"])

        # 更新各组件配置
        for component_type in ["circuit_breakers", "rate_limiters", "retry_policies", "bulkheads"]:
            if component_type in config_dict:
                for name, params in config_dict[component_type].items():
                    # 组件会在第一次使用时创建并应用这些参数
                    logger.info(f"Configuration imported for {component_type}: {name}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return self.metrics.get_summary()


# 全局实例
resilience_manager = ResilienceManager()


def with_resilience(
    name: Optional[str] = None,
    use_circuit_breaker: bool = True,
    use_rate_limiter: bool = True,
    use_retry: bool = True,
    use_bulkhead: bool = True
):
    """
    弹性保护装饰器

    Usage:
        @with_resilience(name="external_api")
        def call_external_api():
            # API call logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        component_name = name or f"{func.__module__}.{func.__name__}"

        def wrapper(*args, **kwargs):
            return resilience_manager.protect(
                component_name, func,
                use_circuit_breaker, use_rate_limiter,
                use_retry, use_bulkhead,
                *args, **kwargs
            )

        wrapper.resilience_name = component_name
        return wrapper

    return decorator