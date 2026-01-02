"""
Health Check Resilience Extension
健康检查韧性扩展 - 暴露 Resilience 组件状态
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class CircuitState(str, Enum):
    """熔断器状态"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerStatus:
    """熔断器状态信息"""

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    failure_threshold: int
    last_failure_time: Optional[datetime] = None
    recovery_timeout: int = 60
    half_open_max_calls: int = 3


@dataclass
class RateLimiterStatus:
    """限流器状态信息"""

    name: str
    current_tokens: float
    max_tokens: float
    refill_rate: float
    algorithm: str = "token_bucket"
    window_size: int = 60
    last_refill: Optional[datetime] = None
    requests_accepted: int = 0
    requests_rejected: int = 0


@dataclass
class RetryPolicyStatus:
    """重试策略状态"""

    name: str
    max_attempts: int
    current_attempt: int = 0
    strategy: str = "exponential"
    base_delay: float = 1.0
    total_retries: int = 0
    successful_retries: int = 0


@dataclass
class BulkheadStatus:
    """隔离舱状态"""

    name: str
    max_concurrent: int
    current_concurrent: int
    queued_calls: int
    rejected_calls: int
    pool_type: str = "thread"


@dataclass
class AdaptiveStatus:
    """自适应策略状态"""

    enabled: bool
    current_rate_multiplier: float
    target_error_rate: float
    actual_error_rate: float
    last_adjustment: Optional[datetime] = None
    adjustment_count: int = 0


class ResilienceHealthCollector:
    """韧性层健康状态收集器"""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerStatus] = {}
        self.rate_limiters: Dict[str, RateLimiterStatus] = {}
        self.retry_policies: Dict[str, RetryPolicyStatus] = {}
        self.bulkheads: Dict[str, BulkheadStatus] = {}
        self.adaptive_status = AdaptiveStatus(
            enabled=False,
            current_rate_multiplier=1.0,
            target_error_rate=0.01,
            actual_error_rate=0.0,
        )
        self.adaptive_rate_limiters: Dict[str, Any] = {}

    def register_circuit_breaker(self, name: str, breaker: Any) -> None:
        """注册熔断器"""
        # 从实际熔断器对象提取状态
        status = CircuitBreakerStatus(
            name=name,
            state=getattr(breaker, "state", CircuitState.CLOSED),
            failure_count=getattr(breaker, "failure_count", 0),
            success_count=getattr(breaker, "success_count", 0),
            failure_threshold=getattr(breaker, "failure_threshold", 5),
            last_failure_time=getattr(breaker, "last_failure_time", None),
            recovery_timeout=getattr(breaker, "recovery_timeout", 60),
        )
        self.circuit_breakers[name] = status

    def register_rate_limiter(self, name: str, limiter: Any) -> None:
        """注册限流器"""
        status = RateLimiterStatus(
            name=name,
            current_tokens=getattr(limiter, "tokens", 0),
            max_tokens=getattr(limiter, "capacity", 100),
            refill_rate=getattr(limiter, "rate", 10),
            algorithm=getattr(limiter, "algorithm", "token_bucket"),
        )
        self.rate_limiters[name] = status

    def update_adaptive_status(
        self, enabled: bool, rate_multiplier: float, error_rate: float
    ) -> None:
        """更新自适应状态"""
        self.adaptive_status.enabled = enabled
        self.adaptive_status.current_rate_multiplier = rate_multiplier
        self.adaptive_status.actual_error_rate = error_rate
        self.adaptive_status.last_adjustment = datetime.now()
        self.adaptive_status.adjustment_count += 1

    def register_adaptive_rate_limiter(self, name: str, limiter: Any) -> None:
        """注册自适应限流器"""
        self.adaptive_rate_limiters[name] = limiter

    def get_health_status(self) -> Dict[str, Any]:
        """获取完整健康状态"""
        return {
            "resilience": {
                "status": self._calculate_overall_status(),
                "circuit_breakers": self._format_circuit_breakers(),
                "rate_limiters": self._format_rate_limiters(),
                "retry_policies": self._format_retry_policies(),
                "bulkheads": self._format_bulkheads(),
                "adaptive": self._format_adaptive_status(),
                "adaptive_rate_limit": self._format_adaptive_rate_limiters(),
                "metrics": self._collect_metrics(),
            }
        }

    def _calculate_overall_status(self) -> str:
        """计算整体状态"""
        # 检查是否有开路的熔断器
        open_circuits = sum(
            1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN
        )

        # 检查限流器状态
        exhausted_limiters = sum(
            1 for rl in self.rate_limiters.values() if rl.current_tokens < rl.max_tokens * 0.1
        )

        if open_circuits > 0:
            return "degraded"
        elif exhausted_limiters > 0:
            return "stressed"
        else:
            return "healthy"

    def _format_circuit_breakers(self) -> Dict[str, Any]:
        """格式化熔断器状态"""
        result = {}
        for name, cb in self.circuit_breakers.items():
            result[name] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "threshold": cb.failure_threshold,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None,
            }
        return result

    def _format_rate_limiters(self) -> Dict[str, Any]:
        """格式化限流器状态"""
        result = {}
        for name, rl in self.rate_limiters.items():
            result[name] = {
                "current_tokens": round(rl.current_tokens, 2),
                "max_tokens": rl.max_tokens,
                "refill_rate": rl.refill_rate,
                "algorithm": rl.algorithm,
                "utilization": round(1 - (rl.current_tokens / rl.max_tokens), 2),
            }
        return result

    def _format_retry_policies(self) -> Dict[str, Any]:
        """格式化重试策略状态"""
        result = {}
        for name, rp in self.retry_policies.items():
            result[name] = {
                "max_attempts": rp.max_attempts,
                "strategy": rp.strategy,
                "total_retries": rp.total_retries,
                "success_rate": (
                    round(rp.successful_retries / rp.total_retries, 2)
                    if rp.total_retries > 0
                    else 0
                ),
            }
        return result

    def _format_bulkheads(self) -> Dict[str, Any]:
        """格式化隔离舱状态"""
        result = {}
        for name, bh in self.bulkheads.items():
            result[name] = {
                "max_concurrent": bh.max_concurrent,
                "current_concurrent": bh.current_concurrent,
                "utilization": round(bh.current_concurrent / bh.max_concurrent, 2),
                "queued": bh.queued_calls,
                "rejected": bh.rejected_calls,
            }
        return result

    def _format_adaptive_status(self) -> Dict[str, Any]:
        """格式化自适应状态"""
        return {
            "enabled": self.adaptive_status.enabled,
            "rate_multiplier": round(self.adaptive_status.current_rate_multiplier, 2),
            "target_error_rate": self.adaptive_status.target_error_rate,
            "actual_error_rate": round(self.adaptive_status.actual_error_rate, 4),
            "last_adjustment": (
                self.adaptive_status.last_adjustment.isoformat()
                if self.adaptive_status.last_adjustment
                else None
            ),
            "adjustments_made": self.adaptive_status.adjustment_count,
        }

    def _format_adaptive_rate_limiters(self) -> Dict[str, Any]:
        """格式化自适应限流器状态"""
        result = {}
        for name, limiter in self.adaptive_rate_limiters.items():
            if hasattr(limiter, "get_status"):
                status = limiter.get_status()
                result[name] = {
                    "phase": status.get("phase", "unknown"),
                    "base_rate": status.get("base_rate", 0),
                    "current_rate": status.get("current_rate", 0),
                    "error_ema": status.get("error_ema", 0),
                    "tokens_available": status.get("tokens_available", 0),
                    "consecutive_failures": status.get("consecutive_failures", 0),
                    "in_cooldown": status.get("in_cooldown", False),
                    "recent_adjustments": status.get("recent_adjustments", []),
                }
        return result

    def _collect_metrics(self) -> Dict[str, Any]:
        """收集汇总指标"""
        total_circuits = len(self.circuit_breakers)
        open_circuits = sum(
            1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN
        )

        total_limiters = len(self.rate_limiters)
        avg_utilization = (
            sum(1 - (rl.current_tokens / rl.max_tokens) for rl in self.rate_limiters.values())
            / total_limiters
            if total_limiters > 0
            else 0
        )

        total_retries = sum(rp.total_retries for rp in self.retry_policies.values())

        # 统计自适应限流器状态
        adaptive_phases = {}
        for limiter in self.adaptive_rate_limiters.values():
            if hasattr(limiter, "get_status"):
                phase = limiter.get_status().get("phase", "unknown")
                adaptive_phases[phase] = adaptive_phases.get(phase, 0) + 1

        return {
            "circuit_breaker_open_ratio": (
                round(open_circuits / total_circuits, 2) if total_circuits > 0 else 0
            ),
            "rate_limiter_avg_utilization": round(avg_utilization, 2),
            "total_retries": total_retries,
            "bulkhead_rejections": sum(bh.rejected_calls for bh in self.bulkheads.values()),
            "adaptive_limiter_phases": adaptive_phases,
        }


resilience_collector = ResilienceHealthCollector()


def get_resilience_health() -> Dict[str, Any]:
    """获取韧性层健康状态"""
    return resilience_collector.get_health_status()


def update_resilience_metrics(component: str, **kwargs) -> None:
    """更新韧性指标"""
    if component == "circuit_breaker":
        # 更新熔断器指标
        pass
    elif component == "rate_limiter":
        # 更新限流器指标
        pass
    elif component == "adaptive":
        resilience_collector.update_adaptive_status(**kwargs)
