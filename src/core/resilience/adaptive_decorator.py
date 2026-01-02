"""
Adaptive Rate Limiting Decorator
自适应限流装饰器 - 简化集成到Vision/OCR管理器
"""

import asyncio
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from src.core.resilience.adaptive_rate_limiter import AdaptiveConfig, adaptive_manager

# 注意：避免在模块导入时引入 src.api 包以防循环依赖。
# 需要注册到健康收集器时，改为在运行时惰性导入。

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class AdaptiveDecoratorConfig:
    """自适应装饰器配置"""

    service_name: str
    endpoint_name: str

    # 从环境变量读取配置
    enabled: bool = bool(int(os.getenv("ADAPTIVE_RATE_LIMIT_ENABLED", "1")))
    base_rate: float = float(os.getenv("ADAPTIVE_BASE_RATE", "100.0"))
    error_threshold: float = float(os.getenv("ADAPTIVE_ERROR_THRESHOLD", "0.02"))
    recover_threshold: float = float(os.getenv("ADAPTIVE_RECOVER_THRESHOLD", "0.008"))
    latency_multiplier: float = float(os.getenv("ADAPTIVE_LATENCY_P95_THRESHOLD_MULTIPLIER", "1.3"))
    min_rate_ratio: float = float(os.getenv("ADAPTIVE_MIN_RATE_RATIO", "0.15"))
    adjust_interval_ms: int = int(os.getenv("ADAPTIVE_ADJUST_MIN_INTERVAL_MS", "2000"))
    recover_step: float = float(os.getenv("ADAPTIVE_RECOVER_STEP", "0.05"))
    error_alpha: float = float(os.getenv("ADAPTIVE_ERROR_ALPHA", "0.25"))
    max_failure_streak: int = int(os.getenv("ADAPTIVE_MAX_FAILURE_STREAK", "5"))
    max_adjustments_per_minute: int = int(os.getenv("ADAPTIVE_MAX_ADJUSTMENTS_PER_MINUTE", "20"))

    # 强制禁用开关
    force_disable: bool = bool(int(os.getenv("ADAPTIVE_FORCE_DISABLE", "0")))

    # 基线P95延迟（毫秒）
    baseline_p95_ms: float = float(os.getenv("ADAPTIVE_BASELINE_P95_MS", "1000.0"))


def adaptive_rate_limit(
    service: str = None,
    endpoint: str = None,
    config: Optional[AdaptiveDecoratorConfig] = None,
    **kwargs,
) -> Callable:
    """
    自适应限流装饰器

    使用示例:
    @adaptive_rate_limit(service="ocr", endpoint="process")
    def process_ocr(image):
        return ocr_provider.process(image)

    @adaptive_rate_limit(service="vision", endpoint="analyze")
    async def analyze_vision(image):
        return await vision_provider.analyze(image)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 获取服务和端点名称
        service_name = service or func.__module__.split(".")[-1]
        endpoint_name = endpoint or func.__name__

        # 创建配置
        if config:
            decorator_config = config
        else:
            decorator_config = AdaptiveDecoratorConfig(
                service_name=service_name, endpoint_name=endpoint_name, **kwargs
            )

        # 检查是否强制禁用
        if decorator_config.force_disable:
            logger.warning(
                f"Adaptive rate limiting force disabled for {service_name}:{endpoint_name}"
            )
            return func

        # 若未启用则直接返回原函数（不包裹，不产生任何开销）
        if not decorator_config.enabled:
            logger.info(f"Adaptive limiter disabled for {service_name}:{endpoint_name}")
            return func

        # 创建自适应配置
        adaptive_config = AdaptiveConfig(
            enabled=decorator_config.enabled,
            base_rate=decorator_config.base_rate,
            error_threshold=decorator_config.error_threshold,
            recover_threshold=decorator_config.recover_threshold,
            latency_p95_threshold_multiplier=decorator_config.latency_multiplier,
            min_rate_ratio=decorator_config.min_rate_ratio,
            adjust_min_interval_ms=decorator_config.adjust_interval_ms,
            recover_step=decorator_config.recover_step,
            error_alpha=decorator_config.error_alpha,
            max_failure_streak=decorator_config.max_failure_streak,
            max_adjustments_per_minute=decorator_config.max_adjustments_per_minute,
        )

        # 获取或创建限流器
        limiter = adaptive_manager.get_or_create(service_name, endpoint_name, adaptive_config)

        # 设置基线
        limiter.set_baseline(decorator_config.baseline_p95_ms)

        # 惰性注册到健康收集器（避免导入时循环依赖）
        _registered = {"done": False}

        @functools.wraps(func)
        def sync_wrapper(*args, **wrapper_kwargs) -> T:
            start_time = time.time()

            # 首次调用时尝试注册健康信息
            if not _registered["done"]:
                try:
                    from src.api.health_resilience import resilience_collector  # local import

                    resilience_collector.register_adaptive_rate_limiter(
                        f"{service_name}:{endpoint_name}", limiter
                    )
                except Exception:
                    # 健康模块不可用时静默忽略
                    pass
                _registered["done"] = True

            # 检查限流
            if not limiter.acquire():
                limiter.record_error()
                raise Exception(f"Adaptive rate limit exceeded for {service_name}:{endpoint_name}")

            try:
                # 执行函数
                result = func(*args, **wrapper_kwargs)

                # 记录成功
                limiter.record_success()

                return result

            except Exception:
                # 记录错误
                limiter.record_error()
                raise

            finally:
                # 记录延迟
                latency_ms = (time.time() - start_time) * 1000
                limiter.record_latency(latency_ms)

                # 评估并调整（异步进行，不阻塞主流程）
                limiter.evaluate_and_adjust()

        @functools.wraps(func)
        async def async_wrapper(*args, **wrapper_kwargs) -> T:
            start_time = time.time()

            # 首次调用时尝试注册健康信息
            if not _registered["done"]:
                try:
                    from src.api.health_resilience import resilience_collector  # local import

                    resilience_collector.register_adaptive_rate_limiter(
                        f"{service_name}:{endpoint_name}", limiter
                    )
                except Exception:
                    pass
                _registered["done"] = True

            # 检查限流
            if not limiter.acquire():
                limiter.record_error()
                raise Exception(f"Adaptive rate limit exceeded for {service_name}:{endpoint_name}")

            try:
                # 执行异步函数
                result = await func(*args, **wrapper_kwargs)

                # 记录成功
                limiter.record_success()

                return result

            except Exception:
                # 记录错误
                limiter.record_error()
                raise

            finally:
                # 记录延迟
                latency_ms = (time.time() - start_time) * 1000
                limiter.record_latency(latency_ms)

                # 异步评估并调整
                asyncio.create_task(_async_evaluate(limiter))

        async def _async_evaluate(lim):
            """异步评估（不阻塞主流程）"""
            try:
                lim.evaluate_and_adjust()
            except Exception:
                logger.error("Error in adaptive evaluation")

        # 返回合适的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# 便捷装饰器预设
def adaptive_ocr(func: Callable) -> Callable:
    """OCR服务专用自适应限流"""
    return adaptive_rate_limit(
        service="ocr",
        endpoint=func.__name__,
        base_rate=50.0,
        error_threshold=0.03,
        baseline_p95_ms=2000.0,
    )(func)


def adaptive_vision(func: Callable) -> Callable:
    """Vision服务专用自适应限流"""
    return adaptive_rate_limit(
        service="vision",
        endpoint=func.__name__,
        base_rate=100.0,
        error_threshold=0.02,
        baseline_p95_ms=1500.0,
    )(func)


def adaptive_api(func: Callable) -> Callable:
    """API端点专用自适应限流"""
    return adaptive_rate_limit(
        service="api",
        endpoint=func.__name__,
        base_rate=200.0,
        error_threshold=0.01,
        baseline_p95_ms=500.0,
    )(func)


# 管理函数
def get_adaptive_status() -> dict:
    """获取所有自适应限流器状态"""
    return adaptive_manager.get_all_status()


def reset_adaptive_limiters():
    """重置所有自适应限流器"""
    for limiter in adaptive_manager.limiters.values():
        limiter.reset()
    logger.info("All adaptive rate limiters reset")


def evaluate_all_limiters():
    """手动触发所有限流器评估"""
    adaptive_manager.evaluate_all()
    logger.info("All adaptive rate limiters evaluated")
