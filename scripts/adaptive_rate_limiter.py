#!/usr/bin/env python3
"""
Adaptive Rate Limiter
自适应限流器，根据系统负载动态调整限流阈值
"""

import time
import threading
import logging
import json
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Deque
from pathlib import Path
import hashlib
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Decision(Enum):
    """限流决策"""
    ALLOW = "allow"
    THROTTLE = "throttle"
    REJECT = "reject"


class TrafficPattern(Enum):
    """流量模式"""
    NORMAL = "normal"  # 正常流量
    BURST = "burst"  # 突发流量
    SUSTAINED = "sustained"  # 持续高流量
    ATTACK = "attack"  # 疑似攻击
    DEGRADED = "degraded"  # 降级模式


@dataclass
class RateLimitConfig:
    """限流配置"""
    # 基础配置
    initial_rate: float = 1000.0  # 初始速率(req/s)
    burst_size: int = 100  # 突发容量
    window_size: int = 60  # 窗口大小(秒)

    # 自适应配置
    adaptation_enabled: bool = True  # 启用自适应
    adjustment_interval: int = 30  # 调整间隔(秒)
    max_adjustment_ratio: float = 0.2  # 最大调整比例

    # 阈值配置
    cpu_threshold_high: float = 80.0  # CPU高阈值(%)
    cpu_threshold_low: float = 50.0  # CPU低阈值(%)
    memory_threshold_high: float = 85.0  # 内存高阈值(%)
    latency_threshold_p95: float = 100.0  # P95延迟阈值(ms)
    error_rate_threshold: float = 0.01  # 错误率阈值(1%)

    # 算法选择
    algorithm: str = "token_bucket"  # 默认算法


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0  # CPU使用率(%)
    memory_usage: float = 0.0  # 内存使用率(%)
    request_rate: float = 0.0  # 请求速率(req/s)
    error_rate: float = 0.0  # 错误率
    latency_p50: float = 0.0  # P50延迟(ms)
    latency_p95: float = 0.0  # P95延迟(ms)
    latency_p99: float = 0.0  # P99延迟(ms)
    active_connections: int = 0  # 活跃连接数


@dataclass
class Request:
    """请求信息"""
    id: str
    user_id: str
    api_path: str
    method: str
    source_ip: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 优先级(0-10)
    cost: float = 1.0  # 请求成本权重


class TokenBucket:
    """令牌桶算法"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # 令牌生成速率
        self.capacity = capacity  # 桶容量
        self.tokens = capacity  # 当前令牌数
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        """消费令牌"""
        with self.lock:
            now = time.time()
            # 补充令牌
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            # 尝试消费
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def adjust_rate(self, new_rate: float):
        """调整速率"""
        with self.lock:
            self.rate = new_rate
            logger.info(f"Token bucket rate adjusted to {new_rate:.2f} req/s")


class LeakyBucket:
    """漏桶算法"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # 漏出速率
        self.capacity = capacity  # 桶容量
        self.volume = 0.0  # 当前水量
        self.last_leak = time.time()
        self.lock = threading.Lock()

    def add(self, amount: float = 1.0) -> bool:
        """添加请求"""
        with self.lock:
            now = time.time()
            # 漏水
            elapsed = now - self.last_leak
            self.volume = max(0, self.volume - elapsed * self.rate)
            self.last_leak = now

            # 尝试添加
            if self.volume + amount <= self.capacity:
                self.volume += amount
                return True
            return False

    def adjust_rate(self, new_rate: float):
        """调整速率"""
        with self.lock:
            self.rate = new_rate
            logger.info(f"Leaky bucket rate adjusted to {new_rate:.2f} req/s")


class SlidingWindow:
    """滑动窗口算法"""

    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # 窗口大小(秒)
        self.max_requests = max_requests  # 最大请求数
        self.requests: Deque[float] = deque()  # 请求时间戳
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """判断是否允许请求"""
        with self.lock:
            now = time.time()
            # 清理过期请求
            cutoff = now - self.window_size
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # 检查是否超限
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def adjust_limit(self, new_limit: int):
        """调整限制"""
        with self.lock:
            self.max_requests = new_limit
            logger.info(f"Sliding window limit adjusted to {new_limit} requests")


class AdaptiveWindow:
    """自适应窗口算法"""

    def __init__(self, initial_window: int, initial_limit: int):
        self.window_size = initial_window
        self.limit = initial_limit
        self.requests: Deque[Tuple[float, float]] = deque()  # (timestamp, cost)
        self.total_cost = 0.0
        self.lock = threading.Lock()

    def allow_request(self, cost: float = 1.0) -> bool:
        """判断是否允许请求"""
        with self.lock:
            now = time.time()
            # 动态调整窗口
            cutoff = now - self.window_size

            # 清理过期请求
            while self.requests and self.requests[0][0] < cutoff:
                _, old_cost = self.requests.popleft()
                self.total_cost -= old_cost

            # 检查成本限制
            if self.total_cost + cost <= self.limit:
                self.requests.append((now, cost))
                self.total_cost += cost
                return True
            return False

    def adapt_window(self, metrics: SystemMetrics):
        """根据系统指标调整窗口"""
        with self.lock:
            # 基于延迟调整窗口大小
            if metrics.latency_p95 > 100:
                self.window_size = min(120, self.window_size * 1.1)
            elif metrics.latency_p95 < 50:
                self.window_size = max(10, self.window_size * 0.9)

            # 基于错误率调整限制
            if metrics.error_rate > 0.01:
                self.limit = self.limit * 0.9
            elif metrics.error_rate < 0.001:
                self.limit = self.limit * 1.1

            logger.debug(f"Adaptive window: size={self.window_size:.1f}s, limit={self.limit:.1f}")


class AdaptiveRateLimiter:
    """自适应限流器"""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.current_pattern = TrafficPattern.NORMAL

        # 初始化算法
        self.algorithms = {
            'token_bucket': TokenBucket(
                self.config.initial_rate,
                self.config.burst_size
            ),
            'leaky_bucket': LeakyBucket(
                self.config.initial_rate,
                self.config.burst_size
            ),
            'sliding_window': SlidingWindow(
                self.config.window_size,
                int(self.config.initial_rate * self.config.window_size)
            ),
            'adaptive_window': AdaptiveWindow(
                self.config.window_size,
                self.config.initial_rate * self.config.window_size
            )
        }

        # 统计信息
        self.stats = defaultdict(lambda: {
            'allowed': 0,
            'throttled': 0,
            'rejected': 0
        })

        # 历史指标
        self.metrics_history: Deque[SystemMetrics] = deque(maxlen=100)

        # 自适应线程
        self.adaptation_thread = None
        self.running = False

        if self.config.adaptation_enabled:
            self.start_adaptation()

    def should_allow_request(self, request: Request, metrics: Optional[SystemMetrics] = None) -> Decision:
        """
        判断是否允许请求

        Args:
            request: 请求信息
            metrics: 当前系统指标

        Returns:
            限流决策
        """
        # 更新指标历史
        if metrics:
            self.metrics_history.append(metrics)

        # 选择算法
        algorithm = self._select_algorithm(request, metrics)

        # 执行限流判断
        if algorithm == 'token_bucket':
            allowed = self.algorithms['token_bucket'].consume(request.cost)
        elif algorithm == 'leaky_bucket':
            allowed = self.algorithms['leaky_bucket'].add(request.cost)
        elif algorithm == 'sliding_window':
            allowed = self.algorithms['sliding_window'].allow_request()
        elif algorithm == 'adaptive_window':
            allowed = self.algorithms['adaptive_window'].allow_request(request.cost)
        else:
            allowed = True

        # 确定决策
        if allowed:
            decision = Decision.ALLOW
        elif self._should_throttle(request, metrics):
            decision = Decision.THROTTLE
        else:
            decision = Decision.REJECT

        # 更新统计
        self.stats[request.user_id][decision.value] += 1

        return decision

    def _select_algorithm(self, request: Request, metrics: Optional[SystemMetrics]) -> str:
        """选择合适的限流算法"""
        # 基于流量模式选择
        if self.current_pattern == TrafficPattern.BURST:
            return 'token_bucket'  # 令牌桶适合处理突发
        elif self.current_pattern == TrafficPattern.SUSTAINED:
            return 'leaky_bucket'  # 漏桶适合平滑流量
        elif self.current_pattern == TrafficPattern.ATTACK:
            return 'sliding_window'  # 滑动窗口精确限流
        else:
            return self.config.algorithm

    def _should_throttle(self, request: Request, metrics: Optional[SystemMetrics]) -> bool:
        """判断是否应该节流而非拒绝"""
        # 高优先级请求更可能被节流而非拒绝
        if request.priority >= 8:
            return True

        # 系统负载不太高时节流
        if metrics and metrics.cpu_usage < self.config.cpu_threshold_high:
            return True

        return False

    def adapt_thresholds(self, metrics: SystemMetrics):
        """
        根据系统指标自适应调整阈值

        Args:
            metrics: 系统指标
        """
        if not self.config.adaptation_enabled:
            return

        # 计算调整因子
        adjustment_factor = self._calculate_adjustment_factor(metrics)

        # 限制调整幅度
        adjustment_factor = max(
            1 - self.config.max_adjustment_ratio,
            min(1 + self.config.max_adjustment_ratio, adjustment_factor)
        )

        # 应用调整
        if abs(adjustment_factor - 1.0) > 0.01:  # 有意义的调整
            self._apply_adjustment(adjustment_factor)
            logger.info(f"Rate limit adjusted by factor {adjustment_factor:.2f}")

    def _calculate_adjustment_factor(self, metrics: SystemMetrics) -> float:
        """计算调整因子"""
        factor = 1.0

        # CPU影响
        if metrics.cpu_usage > self.config.cpu_threshold_high:
            factor *= (100 - metrics.cpu_usage) / (100 - self.config.cpu_threshold_high)
        elif metrics.cpu_usage < self.config.cpu_threshold_low:
            factor *= 1.1

        # 内存影响
        if metrics.memory_usage > self.config.memory_threshold_high:
            factor *= 0.9

        # 延迟影响
        if metrics.latency_p95 > self.config.latency_threshold_p95:
            factor *= self.config.latency_threshold_p95 / metrics.latency_p95

        # 错误率影响
        if metrics.error_rate > self.config.error_rate_threshold:
            factor *= 0.8

        return factor

    def _apply_adjustment(self, factor: float):
        """应用调整因子"""
        # 调整令牌桶
        if 'token_bucket' in self.algorithms:
            tb = self.algorithms['token_bucket']
            tb.adjust_rate(tb.rate * factor)

        # 调整漏桶
        if 'leaky_bucket' in self.algorithms:
            lb = self.algorithms['leaky_bucket']
            lb.adjust_rate(lb.rate * factor)

        # 调整滑动窗口
        if 'sliding_window' in self.algorithms:
            sw = self.algorithms['sliding_window']
            sw.adjust_limit(int(sw.max_requests * factor))

        # 调整自适应窗口
        if 'adaptive_window' in self.algorithms:
            aw = self.algorithms['adaptive_window']
            aw.limit *= factor

    def predict_traffic_pattern(self) -> TrafficPattern:
        """
        预测流量模式

        Returns:
            预测的流量模式
        """
        if len(self.metrics_history) < 5:
            return TrafficPattern.NORMAL

        recent_metrics = list(self.metrics_history)[-10:]

        # 计算请求率变化
        rates = [m.request_rate for m in recent_metrics]
        avg_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        rate_variance = sum((r - avg_rate) ** 2 for r in rates) / len(rates)

        # 计算错误率
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)

        # 判断模式
        if avg_error_rate > 0.05:
            pattern = TrafficPattern.ATTACK
        elif max_rate > avg_rate * 2 and rate_variance > avg_rate:
            pattern = TrafficPattern.BURST
        elif avg_rate > self.config.initial_rate * 0.8:
            pattern = TrafficPattern.SUSTAINED
        elif any(m.cpu_usage > 90 or m.memory_usage > 90 for m in recent_metrics):
            pattern = TrafficPattern.DEGRADED
        else:
            pattern = TrafficPattern.NORMAL

        if pattern != self.current_pattern:
            logger.info(f"Traffic pattern changed: {self.current_pattern.value} -> {pattern.value}")
            self.current_pattern = pattern

        return pattern

    def start_adaptation(self):
        """启动自适应线程"""
        if self.running:
            return

        self.running = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        logger.info("Adaptive rate limiting started")

    def stop_adaptation(self):
        """停止自适应线程"""
        self.running = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=5)
        logger.info("Adaptive rate limiting stopped")

    def _adaptation_loop(self):
        """自适应循环"""
        while self.running:
            try:
                # 预测流量模式
                self.predict_traffic_pattern()

                # 获取最新指标
                if self.metrics_history:
                    latest_metrics = self.metrics_history[-1]

                    # 自适应调整
                    self.adapt_thresholds(latest_metrics)

                    # 更新自适应窗口
                    if 'adaptive_window' in self.algorithms:
                        self.algorithms['adaptive_window'].adapt_window(latest_metrics)

                # 等待下次调整
                time.sleep(self.config.adjustment_interval)

            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                time.sleep(5)

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        if user_id:
            return dict(self.stats.get(user_id, {}))

        # 汇总所有用户
        total_stats = {
            'allowed': sum(s['allowed'] for s in self.stats.values()),
            'throttled': sum(s['throttled'] for s in self.stats.values()),
            'rejected': sum(s['rejected'] for s in self.stats.values())
        }

        total_requests = sum(total_stats.values())
        if total_requests > 0:
            total_stats['allow_rate'] = total_stats['allowed'] / total_requests
            total_stats['throttle_rate'] = total_stats['throttled'] / total_requests
            total_stats['reject_rate'] = total_stats['rejected'] / total_requests

        return total_stats

    def export_config(self) -> Dict[str, Any]:
        """导出当前配置"""
        config = asdict(self.config)

        # 添加当前状态
        config['current_pattern'] = self.current_pattern.value

        # 添加算法状态
        if 'token_bucket' in self.algorithms:
            tb = self.algorithms['token_bucket']
            config['token_bucket_rate'] = tb.rate
            config['token_bucket_tokens'] = tb.tokens

        if 'sliding_window' in self.algorithms:
            sw = self.algorithms['sliding_window']
            config['sliding_window_limit'] = sw.max_requests

        return config


def simulate_traffic(limiter: AdaptiveRateLimiter, duration: int = 60):
    """模拟流量用于测试"""
    start_time = time.time()
    request_count = 0

    while time.time() - start_time < duration:
        # 生成模拟请求
        request = Request(
            id=f"req_{request_count}",
            user_id=f"user_{random.randint(1, 100)}",
            api_path=f"/api/v1/{random.choice(['get', 'post', 'put', 'delete'])}",
            method=random.choice(['GET', 'POST', 'PUT', 'DELETE']),
            source_ip=f"192.168.1.{random.randint(1, 255)}",
            priority=random.randint(0, 10),
            cost=random.uniform(0.5, 2.0)
        )

        # 生成模拟指标
        elapsed = time.time() - start_time
        traffic_multiplier = 1 + 0.5 * math.sin(elapsed / 10)  # 周期性流量

        metrics = SystemMetrics(
            cpu_usage=min(95, 30 + 30 * traffic_multiplier + random.uniform(-5, 5)),
            memory_usage=min(90, 40 + 20 * traffic_multiplier + random.uniform(-5, 5)),
            request_rate=100 * traffic_multiplier + random.uniform(-10, 10),
            error_rate=max(0, 0.01 * traffic_multiplier + random.uniform(-0.005, 0.005)),
            latency_p50=30 + 20 * traffic_multiplier,
            latency_p95=50 + 50 * traffic_multiplier,
            latency_p99=100 + 100 * traffic_multiplier
        )

        # 执行限流判断
        decision = limiter.should_allow_request(request, metrics)

        request_count += 1

        # 模拟请求间隔
        time.sleep(random.uniform(0.001, 0.01))

    return request_count


def main():
    """主函数 - 用于测试"""
    import argparse

    parser = argparse.ArgumentParser(description='Adaptive Rate Limiter')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--simulate', type=int, help='Simulate traffic for N seconds')
    parser.add_argument('--export', help='Export current config to file')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = RateLimitConfig(**config_dict)
    else:
        config = RateLimitConfig()

    # 创建限流器
    limiter = AdaptiveRateLimiter(config)

    print("🚦 Adaptive Rate Limiter Started")
    print(f"   Algorithm: {config.algorithm}")
    print(f"   Initial Rate: {config.initial_rate} req/s")
    print(f"   Adaptation: {'Enabled' if config.adaptation_enabled else 'Disabled'}")

    # 模拟流量
    if args.simulate:
        print(f"\n📊 Simulating traffic for {args.simulate} seconds...")
        request_count = simulate_traffic(limiter, args.simulate)

        # 显示统计
        stats = limiter.get_stats()
        print(f"\n📈 Statistics:")
        print(f"   Total Requests: {request_count}")
        print(f"   Allowed: {stats['allowed']} ({stats.get('allow_rate', 0):.1%})")
        print(f"   Throttled: {stats['throttled']} ({stats.get('throttle_rate', 0):.1%})")
        print(f"   Rejected: {stats['rejected']} ({stats.get('reject_rate', 0):.1%})")
        print(f"   Current Pattern: {limiter.current_pattern.value}")

    # 导出配置
    if args.export:
        config_data = limiter.export_config()
        with open(args.export, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"\n✅ Config exported to {args.export}")

    # 停止自适应
    limiter.stop_adaptation()


if __name__ == "__main__":
    main()