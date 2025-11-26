"""
Resilience Metrics Collection
弹性指标收集和监控
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """指标摘要"""
    name: str
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class ResilienceMetrics:
    """
    弹性指标收集器

    功能:
    - 收集各弹性组件的指标
    - 计算统计摘要
    - 导出 Prometheus 格式
    - 时间窗口聚合
    """

    def __init__(self, window_size: int = 300):  # 默认5分钟窗口
        """
        初始化指标收集器

        Args:
            window_size: 时间窗口大小（秒）
        """
        self.window_size = window_size
        self._lock = threading.Lock()

        # 指标存储
        self.circuit_breaker_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.rate_limiter_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.retry_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.bulkhead_metrics = defaultdict(lambda: deque(maxlen=1000))

        # 计数器
        self.counters = defaultdict(int)

        # 直方图数据
        self.histograms = defaultdict(list)

    def record_circuit_breaker_event(self, event: Dict[str, Any]):
        """记录熔断器事件"""
        with self._lock:
            metric_name = f"circuit_breaker_{event['event']}_total"
            labels = {
                "name": event["circuit_breaker"],
                "state": event["state"]
            }

            # 增加计数器
            counter_key = (metric_name, tuple(labels.items()))
            self.counters[counter_key] += 1

            # 记录事件
            self.circuit_breaker_metrics[event["circuit_breaker"]].append({
                "timestamp": datetime.now(),
                "event": event["event"],
                "state": event["state"],
                "duration": event.get("duration", 0),
                "detail": event.get("detail", "")
            })

            # 记录延迟
            if event.get("duration", 0) > 0:
                self.histograms["circuit_breaker_duration_seconds"].append(event["duration"])

    def record_rate_limiter_event(self, event: Dict[str, Any]):
        """记录限流器事件"""
        with self._lock:
            metric_name = f"rate_limiter_{event['event']}_total"
            labels = {
                "name": event["rate_limiter"],
                "identifier": event.get("identifier", "default")
            }

            # 增加计数器
            counter_key = (metric_name, tuple(labels.items()))
            self.counters[counter_key] += 1

            # 记录事件
            self.rate_limiter_metrics[event["rate_limiter"]].append({
                "timestamp": datetime.now(),
                "event": event["event"],
                "identifier": event.get("identifier", "default")
            })

    def record_retry_event(self, event: Dict[str, Any]):
        """记录重试事件"""
        with self._lock:
            metric_name = f"retry_{event['event']}_total"
            labels = {
                "name": event["retry_policy"],
                "attempt": str(event.get("attempt", 0))
            }

            # 增加计数器
            counter_key = (metric_name, tuple(labels.items()))
            self.counters[counter_key] += 1

            # 记录事件
            self.retry_metrics[event["retry_policy"]].append({
                "timestamp": datetime.now(),
                "event": event["event"],
                "attempt": event.get("attempt", 0),
                "delay": event.get("delay", 0)
            })

            # 记录延迟
            if event.get("delay", 0) > 0:
                self.histograms["retry_delay_seconds"].append(event["delay"])

    def record_bulkhead_event(self, event: Dict[str, Any]):
        """记录隔板事件"""
        with self._lock:
            metric_name = f"bulkhead_{event['event']}_total"
            labels = {
                "name": event["bulkhead"],
                "active": str(event.get("active_calls", 0))
            }

            # 增加计数器
            counter_key = (metric_name, tuple(labels.items()))
            self.counters[counter_key] += 1

            # 记录事件
            self.bulkhead_metrics[event["bulkhead"]].append({
                "timestamp": datetime.now(),
                "event": event["event"],
                "active_calls": event.get("active_calls", 0),
                "duration": event.get("duration", 0)
            })

            # 记录执行时间
            if event.get("duration", 0) > 0:
                self.histograms["bulkhead_execution_seconds"].append(event["duration"])

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_size)

            summary = {
                "timestamp": now.isoformat(),
                "window_size": self.window_size,
                "circuit_breakers": self._summarize_circuit_breakers(window_start),
                "rate_limiters": self._summarize_rate_limiters(window_start),
                "retry_policies": self._summarize_retry_policies(window_start),
                "bulkheads": self._summarize_bulkheads(window_start),
                "counters": self._get_counter_summary(),
                "histograms": self._get_histogram_summary()
            }

            return summary

    def _summarize_circuit_breakers(self, window_start: datetime) -> Dict[str, Any]:
        """汇总熔断器指标"""
        summary = {}
        for name, events in self.circuit_breaker_metrics.items():
            recent_events = [e for e in events if e["timestamp"] >= window_start]
            if recent_events:
                summary[name] = {
                    "total_events": len(recent_events),
                    "state_changes": sum(1 for e in recent_events if e["event"] == "state_change"),
                    "rejections": sum(1 for e in recent_events if e["event"] == "rejected"),
                    "successes": sum(1 for e in recent_events if e["event"] == "success"),
                    "failures": sum(1 for e in recent_events if e["event"] == "failure"),
                    "current_state": recent_events[-1]["state"] if recent_events else "unknown"
                }
        return summary

    def _summarize_rate_limiters(self, window_start: datetime) -> Dict[str, Any]:
        """汇总限流器指标"""
        summary = {}
        for name, events in self.rate_limiter_metrics.items():
            recent_events = [e for e in events if e["timestamp"] >= window_start]
            if recent_events:
                allowed = sum(1 for e in recent_events if e["event"] == "allowed")
                rejected = sum(1 for e in recent_events if e["event"] == "rejected")
                total = allowed + rejected
                summary[name] = {
                    "total_requests": total,
                    "allowed": allowed,
                    "rejected": rejected,
                    "rejection_rate": rejected / total if total > 0 else 0
                }
        return summary

    def _summarize_retry_policies(self, window_start: datetime) -> Dict[str, Any]:
        """汇总重试策略指标"""
        summary = {}
        for name, events in self.retry_metrics.items():
            recent_events = [e for e in events if e["timestamp"] >= window_start]
            if recent_events:
                retries = [e for e in recent_events if e["event"] == "retry"]
                delays = [e["delay"] for e in retries if e.get("delay", 0) > 0]
                summary[name] = {
                    "total_attempts": len(recent_events),
                    "retries": len(retries),
                    "successes": sum(1 for e in recent_events if e["event"] == "success"),
                    "exhausted": sum(1 for e in recent_events if e["event"] == "exhausted"),
                    "avg_delay": sum(delays) / len(delays) if delays else 0,
                    "total_delay": sum(delays)
                }
        return summary

    def _summarize_bulkheads(self, window_start: datetime) -> Dict[str, Any]:
        """汇总隔板指标"""
        summary = {}
        for name, events in self.bulkhead_metrics.items():
            recent_events = [e for e in events if e["timestamp"] >= window_start]
            if recent_events:
                durations = [e["duration"] for e in recent_events if e.get("duration", 0) > 0]
                active_calls = [e["active_calls"] for e in recent_events if "active_calls" in e]
                summary[name] = {
                    "total_calls": len(recent_events),
                    "successes": sum(1 for e in recent_events if e["event"] == "success"),
                    "rejections": sum(1 for e in recent_events if e["event"] == "rejected"),
                    "timeouts": sum(1 for e in recent_events if e["event"] == "timeout"),
                    "avg_duration": sum(durations) / len(durations) if durations else 0,
                    "avg_active_calls": sum(active_calls) / len(active_calls) if active_calls else 0,
                    "max_active_calls": max(active_calls) if active_calls else 0
                }
        return summary

    def _get_counter_summary(self) -> Dict[str, int]:
        """获取计数器摘要"""
        summary = {}
        for (metric_name, labels), count in self.counters.items():
            # Build label string safely without nested f-strings (Python 3.10+ compatible)
            try:
                label_str = ",".join([f"{k}=\"{v}\"" for k, v in labels])
            except Exception:
                label_str = ""
            key = f"{metric_name}{{{label_str}}}"
            summary[key] = count
        return summary

    def _get_histogram_summary(self) -> Dict[str, Dict[str, float]]:
        """获取直方图摘要"""
        summary = {}
        for name, values in self.histograms.items():
            if values:
                sorted_values = sorted(values)
                count = len(sorted_values)
                summary[name] = {
                    "count": count,
                    "sum": sum(sorted_values),
                    "min": sorted_values[0],
                    "max": sorted_values[-1],
                    "avg": sum(sorted_values) / count,
                    "p50": self._percentile(sorted_values, 0.50),
                    "p95": self._percentile(sorted_values, 0.95),
                    "p99": self._percentile(sorted_values, 0.99)
                }
        return summary

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not sorted_values:
            return 0
        index = int(len(sorted_values) * percentile)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]

    def export_prometheus(self) -> str:
        """导出 Prometheus 格式"""
        lines = []
        lines.append("# HELP resilience_metrics Resilience layer metrics")
        lines.append("# TYPE resilience_metrics gauge")

        with self._lock:
            # 导出计数器
            for (metric_name, labels), count in self.counters.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{metric_name}{{{label_str}}} {count}")

            # 导出直方图
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    count = len(sorted_values)

                    # 导出桶
                    for quantile in [0.5, 0.9, 0.95, 0.99, 1.0]:
                        value = self._percentile(sorted_values, quantile)
                        lines.append(f'{name}_bucket{{le="{quantile}"}} {value}')

                    lines.append(f"{name}_count {count}")
                    lines.append(f"{name}_sum {sum(sorted_values)}")

        return "\n".join(lines)

    def clear_old_metrics(self):
        """清理过期指标"""
        with self._lock:
            cutoff = datetime.now() - timedelta(seconds=self.window_size * 2)

            # 清理各组件的旧事件
            for metrics in [
                self.circuit_breaker_metrics,
                self.rate_limiter_metrics,
                self.retry_metrics,
                self.bulkhead_metrics
            ]:
                for name, events in metrics.items():
                    # deque 会自动限制大小，这里只是额外的清理
                    while events and events[0]["timestamp"] < cutoff:
                        events.popleft()

    def reset(self):
        """重置所有指标"""
        with self._lock:
            self.circuit_breaker_metrics.clear()
            self.rate_limiter_metrics.clear()
            self.retry_metrics.clear()
            self.bulkhead_metrics.clear()
            self.counters.clear()
            self.histograms.clear()
