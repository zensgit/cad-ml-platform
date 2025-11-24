#!/usr/bin/env python3
"""
Rate Limit Analyzer
分析流量模式，识别异常行为，生成用户画像
"""

import json
import logging
import statistics
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
from pathlib import Path
import math
import hashlib

# 尝试导入相关模块
try:
    from .adaptive_rate_limiter import Request, SystemMetrics, TrafficPattern
except ImportError:
    from adaptive_rate_limiter import Request, SystemMetrics, TrafficPattern  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """流量模式类型"""
    NORMAL = "normal"  # 正常业务
    SPIKE = "spike"  # 流量尖峰
    GRADUAL_INCREASE = "gradual_increase"  # 渐进增长
    PERIODIC = "periodic"  # 周期性模式
    RANDOM = "random"  # 随机模式
    CRAWLER = "crawler"  # 爬虫
    DDOS = "ddos"  # DDoS攻击
    BRUTE_FORCE = "brute_force"  # 暴力破解


class AnomalyType(Enum):
    """异常类型"""
    RATE_ANOMALY = "rate_anomaly"  # 速率异常
    PATTERN_ANOMALY = "pattern_anomaly"  # 模式异常
    VOLUME_ANOMALY = "volume_anomaly"  # 流量异常
    BEHAVIOR_ANOMALY = "behavior_anomaly"  # 行为异常
    SOURCE_ANOMALY = "source_anomaly"  # 来源异常


@dataclass
class TimeWindow:
    """时间窗口"""
    start: float
    end: float
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = self.end - self.start

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp <= self.end


@dataclass
class TrafficMetrics:
    """流量指标"""
    request_count: int = 0
    unique_users: int = 0
    unique_ips: int = 0
    avg_request_rate: float = 0.0
    peak_request_rate: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    bandwidth_usage: float = 0.0
    api_distribution: Dict[str, int] = field(default_factory=dict)
    method_distribution: Dict[str, int] = field(default_factory=dict)
    status_distribution: Dict[int, int] = field(default_factory=dict)


@dataclass
class TrafficAnalysis:
    """流量分析结果"""
    time_window: TimeWindow
    metrics: TrafficMetrics
    pattern_type: PatternType
    anomalies: List['Anomaly']
    confidence: float  # 分析置信度 0-1
    recommendations: List[str]


@dataclass
class Anomaly:
    """异常检测结果"""
    type: AnomalyType
    severity: str  # low, medium, high, critical
    timestamp: float
    description: str
    affected_users: List[str] = field(default_factory=list)
    affected_ips: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """用户行为画像"""
    user_id: str
    total_requests: int = 0
    avg_request_rate: float = 0.0  # 平均请求速率
    peak_request_rate: float = 0.0  # 峰值请求速率
    error_rate: float = 0.0  # 错误率
    avg_latency: float = 0.0  # 平均延迟
    api_usage: Dict[str, int] = field(default_factory=dict)  # API使用分布
    access_pattern: str = "normal"  # 访问模式
    reputation_score: float = 100.0  # 信誉分数 0-100
    is_suspicious: bool = False
    last_activity: float = 0.0
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())


class PatternDetector:
    """流量模式检测器"""

    def __init__(self):
        self.patterns = {
            PatternType.NORMAL: self._detect_normal,
            PatternType.SPIKE: self._detect_spike,
            PatternType.GRADUAL_INCREASE: self._detect_gradual_increase,
            PatternType.PERIODIC: self._detect_periodic,
            PatternType.CRAWLER: self._detect_crawler,
            PatternType.DDOS: self._detect_ddos,
            PatternType.BRUTE_FORCE: self._detect_brute_force
        }

    def detect(self, requests: List[Request], metrics: TrafficMetrics) -> PatternType:
        """检测流量模式"""
        if not requests:
            return PatternType.NORMAL

        # 计算时间序列
        timestamps = [r.timestamp for r in requests]
        timestamps.sort()

        # 检测各种模式
        scores = {}
        for pattern_type, detector in self.patterns.items():
            score = detector(requests, timestamps, metrics)
            scores[pattern_type] = score

        # 返回得分最高的模式
        return max(scores, key=scores.get)

    def _detect_normal(self, requests: List[Request], timestamps: List[float],
                      metrics: TrafficMetrics) -> float:
        """检测正常模式"""
        if not timestamps:
            return 1.0

        # 计算请求间隔
        intervals = [timestamps[i] - timestamps[i-1]
                    for i in range(1, len(timestamps))]

        if not intervals:
            return 1.0

        # 正常流量的特征：间隔相对均匀
        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0

        cv = statistics.stdev(intervals) / mean_interval if len(intervals) > 1 else 0
        return max(0, 1 - cv)

    def _detect_spike(self, requests: List[Request], timestamps: List[float],
                     metrics: TrafficMetrics) -> float:
        """检测流量尖峰"""
        if len(timestamps) < 10:
            return 0.0

        # 计算移动平均
        window_size = min(10, len(timestamps) // 3)
        rates = []

        for i in range(window_size, len(timestamps)):
            duration = timestamps[i] - timestamps[i - window_size]
            if duration > 0:
                rate = window_size / duration
                rates.append(rate)

        if not rates:
            return 0.0

        # 尖峰特征：存在显著高于平均的点
        mean_rate = statistics.mean(rates)
        max_rate = max(rates)

        if mean_rate == 0:
            return 0.0

        spike_factor = max_rate / mean_rate
        return min(1.0, (spike_factor - 1) / 5)  # 5倍以上算完全尖峰

    def _detect_gradual_increase(self, requests: List[Request], timestamps: List[float],
                                 metrics: TrafficMetrics) -> float:
        """检测渐进增长"""
        if len(timestamps) < 20:
            return 0.0

        # 将时间序列分成若干段，计算每段的速率
        segments = 5
        segment_size = len(timestamps) // segments
        segment_rates = []

        for i in range(segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(timestamps))

            if end_idx - start_idx > 1:
                duration = timestamps[end_idx - 1] - timestamps[start_idx]
                if duration > 0:
                    rate = (end_idx - start_idx) / duration
                    segment_rates.append(rate)

        if len(segment_rates) < 2:
            return 0.0

        # 渐进增长特征：速率单调递增
        increases = sum(1 for i in range(1, len(segment_rates))
                       if segment_rates[i] > segment_rates[i-1])
        return increases / (len(segment_rates) - 1)

    def _detect_periodic(self, requests: List[Request], timestamps: List[float],
                        metrics: TrafficMetrics) -> float:
        """检测周期性模式"""
        if len(timestamps) < 50:
            return 0.0

        # 计算自相关性
        intervals = [timestamps[i] - timestamps[i-1]
                    for i in range(1, len(timestamps))]

        if len(intervals) < 20:
            return 0.0

        # 简化的周期性检测：检查间隔的规律性
        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0

        # 计算间隔的分布
        interval_counts = Counter([round(i / mean_interval) for i in intervals])

        # 周期性特征：间隔分布集中
        total_counts = sum(interval_counts.values())
        max_count = max(interval_counts.values())

        return max_count / total_counts if total_counts > 0 else 0.0

    def _detect_crawler(self, requests: List[Request], timestamps: List[float],
                       metrics: TrafficMetrics) -> float:
        """检测爬虫行为"""
        if not requests:
            return 0.0

        # 爬虫特征
        crawler_score = 0.0

        # 特征1：访问大量不同的API路径
        unique_apis = len(set(r.api_path for r in requests))
        if unique_apis > len(requests) * 0.8:
            crawler_score += 0.3

        # 特征2：请求间隔非常均匀
        if len(timestamps) > 10:
            intervals = [timestamps[i] - timestamps[i-1]
                        for i in range(1, len(timestamps))]
            if intervals:
                mean_interval = statistics.mean(intervals)
                if mean_interval > 0 and len(intervals) > 1:
                    cv = statistics.stdev(intervals) / mean_interval
                    if cv < 0.1:  # 间隔非常均匀
                        crawler_score += 0.4

        # 特征3：单个IP或用户发起大量请求
        user_counts = Counter(r.user_id for r in requests)
        if user_counts:
            max_user_requests = max(user_counts.values())
            if max_user_requests > len(requests) * 0.5:
                crawler_score += 0.3

        return min(1.0, crawler_score)

    def _detect_ddos(self, requests: List[Request], timestamps: List[float],
                    metrics: TrafficMetrics) -> float:
        """检测DDoS攻击"""
        if not requests or not timestamps:
            return 0.0

        ddos_score = 0.0

        # 特征1：大量不同IP的请求
        unique_ips = len(set(r.source_ip for r in requests))
        if unique_ips > len(requests) * 0.7:
            ddos_score += 0.3

        # 特征2：请求速率异常高
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
        if duration > 0:
            rate = len(requests) / duration
            if rate > 1000:  # 假设正常速率上限是1000 req/s
                ddos_score += 0.4

        # 特征3：错误率高
        if metrics.error_rate > 0.3:
            ddos_score += 0.3

        return min(1.0, ddos_score)

    def _detect_brute_force(self, requests: List[Request], timestamps: List[float],
                           metrics: TrafficMetrics) -> float:
        """检测暴力破解"""
        if not requests:
            return 0.0

        brute_force_score = 0.0

        # 特征1：针对特定API的重复请求
        api_counts = Counter(r.api_path for r in requests)
        if api_counts:
            # 检查是否有认证相关的API被频繁调用
            auth_apis = [api for api in api_counts
                        if 'login' in api.lower() or 'auth' in api.lower()]
            if auth_apis:
                auth_requests = sum(api_counts[api] for api in auth_apis)
                if auth_requests > len(requests) * 0.5:
                    brute_force_score += 0.5

        # 特征2：高错误率
        if metrics.error_rate > 0.5:
            brute_force_score += 0.3

        # 特征3：来自少量IP的大量请求
        ip_counts = Counter(r.source_ip for r in requests)
        if ip_counts and len(ip_counts) <= 5:
            concentrated_requests = sum(ip_counts.values())
            if concentrated_requests > len(requests) * 0.8:
                brute_force_score += 0.2

        return min(1.0, brute_force_score)


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Z-score阈值

    def detect(self, current_metrics: TrafficMetrics,
              historical_metrics: List[TrafficMetrics]) -> List[Anomaly]:
        """检测异常"""
        anomalies = []

        if not historical_metrics:
            return anomalies

        # 检测速率异常
        rate_anomaly = self._detect_rate_anomaly(current_metrics, historical_metrics)
        if rate_anomaly:
            anomalies.append(rate_anomaly)

        # 检测错误率异常
        error_anomaly = self._detect_error_anomaly(current_metrics, historical_metrics)
        if error_anomaly:
            anomalies.append(error_anomaly)

        # 检测延迟异常
        latency_anomaly = self._detect_latency_anomaly(current_metrics, historical_metrics)
        if latency_anomaly:
            anomalies.append(latency_anomaly)

        # 检测流量分布异常
        distribution_anomaly = self._detect_distribution_anomaly(current_metrics, historical_metrics)
        if distribution_anomaly:
            anomalies.append(distribution_anomaly)

        return anomalies

    def _detect_rate_anomaly(self, current: TrafficMetrics,
                            historical: List[TrafficMetrics]) -> Optional[Anomaly]:
        """检测速率异常"""
        historical_rates = [m.avg_request_rate for m in historical]
        if not historical_rates:
            return None

        mean_rate = statistics.mean(historical_rates)
        std_rate = statistics.stdev(historical_rates) if len(historical_rates) > 1 else 0

        if std_rate == 0:
            return None

        z_score = (current.avg_request_rate - mean_rate) / std_rate

        if abs(z_score) > self.sensitivity:
            severity = self._calculate_severity(abs(z_score))
            return Anomaly(
                type=AnomalyType.RATE_ANOMALY,
                severity=severity,
                timestamp=datetime.now().timestamp(),
                description=f"Request rate anomaly: {current.avg_request_rate:.2f} req/s "
                          f"(normal: {mean_rate:.2f} ± {std_rate:.2f})",
                metrics={
                    'current_rate': current.avg_request_rate,
                    'mean_rate': mean_rate,
                    'std_rate': std_rate,
                    'z_score': z_score
                }
            )

        return None

    def _detect_error_anomaly(self, current: TrafficMetrics,
                             historical: List[TrafficMetrics]) -> Optional[Anomaly]:
        """检测错误率异常"""
        historical_errors = [m.error_rate for m in historical]
        if not historical_errors:
            return None

        mean_error = statistics.mean(historical_errors)
        std_error = statistics.stdev(historical_errors) if len(historical_errors) > 1 else 0

        # 错误率增加更敏感
        if current.error_rate > mean_error + std_error * self.sensitivity:
            severity = self._calculate_severity(
                (current.error_rate - mean_error) / max(std_error, 0.001)
            )
            return Anomaly(
                type=AnomalyType.RATE_ANOMALY,
                severity=severity,
                timestamp=datetime.now().timestamp(),
                description=f"Error rate anomaly: {current.error_rate:.2%} "
                          f"(normal: {mean_error:.2%})",
                metrics={
                    'current_error_rate': current.error_rate,
                    'mean_error_rate': mean_error,
                    'std_error_rate': std_error
                }
            )

        return None

    def _detect_latency_anomaly(self, current: TrafficMetrics,
                               historical: List[TrafficMetrics]) -> Optional[Anomaly]:
        """检测延迟异常"""
        historical_latencies = [m.p95_latency for m in historical]
        if not historical_latencies:
            return None

        mean_latency = statistics.mean(historical_latencies)
        std_latency = statistics.stdev(historical_latencies) if len(historical_latencies) > 1 else 0

        if std_latency == 0:
            return None

        z_score = (current.p95_latency - mean_latency) / std_latency

        if z_score > self.sensitivity:  # 只关注延迟增加
            severity = self._calculate_severity(z_score)
            return Anomaly(
                type=AnomalyType.PATTERN_ANOMALY,
                severity=severity,
                timestamp=datetime.now().timestamp(),
                description=f"Latency anomaly: P95={current.p95_latency:.2f}ms "
                          f"(normal: {mean_latency:.2f}ms)",
                metrics={
                    'current_p95': current.p95_latency,
                    'mean_p95': mean_latency,
                    'z_score': z_score
                }
            )

        return None

    def _detect_distribution_anomaly(self, current: TrafficMetrics,
                                    historical: List[TrafficMetrics]) -> Optional[Anomaly]:
        """检测流量分布异常"""
        # 简化实现：检查API分布的变化
        if not current.api_distribution or not historical:
            return None

        # 计算当前分布的熵
        total_requests = sum(current.api_distribution.values())
        if total_requests == 0:
            return None

        current_entropy = -sum(
            (count / total_requests) * math.log(count / total_requests)
            for count in current.api_distribution.values() if count > 0
        )

        # 计算历史平均熵
        historical_entropies = []
        for h in historical:
            if h.api_distribution:
                h_total = sum(h.api_distribution.values())
                if h_total > 0:
                    h_entropy = -sum(
                        (count / h_total) * math.log(count / h_total)
                        for count in h.api_distribution.values() if count > 0
                    )
                    historical_entropies.append(h_entropy)

        if not historical_entropies:
            return None

        mean_entropy = statistics.mean(historical_entropies)

        # 熵的显著变化可能表示异常
        entropy_change = abs(current_entropy - mean_entropy)
        if entropy_change > mean_entropy * 0.5:  # 50%的变化
            return Anomaly(
                type=AnomalyType.PATTERN_ANOMALY,
                severity="medium",
                timestamp=datetime.now().timestamp(),
                description=f"Traffic distribution anomaly detected",
                metrics={
                    'current_entropy': current_entropy,
                    'mean_entropy': mean_entropy,
                    'change_ratio': entropy_change / mean_entropy
                }
            )

        return None

    def _calculate_severity(self, z_score: float) -> str:
        """根据Z-score计算严重程度"""
        abs_z = abs(z_score)
        if abs_z < 2:
            return "low"
        elif abs_z < 3:
            return "medium"
        elif abs_z < 4:
            return "high"
        else:
            return "critical"


class RateLimitAnalyzer:
    """限流分析器"""

    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.anomaly_detector = AnomalyDetector()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.historical_metrics: List[TrafficMetrics] = []
        self.max_history = 100

    def analyze_traffic(self, requests: List[Request],
                       time_window: TimeWindow) -> TrafficAnalysis:
        """
        分析时间窗口内的流量

        Args:
            requests: 请求列表
            time_window: 时间窗口

        Returns:
            流量分析结果
        """
        # 计算流量指标
        metrics = self._calculate_metrics(requests, time_window)

        # 检测流量模式
        pattern = self.pattern_detector.detect(requests, metrics)

        # 检测异常
        anomalies = self.anomaly_detector.detect(metrics, self.historical_metrics)

        # 计算置信度
        confidence = self._calculate_confidence(requests, metrics, pattern)

        # 生成建议
        recommendations = self._generate_recommendations(pattern, anomalies, metrics)

        # 更新历史
        self.historical_metrics.append(metrics)
        if len(self.historical_metrics) > self.max_history:
            self.historical_metrics.pop(0)

        return TrafficAnalysis(
            time_window=time_window,
            metrics=metrics,
            pattern_type=pattern,
            anomalies=anomalies,
            confidence=confidence,
            recommendations=recommendations
        )

    def _calculate_metrics(self, requests: List[Request],
                         time_window: TimeWindow) -> TrafficMetrics:
        """计算流量指标"""
        metrics = TrafficMetrics()

        if not requests:
            return metrics

        metrics.request_count = len(requests)
        metrics.unique_users = len(set(r.user_id for r in requests))
        metrics.unique_ips = len(set(r.source_ip for r in requests))

        # 计算速率
        if time_window.duration > 0:
            metrics.avg_request_rate = len(requests) / time_window.duration

            # 计算峰值速率（简化：使用1秒窗口）
            timestamps = [r.timestamp for r in requests]
            if timestamps:
                timestamps.sort()
                max_count_per_second = 0
                for t in range(int(timestamps[0]), int(timestamps[-1]) + 1):
                    count = sum(1 for ts in timestamps if t <= ts < t + 1)
                    max_count_per_second = max(max_count_per_second, count)
                metrics.peak_request_rate = max_count_per_second

        # API分布
        for r in requests:
            metrics.api_distribution[r.api_path] = \
                metrics.api_distribution.get(r.api_path, 0) + 1
            metrics.method_distribution[r.method] = \
                metrics.method_distribution.get(r.method, 0) + 1

        return metrics

    def _calculate_confidence(self, requests: List[Request],
                            metrics: TrafficMetrics,
                            pattern: PatternType) -> float:
        """计算分析置信度"""
        confidence = 1.0

        # 样本量影响
        if len(requests) < 10:
            confidence *= 0.5
        elif len(requests) < 100:
            confidence *= 0.8
        elif len(requests) < 1000:
            confidence *= 0.9

        # 历史数据影响
        if len(self.historical_metrics) < 10:
            confidence *= 0.8

        # 模式清晰度影响
        if pattern == PatternType.RANDOM:
            confidence *= 0.7

        return min(1.0, confidence)

    def _generate_recommendations(self, pattern: PatternType,
                                 anomalies: List[Anomaly],
                                 metrics: TrafficMetrics) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于模式的建议
        pattern_recommendations = {
            PatternType.SPIKE: "启用令牌桶算法处理突发流量",
            PatternType.SUSTAINED: "使用漏桶算法平滑流量",
            PatternType.CRAWLER: "对疑似爬虫IP进行额外限流",
            PatternType.DDOS: "启用DDoS防护模式，考虑使用CDN",
            PatternType.BRUTE_FORCE: "对认证API增加额外保护，如验证码"
        }

        if pattern in pattern_recommendations:
            recommendations.append(pattern_recommendations[pattern])

        # 基于异常的建议
        for anomaly in anomalies:
            if anomaly.type == AnomalyType.RATE_ANOMALY:
                recommendations.append("动态调整限流阈值以适应流量变化")
            elif anomaly.type == AnomalyType.PATTERN_ANOMALY:
                if 'latency' in anomaly.description.lower():
                    recommendations.append("考虑降低限流阈值以减轻系统压力")

        # 基于指标的建议
        if metrics.error_rate > 0.05:
            recommendations.append("错误率较高，建议检查系统健康状态")

        if metrics.unique_ips > metrics.unique_users * 10:
            recommendations.append("检测到可能的IP伪造，建议加强身份验证")

        return recommendations

    def classify_traffic_pattern(self, metrics: TrafficMetrics) -> PatternType:
        """
        分类流量模式

        Args:
            metrics: 流量指标

        Returns:
            流量模式类型
        """
        # 简化的分类逻辑
        if metrics.error_rate > 0.3:
            if metrics.unique_ips > metrics.unique_users * 5:
                return PatternType.DDOS
            else:
                return PatternType.BRUTE_FORCE

        if metrics.peak_request_rate > metrics.avg_request_rate * 3:
            return PatternType.SPIKE

        return PatternType.NORMAL

    def generate_user_profile(self, user_id: str, requests: List[Request]) -> UserProfile:
        """
        生成或更新用户行为画像

        Args:
            user_id: 用户ID
            requests: 该用户的请求列表

        Returns:
            用户画像
        """
        # 获取或创建用户画像
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        profile = self.user_profiles[user_id]

        # 更新基础统计
        profile.total_requests += len(requests)

        if requests:
            # 更新最后活动时间
            profile.last_activity = max(r.timestamp for r in requests)

            # 计算请求速率
            timestamps = [r.timestamp for r in requests]
            if len(timestamps) > 1:
                duration = max(timestamps) - min(timestamps)
                if duration > 0:
                    current_rate = len(requests) / duration
                    # 指数移动平均
                    alpha = 0.3
                    profile.avg_request_rate = (1 - alpha) * profile.avg_request_rate + alpha * current_rate
                    profile.peak_request_rate = max(profile.peak_request_rate, current_rate)

            # 更新API使用分布
            for r in requests:
                profile.api_usage[r.api_path] = profile.api_usage.get(r.api_path, 0) + 1

        # 评估用户行为
        profile = self._evaluate_user_behavior(profile, requests)

        return profile

    def _evaluate_user_behavior(self, profile: UserProfile,
                               recent_requests: List[Request]) -> UserProfile:
        """评估用户行为"""
        # 检查是否可疑
        suspicious_indicators = 0

        # 指标1：请求速率异常高
        if profile.avg_request_rate > 100:  # 100 req/s
            suspicious_indicators += 1

        # 指标2：错误率高
        if profile.error_rate > 0.3:
            suspicious_indicators += 1

        # 指标3：访问模式异常
        if len(profile.api_usage) > 100:  # 访问太多不同的API
            suspicious_indicators += 1

        profile.is_suspicious = suspicious_indicators >= 2

        # 计算信誉分数
        base_score = 100.0
        base_score -= profile.error_rate * 50  # 错误率扣分
        base_score -= min(20, profile.avg_request_rate / 10)  # 高频率扣分

        if profile.is_suspicious:
            base_score -= 30

        profile.reputation_score = max(0, min(100, base_score))

        # 确定访问模式
        if profile.is_suspicious:
            if profile.error_rate > 0.5:
                profile.access_pattern = "brute_force"
            elif len(profile.api_usage) > 100:
                profile.access_pattern = "crawler"
            else:
                profile.access_pattern = "suspicious"
        elif profile.avg_request_rate > 50:
            profile.access_pattern = "heavy"
        else:
            profile.access_pattern = "normal"

        return profile

    def export_analysis(self, analysis: TrafficAnalysis, output_file: str):
        """导出分析结果"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'time_window': {
                'start': analysis.time_window.start,
                'end': analysis.time_window.end,
                'duration': analysis.time_window.duration
            },
            'metrics': asdict(analysis.metrics),
            'pattern': analysis.pattern_type.value,
            'anomalies': [asdict(a) for a in analysis.anomalies],
            'confidence': analysis.confidence,
            'recommendations': analysis.recommendations
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_user_profiles_summary(self) -> Dict[str, Any]:
        """获取用户画像汇总"""
        total_users = len(self.user_profiles)
        suspicious_users = sum(1 for p in self.user_profiles.values() if p.is_suspicious)

        pattern_distribution = Counter(
            p.access_pattern for p in self.user_profiles.values()
        )

        avg_reputation = statistics.mean(
            [p.reputation_score for p in self.user_profiles.values()]
        ) if self.user_profiles else 0

        return {
            'total_users': total_users,
            'suspicious_users': suspicious_users,
            'suspicious_rate': suspicious_users / total_users if total_users > 0 else 0,
            'pattern_distribution': dict(pattern_distribution),
            'avg_reputation_score': avg_reputation,
            'top_requesters': sorted(
                self.user_profiles.values(),
                key=lambda p: p.total_requests,
                reverse=True
            )[:10]
        }


def main():
    """主函数 - 用于测试"""
    import argparse
    import random

    parser = argparse.ArgumentParser(description='Rate Limit Analyzer')
    parser.add_argument('--analyze', help='Analyze request log file')
    parser.add_argument('--output', help='Output analysis results')
    parser.add_argument('--simulate', action='store_true', help='Simulate and analyze traffic')

    args = parser.parse_args()

    analyzer = RateLimitAnalyzer()

    if args.simulate:
        print("🔍 Simulating and analyzing traffic...")

        # 生成模拟请求
        requests = []
        start_time = time.time()

        # 模拟不同的流量模式
        for i in range(1000):
            # 创建请求
            request = Request(
                id=f"req_{i}",
                user_id=f"user_{random.randint(1, 20)}",
                api_path=random.choice([
                    "/api/v1/users",
                    "/api/v1/products",
                    "/api/v1/orders",
                    "/api/v1/login",
                    "/api/v1/search"
                ]),
                method=random.choice(["GET", "POST", "PUT", "DELETE"]),
                source_ip=f"192.168.1.{random.randint(1, 50)}",
                timestamp=start_time + i * random.uniform(0.01, 0.5)
            )
            requests.append(request)

        # 分析流量
        end_time = start_time + 60  # 60秒窗口
        time_window = TimeWindow(start=start_time, end=end_time)
        analysis = analyzer.analyze_traffic(requests, time_window)

        print(f"\n📊 Analysis Results:")
        print(f"   Pattern: {analysis.pattern_type.value}")
        print(f"   Confidence: {analysis.confidence:.2%}")
        print(f"   Request Count: {analysis.metrics.request_count}")
        print(f"   Unique Users: {analysis.metrics.unique_users}")
        print(f"   Avg Rate: {analysis.metrics.avg_request_rate:.2f} req/s")

        if analysis.anomalies:
            print(f"\n⚠️ Anomalies Detected:")
            for anomaly in analysis.anomalies:
                print(f"   - {anomaly.type.value}: {anomaly.description}")

        if analysis.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in analysis.recommendations:
                print(f"   - {rec}")

        # 用户画像汇总
        for req in requests:
            user_requests = [r for r in requests if r.user_id == req.user_id]
            analyzer.generate_user_profile(req.user_id, user_requests)

        summary = analyzer.get_user_profiles_summary()
        print(f"\n👥 User Profiles:")
        print(f"   Total Users: {summary['total_users']}")
        print(f"   Suspicious Users: {summary['suspicious_users']}")
        print(f"   Avg Reputation: {summary['avg_reputation_score']:.1f}")

        # 导出结果
        if args.output:
            analyzer.export_analysis(analysis, args.output)
            print(f"\n✅ Analysis exported to {args.output}")


if __name__ == "__main__":
    main()