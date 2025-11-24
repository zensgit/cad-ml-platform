#!/usr/bin/env python3
"""
性能影响监控器 (Performance Monitor)
监控限流对系统性能的影响，检查SLA合规性，生成告警
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics
import asyncio
from collections import deque, defaultdict
import threading
import signal
import sys

# 导入兼容性处理
try:
    from .adaptive_rate_limiter import (
        SystemMetrics,
        Decision,
        AdaptiveRateLimiter,
    )
    from .rate_limit_analyzer import (
        TrafficAnalysis,
        PatternType,
        RateLimitAnalyzer,
    )
    from .auto_calibrator import (
        Parameters,
        PerformanceScore,
        AutoCalibrator,
    )
except ImportError:
    from adaptive_rate_limiter import (
        SystemMetrics,
        Decision,
        AdaptiveRateLimiter,
    )
    from rate_limit_analyzer import (
        TrafficAnalysis,
        PatternType,
        RateLimitAnalyzer,
    )
    from auto_calibrator import (
        Parameters,
        PerformanceScore,
        AutoCalibrator,
    )

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """SLA合规状态"""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATION = "violation"
    CRITICAL_VIOLATION = "critical_violation"


class ImpactLevel(Enum):
    """影响级别"""
    NEGLIGIBLE = "negligible"  # 可忽略
    LOW = "low"  # 低影响
    MEDIUM = "medium"  # 中等影响
    HIGH = "high"  # 高影响
    SEVERE = "severe"  # 严重影响


@dataclass
class SLAConfig:
    """SLA配置"""
    availability_target: float = 0.999  # 99.9%可用性
    latency_p95_target: float = 100  # P95延迟 < 100ms
    latency_p99_target: float = 200  # P99延迟 < 200ms
    error_rate_target: float = 0.001  # 错误率 < 0.1%
    min_throughput: float = 1000  # 最小吞吐量 req/s

    # 告警阈值
    warning_buffer: float = 0.1  # 警告缓冲区 (10%)
    critical_buffer: float = 0.05  # 严重缓冲区 (5%)

    # 评估窗口
    evaluation_window: int = 300  # 评估窗口（秒）
    rolling_windows: List[int] = field(default_factory=lambda: [60, 300, 3600])  # 1分钟、5分钟、1小时


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    severity: AlertSeverity
    category: str  # 限流过度、性能退化、SLA违规等
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    action_required: Optional[str] = None


@dataclass
class ImpactReport:
    """影响报告"""
    time_range: Tuple[datetime, datetime]
    impact_level: ImpactLevel
    affected_metrics: Dict[str, float]
    baseline_comparison: Dict[str, float]  # 相对基线的变化
    sla_violations: List[str]
    recommendations: List[str]
    confidence: float  # 置信度


@dataclass
class MetricSnapshot:
    """指标快照"""
    timestamp: datetime
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    rate_limited_requests: int
    total_requests: int
    active_connections: int


class PerformanceMonitor:
    """性能影响监控器"""

    def __init__(
        self,
        sla_config: Optional[SLAConfig] = None,
        baseline_file: Optional[str] = None
    ):
        """
        初始化监控器

        Args:
            sla_config: SLA配置
            baseline_file: 基线数据文件
        """
        self.sla = sla_config or SLAConfig()
        self.baseline_file = baseline_file or "performance_baseline.json"

        # 指标存储
        self.metrics_history: deque[MetricSnapshot] = deque(maxlen=10000)
        self.baseline_metrics: Optional[Dict[str, float]] = None

        # 告警管理
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # 统计缓存
        self.stats_cache: Dict[str, Dict[str, float]] = {}
        self.cache_timestamp: Optional[datetime] = None

        # 监控状态
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 限流影响追踪
        self.rate_limit_impacts: List[Dict[str, Any]] = []
        self.last_adjustment_time: Optional[datetime] = None

        # 加载基线
        self._load_baseline()

    def monitor_impact(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> ImpactReport:
        """
        监控限流影响

        Args:
            time_range: 时间范围，None表示最近评估窗口

        Returns:
            影响报告
        """
        if time_range is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=self.sla.evaluation_window)
            time_range = (start_time, end_time)

        logger.info(f"监控时间范围: {time_range[0]} 到 {time_range[1]}")

        # 获取时间范围内的指标
        metrics = self._get_metrics_in_range(time_range)

        if not metrics:
            logger.warning("没有找到指标数据")
            return ImpactReport(
                time_range=time_range,
                impact_level=ImpactLevel.NEGLIGIBLE,
                affected_metrics={},
                baseline_comparison={},
                sla_violations=[],
                recommendations=["需要更多数据进行评估"],
                confidence=0.0
            )

        # 计算聚合指标
        affected_metrics = self._calculate_aggregated_metrics(metrics)

        # 对比基线
        baseline_comparison = self._compare_with_baseline(affected_metrics)

        # 检查SLA违规
        sla_violations = self._check_sla_violations(affected_metrics)

        # 评估影响级别
        impact_level = self._assess_impact_level(
            affected_metrics,
            baseline_comparison,
            sla_violations
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            impact_level,
            affected_metrics,
            sla_violations
        )

        # 计算置信度
        confidence = min(1.0, len(metrics) / 100)  # 基于样本量的简单置信度

        report = ImpactReport(
            time_range=time_range,
            impact_level=impact_level,
            affected_metrics=affected_metrics,
            baseline_comparison=baseline_comparison,
            sla_violations=sla_violations,
            recommendations=recommendations,
            confidence=confidence
        )

        # 记录影响
        self._record_impact(report)

        return report

    def check_sla_compliance(self) -> ComplianceStatus:
        """
        检查SLA合规性

        Returns:
            合规状态
        """
        # 获取最近的指标
        recent_metrics = self._get_recent_metrics(self.sla.evaluation_window)

        if not recent_metrics:
            logger.warning("没有足够的数据检查SLA合规性")
            return ComplianceStatus.COMPLIANT

        # 计算聚合指标
        aggregated = self._calculate_aggregated_metrics(recent_metrics)

        # 检查各项SLA
        violations = []
        warnings = []

        # 可用性检查
        availability = 1.0 - aggregated.get('error_rate', 0)
        if availability < self.sla.availability_target:
            if availability < self.sla.availability_target * (1 - self.sla.critical_buffer):
                violations.append(f"可用性严重违规: {availability:.3%} < {self.sla.availability_target:.3%}")
            else:
                warnings.append(f"可用性接近违规: {availability:.3%}")

        # P95延迟检查
        p95_latency = aggregated.get('latency_p95', 0)
        if p95_latency > self.sla.latency_p95_target:
            if p95_latency > self.sla.latency_p95_target * (1 + self.sla.warning_buffer):
                violations.append(f"P95延迟违规: {p95_latency:.0f}ms > {self.sla.latency_p95_target:.0f}ms")
            else:
                warnings.append(f"P95延迟接近违规: {p95_latency:.0f}ms")

        # P99延迟检查
        p99_latency = aggregated.get('latency_p99', 0)
        if p99_latency > self.sla.latency_p99_target:
            if p99_latency > self.sla.latency_p99_target * (1 + self.sla.warning_buffer):
                violations.append(f"P99延迟违规: {p99_latency:.0f}ms > {self.sla.latency_p99_target:.0f}ms")
            else:
                warnings.append(f"P99延迟接近违规: {p99_latency:.0f}ms")

        # 错误率检查
        error_rate = aggregated.get('error_rate', 0)
        if error_rate > self.sla.error_rate_target:
            if error_rate > self.sla.error_rate_target * (1 + self.sla.warning_buffer):
                violations.append(f"错误率违规: {error_rate:.3%} > {self.sla.error_rate_target:.3%}")
            else:
                warnings.append(f"错误率接近违规: {error_rate:.3%}")

        # 吞吐量检查
        throughput = aggregated.get('throughput', float('inf'))
        if throughput < self.sla.min_throughput:
            if throughput < self.sla.min_throughput * (1 - self.sla.warning_buffer):
                violations.append(f"吞吐量违规: {throughput:.0f} < {self.sla.min_throughput:.0f} req/s")
            else:
                warnings.append(f"吞吐量接近违规: {throughput:.0f} req/s")

        # 确定合规状态
        if violations:
            logger.error(f"SLA违规: {violations}")
            if len(violations) > 2:
                return ComplianceStatus.CRITICAL_VIOLATION
            return ComplianceStatus.VIOLATION
        elif warnings:
            logger.warning(f"SLA警告: {warnings}")
            return ComplianceStatus.AT_RISK
        else:
            return ComplianceStatus.COMPLIANT

    def generate_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        生成告警

        Args:
            severity: 最低严重级别，None表示所有级别

        Returns:
            告警列表
        """
        alerts = []
        current_time = datetime.now()

        # 检查SLA合规性
        compliance_status = self.check_sla_compliance()

        if compliance_status == ComplianceStatus.CRITICAL_VIOLATION:
            alert = self._create_alert(
                AlertSeverity.CRITICAL,
                "SLA违规",
                "多项SLA指标严重违规",
                {'compliance_status': compliance_status.value}
            )
            alerts.append(alert)
        elif compliance_status == ComplianceStatus.VIOLATION:
            alert = self._create_alert(
                AlertSeverity.ERROR,
                "SLA违规",
                "SLA指标违规",
                {'compliance_status': compliance_status.value}
            )
            alerts.append(alert)
        elif compliance_status == ComplianceStatus.AT_RISK:
            alert = self._create_alert(
                AlertSeverity.WARNING,
                "SLA风险",
                "SLA指标接近违规阈值",
                {'compliance_status': compliance_status.value}
            )
            alerts.append(alert)

        # 检查限流过度
        rate_limit_alert = self._check_rate_limit_impact()
        if rate_limit_alert:
            alerts.append(rate_limit_alert)

        # 检查性能退化
        performance_alert = self._check_performance_degradation()
        if performance_alert:
            alerts.append(performance_alert)

        # 检查资源使用
        resource_alert = self._check_resource_usage()
        if resource_alert:
            alerts.append(resource_alert)

        # 过滤严重级别
        if severity:
            severity_order = {
                AlertSeverity.INFO: 0,
                AlertSeverity.WARNING: 1,
                AlertSeverity.ERROR: 2,
                AlertSeverity.CRITICAL: 3
            }
            min_severity = severity_order[severity]
            alerts = [a for a in alerts if severity_order[a.severity] >= min_severity]

        # 更新活动告警
        for alert in alerts:
            if alert.alert_id not in self.active_alerts:
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                logger.info(f"新告警: [{alert.severity.value}] {alert.message}")

        return alerts

    def _check_rate_limit_impact(self) -> Optional[Alert]:
        """检查限流影响"""
        recent_metrics = self._get_recent_metrics(60)  # 最近1分钟

        if not recent_metrics:
            return None

        # 计算限流比例
        total_requests = sum(m.total_requests for m in recent_metrics)
        rate_limited = sum(m.rate_limited_requests for m in recent_metrics)

        if total_requests > 0:
            rate_limit_ratio = rate_limited / total_requests

            if rate_limit_ratio > 0.2:  # 20%以上请求被限流
                return self._create_alert(
                    AlertSeverity.ERROR,
                    "限流过度",
                    f"过多请求被限流: {rate_limit_ratio:.1%}",
                    {
                        'rate_limited': rate_limited,
                        'total_requests': total_requests,
                        'ratio': rate_limit_ratio
                    }
                )
            elif rate_limit_ratio > 0.1:  # 10%以上请求被限流
                return self._create_alert(
                    AlertSeverity.WARNING,
                    "限流警告",
                    f"部分请求被限流: {rate_limit_ratio:.1%}",
                    {
                        'rate_limited': rate_limited,
                        'total_requests': total_requests,
                        'ratio': rate_limit_ratio
                    }
                )

        return None

    def _check_performance_degradation(self) -> Optional[Alert]:
        """检查性能退化"""
        if not self.baseline_metrics:
            return None

        recent_metrics = self._get_recent_metrics(300)  # 最近5分钟

        if not recent_metrics:
            return None

        # 计算当前性能
        current = self._calculate_aggregated_metrics(recent_metrics)

        # 对比基线
        degradation = {}
        for key in ['latency_p95', 'latency_p99', 'error_rate']:
            if key in current and key in self.baseline_metrics:
                baseline = self.baseline_metrics[key]
                if baseline > 0:
                    change = (current[key] - baseline) / baseline
                    if abs(change) > 0.5:  # 变化超过50%
                        degradation[key] = change

        if degradation:
            severity = AlertSeverity.ERROR if any(v > 1.0 for v in degradation.values()) else AlertSeverity.WARNING
            return self._create_alert(
                severity,
                "性能退化",
                f"性能指标相对基线退化: {list(degradation.keys())}",
                {'degradation': degradation}
            )

        return None

    def _check_resource_usage(self) -> Optional[Alert]:
        """检查资源使用"""
        recent_metrics = self._get_recent_metrics(60)  # 最近1分钟

        if not recent_metrics:
            return None

        # 计算平均资源使用
        avg_cpu = statistics.mean(m.cpu_usage for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage for m in recent_metrics)

        if avg_cpu > 0.9:
            return self._create_alert(
                AlertSeverity.CRITICAL,
                "资源告警",
                f"CPU使用率过高: {avg_cpu:.1%}",
                {'cpu_usage': avg_cpu, 'memory_usage': avg_memory}
            )
        elif avg_memory > 0.9:
            return self._create_alert(
                AlertSeverity.CRITICAL,
                "资源告警",
                f"内存使用率过高: {avg_memory:.1%}",
                {'cpu_usage': avg_cpu, 'memory_usage': avg_memory}
            )
        elif avg_cpu > 0.8 or avg_memory > 0.8:
            return self._create_alert(
                AlertSeverity.WARNING,
                "资源警告",
                f"资源使用率较高: CPU {avg_cpu:.1%}, 内存 {avg_memory:.1%}",
                {'cpu_usage': avg_cpu, 'memory_usage': avg_memory}
            )

        return None

    def _create_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        details: Dict[str, Any]
    ) -> Alert:
        """创建告警"""
        alert_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 添加建议的操作
        action_required = self._get_action_for_alert(severity, category)

        return Alert(
            alert_id=alert_id,
            severity=severity,
            category=category,
            message=message,
            details=details,
            timestamp=datetime.now(),
            action_required=action_required
        )

    def _get_action_for_alert(
        self,
        severity: AlertSeverity,
        category: str
    ) -> str:
        """获取告警建议操作"""
        actions = {
            "SLA违规": "考虑调整限流参数或扩容",
            "限流过度": "增加限流阈值或优化算法",
            "性能退化": "检查系统负载，考虑优化或扩容",
            "资源告警": "立即扩容或优化资源使用",
            "资源警告": "监控资源趋势，准备扩容",
        }
        return actions.get(category, "检查系统状态")

    def add_metric_snapshot(self, snapshot: MetricSnapshot):
        """添加指标快照"""
        self.metrics_history.append(snapshot)

        # 清理缓存
        if self.cache_timestamp and \
           (datetime.now() - self.cache_timestamp).seconds > 60:
            self.stats_cache.clear()
            self.cache_timestamp = None

    def _get_metrics_in_range(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> List[MetricSnapshot]:
        """获取时间范围内的指标"""
        start_time, end_time = time_range
        return [
            m for m in self.metrics_history
            if start_time <= m.timestamp <= end_time
        ]

    def _get_recent_metrics(self, seconds: int) -> List[MetricSnapshot]:
        """获取最近的指标"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        return [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

    def _calculate_aggregated_metrics(
        self,
        metrics: List[MetricSnapshot]
    ) -> Dict[str, float]:
        """计算聚合指标"""
        if not metrics:
            return {}

        # 使用缓存
        cache_key = f"{len(metrics)}_{metrics[0].timestamp}_{metrics[-1].timestamp}"
        if cache_key in self.stats_cache:
            return self.stats_cache[cache_key]

        aggregated = {
            'throughput': statistics.mean(m.throughput for m in metrics),
            'latency_p50': statistics.median(m.latency_p50 for m in metrics),
            'latency_p95': self._percentile([m.latency_p95 for m in metrics], 95),
            'latency_p99': self._percentile([m.latency_p99 for m in metrics], 99),
            'error_rate': statistics.mean(m.error_rate for m in metrics),
            'cpu_usage': statistics.mean(m.cpu_usage for m in metrics),
            'memory_usage': statistics.mean(m.memory_usage for m in metrics),
            'rate_limit_ratio': sum(m.rate_limited_requests for m in metrics) /
                               max(1, sum(m.total_requests for m in metrics))
        }

        # 缓存结果
        self.stats_cache[cache_key] = aggregated
        self.cache_timestamp = datetime.now()

        return aggregated

    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _compare_with_baseline(
        self,
        current_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """对比基线"""
        if not self.baseline_metrics:
            return {}

        comparison = {}
        for key, current_value in current_metrics.items():
            if key in self.baseline_metrics:
                baseline_value = self.baseline_metrics[key]
                if baseline_value > 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    comparison[key] = change_percent

        return comparison

    def _check_sla_violations(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """检查SLA违规"""
        violations = []

        # 检查各项SLA
        availability = 1.0 - metrics.get('error_rate', 0)
        if availability < self.sla.availability_target:
            violations.append(f"可用性: {availability:.3%} < {self.sla.availability_target:.3%}")

        if metrics.get('latency_p95', 0) > self.sla.latency_p95_target:
            violations.append(f"P95延迟: {metrics['latency_p95']:.0f}ms > {self.sla.latency_p95_target:.0f}ms")

        if metrics.get('latency_p99', 0) > self.sla.latency_p99_target:
            violations.append(f"P99延迟: {metrics['latency_p99']:.0f}ms > {self.sla.latency_p99_target:.0f}ms")

        if metrics.get('error_rate', 0) > self.sla.error_rate_target:
            violations.append(f"错误率: {metrics['error_rate']:.3%} > {self.sla.error_rate_target:.3%}")

        if metrics.get('throughput', float('inf')) < self.sla.min_throughput:
            violations.append(f"吞吐量: {metrics.get('throughput', 0):.0f} < {self.sla.min_throughput:.0f}")

        return violations

    def _assess_impact_level(
        self,
        metrics: Dict[str, float],
        baseline_comparison: Dict[str, float],
        sla_violations: List[str]
    ) -> ImpactLevel:
        """评估影响级别"""
        # 基于SLA违规数量
        if len(sla_violations) >= 3:
            return ImpactLevel.SEVERE
        elif len(sla_violations) >= 2:
            return ImpactLevel.HIGH
        elif len(sla_violations) >= 1:
            return ImpactLevel.MEDIUM

        # 基于基线对比
        if baseline_comparison:
            max_degradation = max(abs(v) for v in baseline_comparison.values())
            if max_degradation > 100:  # 退化超过100%
                return ImpactLevel.HIGH
            elif max_degradation > 50:  # 退化超过50%
                return ImpactLevel.MEDIUM
            elif max_degradation > 20:  # 退化超过20%
                return ImpactLevel.LOW

        # 基于限流比例
        rate_limit_ratio = metrics.get('rate_limit_ratio', 0)
        if rate_limit_ratio > 0.3:
            return ImpactLevel.HIGH
        elif rate_limit_ratio > 0.1:
            return ImpactLevel.MEDIUM
        elif rate_limit_ratio > 0.05:
            return ImpactLevel.LOW

        return ImpactLevel.NEGLIGIBLE

    def _generate_recommendations(
        self,
        impact_level: ImpactLevel,
        metrics: Dict[str, float],
        sla_violations: List[str]
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        if impact_level in [ImpactLevel.HIGH, ImpactLevel.SEVERE]:
            recommendations.append("立即调整限流参数或回滚到上一个稳定配置")
            recommendations.append("考虑紧急扩容以应对当前负载")

        if sla_violations:
            if any("延迟" in v for v in sla_violations):
                recommendations.append("优化限流算法以减少延迟影响")
            if any("错误率" in v for v in sla_violations):
                recommendations.append("检查限流策略是否过于严格")
            if any("吞吐量" in v for v in sla_violations):
                recommendations.append("增加限流阈值以提高吞吐量")

        if metrics.get('rate_limit_ratio', 0) > 0.1:
            recommendations.append("分析被限流的请求模式，优化限流规则")

        if metrics.get('cpu_usage', 0) > 0.8:
            recommendations.append("CPU使用率高，考虑优化代码或增加实例")

        if metrics.get('memory_usage', 0) > 0.8:
            recommendations.append("内存使用率高，检查内存泄漏或增加内存")

        if not recommendations:
            recommendations.append("继续监控系统性能")
            recommendations.append("收集更多数据以建立准确基线")

        return recommendations

    def _record_impact(self, report: ImpactReport):
        """记录影响"""
        self.rate_limit_impacts.append({
            'timestamp': datetime.now(),
            'impact_level': report.impact_level.value,
            'sla_violations': len(report.sla_violations),
            'confidence': report.confidence
        })

        # 保持最近100条记录
        if len(self.rate_limit_impacts) > 100:
            self.rate_limit_impacts = self.rate_limit_impacts[-100:]

    def update_baseline(self, metrics: Optional[Dict[str, float]] = None):
        """
        更新基线

        Args:
            metrics: 新的基线指标，None表示使用当前性能
        """
        if metrics:
            self.baseline_metrics = metrics
        else:
            # 使用最近1小时的平均值作为基线
            recent_metrics = self._get_recent_metrics(3600)
            if recent_metrics:
                self.baseline_metrics = self._calculate_aggregated_metrics(recent_metrics)
                logger.info(f"更新基线: {self.baseline_metrics}")

        # 保存基线
        self._save_baseline()

    def _load_baseline(self):
        """加载基线"""
        if Path(self.baseline_file).exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baseline_metrics = json.load(f)
                logger.info(f"加载基线: {self.baseline_metrics}")
            except Exception as e:
                logger.error(f"加载基线失败: {e}")

    def _save_baseline(self):
        """保存基线"""
        if self.baseline_metrics:
            try:
                with open(self.baseline_file, 'w') as f:
                    json.dump(self.baseline_metrics, f, indent=2)
                logger.info(f"基线已保存到 {self.baseline_file}")
            except Exception as e:
                logger.error(f"保存基线失败: {e}")

    def start_monitoring(self, interval: int = 10):
        """
        启动监控

        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            logger.warning("监控已在运行")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"监控已启动，间隔: {interval}秒")

    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("监控已停止")

    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集当前指标
                snapshot = self._collect_current_metrics()
                self.add_metric_snapshot(snapshot)

                # 检查告警
                alerts = self.generate_alerts()

                # 定期检查SLA
                if len(self.metrics_history) % 6 == 0:  # 每分钟检查一次
                    compliance = self.check_sla_compliance()
                    logger.info(f"SLA合规状态: {compliance.value}")

                time.sleep(interval)

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(interval)

    def _collect_current_metrics(self) -> MetricSnapshot:
        """收集当前指标（模拟）"""
        import random

        # 模拟指标收集
        return MetricSnapshot(
            timestamp=datetime.now(),
            throughput=random.uniform(800, 1200),
            latency_p50=random.uniform(40, 60),
            latency_p95=random.uniform(80, 120),
            latency_p99=random.uniform(150, 250),
            error_rate=random.uniform(0.001, 0.01),
            cpu_usage=random.uniform(0.5, 0.8),
            memory_usage=random.uniform(0.6, 0.8),
            rate_limited_requests=random.randint(0, 100),
            total_requests=random.randint(1000, 1500),
            active_connections=random.randint(4000, 6000)
        )

    def export_report(self, format: str = 'json') -> str:
        """
        导出监控报告

        Args:
            format: 报告格式 (json, markdown)

        Returns:
            报告内容
        """
        # 生成影响报告
        impact_report = self.monitor_impact()

        # 检查SLA合规性
        compliance = self.check_sla_compliance()

        # 获取当前告警
        current_alerts = list(self.active_alerts.values())

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'compliance_status': compliance.value,
            'impact_level': impact_report.impact_level.value,
            'active_alerts': len(current_alerts),
            'sla_violations': impact_report.sla_violations,
            'recommendations': impact_report.recommendations,
            'metrics_summary': impact_report.affected_metrics,
            'baseline_comparison': impact_report.baseline_comparison,
            'recent_impacts': self.rate_limit_impacts[-10:]
        }

        if format == 'json':
            return json.dumps(report_data, indent=2)
        elif format == 'markdown':
            return self._format_markdown_report(report_data)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _format_markdown_report(self, data: Dict[str, Any]) -> str:
        """格式化Markdown报告"""
        lines = [
            "# 性能监控报告",
            f"\n**生成时间**: {data['timestamp']}",
            f"\n## SLA合规状态: {data['compliance_status']}",
            f"\n## 影响级别: {data['impact_level']}",
            f"\n## 活动告警: {data['active_alerts']}",
        ]

        if data['sla_violations']:
            lines.append("\n## SLA违规")
            for violation in data['sla_violations']:
                lines.append(f"- {violation}")

        if data['recommendations']:
            lines.append("\n## 建议")
            for rec in data['recommendations']:
                lines.append(f"- {rec}")

        if data['metrics_summary']:
            lines.append("\n## 指标摘要")
            for key, value in data['metrics_summary'].items():
                lines.append(f"- **{key}**: {value:.2f}")

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="性能影响监控器 - 监控限流对系统性能的影响"
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='启动实时监控'
    )
    parser.add_argument(
        '--check-sla',
        action='store_true',
        help='检查SLA合规性'
    )
    parser.add_argument(
        '--export',
        type=str,
        choices=['json', 'markdown'],
        help='导出报告格式'
    )
    parser.add_argument(
        '--update-baseline',
        action='store_true',
        help='更新性能基线'
    )

    args = parser.parse_args()

    # 创建监控器
    monitor = PerformanceMonitor()

    if args.monitor:
        # 启动实时监控
        print("启动性能监控...")
        monitor.start_monitoring(interval=10)

        # 处理退出信号
        def signal_handler(sig, frame):
            print("\n停止监控...")
            monitor.stop_monitoring()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    elif args.check_sla:
        # 检查SLA合规性
        # 添加一些模拟数据
        for _ in range(100):
            monitor.add_metric_snapshot(monitor._collect_current_metrics())
            time.sleep(0.01)

        status = monitor.check_sla_compliance()
        print(f"SLA合规状态: {status.value}")

        # 生成告警
        alerts = monitor.generate_alerts()
        if alerts:
            print(f"\n发现 {len(alerts)} 个告警:")
            for alert in alerts:
                print(f"  [{alert.severity.value}] {alert.message}")
                if alert.action_required:
                    print(f"    建议操作: {alert.action_required}")
        else:
            print("没有发现告警")

    elif args.export:
        # 导出报告
        # 添加一些模拟数据
        for _ in range(100):
            monitor.add_metric_snapshot(monitor._collect_current_metrics())
            time.sleep(0.01)

        report = monitor.export_report(format=args.export)
        print(report)

    elif args.update_baseline:
        # 更新基线
        # 收集1分钟的数据作为基线
        print("收集基线数据（1分钟）...")
        for _ in range(60):
            monitor.add_metric_snapshot(monitor._collect_current_metrics())
            time.sleep(1)

        monitor.update_baseline()
        print("基线已更新")

    else:
        # 运行影响分析
        print("运行性能影响分析...")

        # 添加一些模拟数据
        for _ in range(100):
            monitor.add_metric_snapshot(monitor._collect_current_metrics())
            time.sleep(0.01)

        # 生成影响报告
        report = monitor.monitor_impact()

        print(f"\n影响级别: {report.impact_level.value}")
        print(f"置信度: {report.confidence:.2%}")

        if report.sla_violations:
            print("\nSLA违规:")
            for violation in report.sla_violations:
                print(f"  - {violation}")

        if report.recommendations:
            print("\n建议:")
            for rec in report.recommendations:
                print(f"  - {rec}")

        if report.baseline_comparison:
            print("\n相对基线变化:")
            for key, change in report.baseline_comparison.items():
                sign = "+" if change > 0 else ""
                print(f"  - {key}: {sign}{change:.1f}%")


if __name__ == "__main__":
    main()