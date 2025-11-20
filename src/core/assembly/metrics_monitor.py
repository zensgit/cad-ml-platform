"""
观测与SLO监控
Prometheus指标导出和Grafana面板配置
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Gauge, Histogram

# Prometheus指标定义
assembly_requests_total = Counter(
    "assembly_requests_total", "Total number of assembly analysis requests", ["engine", "status"]
)

assembly_edge_f1 = Gauge(
    "assembly_edge_f1", "Edge F1 score for assembly analysis", ["model_version"]
)

assembly_evidence_coverage = Gauge(
    "assembly_evidence_coverage", "Evidence coverage ratio", ["evidence_type"]
)

assembly_cache_hit_ratio = Gauge(
    "assembly_cache_hit_ratio", "Cache hit ratio for assembly analysis"
)

assembly_latency = Histogram(
    "assembly_latency_ms",
    "Assembly analysis latency in milliseconds",
    ["operation"],
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

assembly_confidence_distribution = Histogram(
    "assembly_confidence_distribution",
    "Distribution of confidence scores",
    ["calibrated"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

assembly_constraint_fallbacks = Counter(
    "assembly_constraint_fallbacks_total",
    "Total number of constraint fallbacks applied",
    ["original_type", "fallback_type", "engine"],
)

assembly_simulation_success_rate = Gauge(
    "assembly_simulation_success_rate", "Simulation success rate", ["engine"]
)

assembly_part_count = Histogram(
    "assembly_part_count", "Number of parts in assembly", buckets=[1, 5, 10, 20, 50, 100, 200, 500]
)

assembly_processing_errors = Counter(
    "assembly_processing_errors_total", "Total number of processing errors", ["error_type", "stage"]
)


@dataclass
class SLOTarget:
    """SLO目标定义"""

    name: str
    target_value: float
    current_value: float
    time_window: timedelta
    is_met: bool
    severity: str  # critical, warning, info


class AssemblyMetricsMonitor:
    """装配分析指标监控器"""

    def __init__(self):
        self.slo_targets = self._init_slo_targets()
        self.metrics_buffer = []
        self.alert_thresholds = self._init_alert_thresholds()

    def _init_slo_targets(self) -> Dict[str, SLOTarget]:
        """初始化SLO目标"""

        return {
            "edge_f1": SLOTarget(
                name="Edge F1 Score",
                target_value=0.75,
                current_value=0.0,
                time_window=timedelta(hours=1),
                is_met=False,
                severity="critical",
            ),
            "latency_p95": SLOTarget(
                name="95th Percentile Latency",
                target_value=2000,  # ms
                current_value=0.0,
                time_window=timedelta(minutes=5),
                is_met=True,
                severity="warning",
            ),
            "simulation_success": SLOTarget(
                name="Simulation Success Rate",
                target_value=0.95,
                current_value=0.0,
                time_window=timedelta(hours=1),
                is_met=False,
                severity="warning",
            ),
            "evidence_coverage": SLOTarget(
                name="Evidence Coverage",
                target_value=0.90,
                current_value=0.0,
                time_window=timedelta(hours=1),
                is_met=False,
                severity="info",
            ),
            "cache_hit_ratio": SLOTarget(
                name="Cache Hit Ratio",
                target_value=0.60,
                current_value=0.0,
                time_window=timedelta(hours=1),
                is_met=False,
                severity="info",
            ),
        }

    def _init_alert_thresholds(self) -> Dict:
        """初始化告警阈值"""

        return {
            "error_rate": {"warning": 0.01, "critical": 0.05},  # 1%  # 5%
            "latency_ms": {"warning": 3000, "critical": 10000},
            "memory_usage_mb": {"warning": 1024, "critical": 2048},
            "constraint_fallback_rate": {"warning": 0.20, "critical": 0.50},  # 20%  # 50%
        }

    def record_request(
        self, engine: str, status: str, latency_ms: float, metadata: Optional[Dict] = None
    ):
        """记录请求指标"""

        # 更新计数器
        assembly_requests_total.labels(engine=engine, status=status).inc()

        # 记录延迟
        assembly_latency.labels(operation="full_analysis").observe(latency_ms)

        # 缓存指标数据
        self.metrics_buffer.append(
            {
                "timestamp": datetime.now(),
                "engine": engine,
                "status": status,
                "latency_ms": latency_ms,
                "metadata": metadata or {},
            }
        )

        # 检查SLO
        self._check_slo_compliance()

    def record_analysis_quality(
        self,
        edge_f1: float,
        evidence_coverage: float,
        confidence_scores: list,
        model_version: str = "v1.0.0",
    ):
        """记录分析质量指标"""

        # 更新质量指标
        assembly_edge_f1.labels(model_version=model_version).set(edge_f1)
        assembly_evidence_coverage.labels(evidence_type="all").set(evidence_coverage)

        # 记录置信度分布
        for score in confidence_scores:
            assembly_confidence_distribution.labels(calibrated="false").observe(score)

        # 更新SLO
        self.slo_targets["edge_f1"].current_value = edge_f1
        self.slo_targets["evidence_coverage"].current_value = evidence_coverage

    def record_cache_metrics(self, hits: int, misses: int):
        """记录缓存指标"""

        total = hits + misses
        if total > 0:
            hit_ratio = hits / total
            assembly_cache_hit_ratio.set(hit_ratio)
            self.slo_targets["cache_hit_ratio"].current_value = hit_ratio

    def record_constraint_fallback(self, original_type: str, fallback_type: str, engine: str):
        """记录约束降级"""

        assembly_constraint_fallbacks.labels(
            original_type=original_type, fallback_type=fallback_type, engine=engine
        ).inc()

    def record_simulation_result(self, engine: str, success: bool):
        """记录仿真结果"""

        # 这里简化处理，实际应该维护滑动窗口
        # Access internal metrics storage safely; fallback to 0.0
        current_rate = 0.0
        try:
            current_rate = assembly_simulation_success_rate._metrics.get((engine,), 0.0)  # type: ignore[attr-defined]
        except Exception:
            pass
        # 简单的指数移动平均
        new_rate = 0.95 * current_rate + 0.05 * (1.0 if success else 0.0)
        assembly_simulation_success_rate.labels(engine=engine).set(new_rate)

        self.slo_targets["simulation_success"].current_value = new_rate

    def record_error(self, error_type: str, stage: str):
        """记录错误"""

        assembly_processing_errors.labels(error_type=error_type, stage=stage).inc()

    def _check_slo_compliance(self):
        """检查SLO合规性"""

        for name, slo in self.slo_targets.items():
            if name == "edge_f1":
                slo.is_met = slo.current_value >= slo.target_value
            elif name == "latency_p95":
                # 计算P95延迟
                if self.metrics_buffer:
                    latencies = [m["latency_ms"] for m in self.metrics_buffer[-100:]]
                    latencies.sort()
                    p95_index = int(len(latencies) * 0.95)
                    p95_latency = latencies[p95_index] if latencies else 0
                    slo.current_value = p95_latency
                    slo.is_met = p95_latency <= slo.target_value
            else:
                slo.is_met = slo.current_value >= slo.target_value

    def get_slo_report(self) -> Dict:
        """获取SLO报告"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "slo_summary": {
                "total": len(self.slo_targets),
                "met": sum(1 for s in self.slo_targets.values() if s.is_met),
                "violated": sum(1 for s in self.slo_targets.values() if not s.is_met),
            },
            "details": [],
        }

        for name, slo in self.slo_targets.items():
            report["details"].append(
                {
                    "name": slo.name,
                    "target": slo.target_value,
                    "current": slo.current_value,
                    "is_met": slo.is_met,
                    "severity": slo.severity,
                    "gap": abs(slo.current_value - slo.target_value),
                }
            )

        # 计算总体健康度
        report["health_score"] = report["slo_summary"]["met"] / report["slo_summary"]["total"]

        return report

    def generate_grafana_dashboard(self) -> Dict:
        """生成Grafana仪表板配置"""

        return {
            "dashboard": {
                "title": "Assembly AI Monitoring",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(assembly_requests_total[5m])",
                                "legendFormat": "{{engine}} - {{status}}",
                            }
                        ],
                    },
                    {
                        "title": "Edge F1 Score",
                        "type": "gauge",
                        "targets": [{"expr": "assembly_edge_f1", "legendFormat": "F1 Score"}],
                        "thresholds": [0.5, 0.75, 0.9],
                    },
                    {
                        "title": "Latency Distribution",
                        "type": "heatmap",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, assembly_latency_ms_bucket)",
                                "legendFormat": "p95",
                            }
                        ],
                    },
                    {
                        "title": "Evidence Coverage",
                        "type": "stat",
                        "targets": [
                            {"expr": "assembly_evidence_coverage", "legendFormat": "Coverage"}
                        ],
                    },
                    {
                        "title": "Cache Hit Ratio",
                        "type": "gauge",
                        "targets": [
                            {"expr": "assembly_cache_hit_ratio", "legendFormat": "Hit Ratio"}
                        ],
                    },
                    {
                        "title": "Constraint Fallbacks",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(assembly_constraint_fallbacks_total[5m])",
                                "legendFormat": "{{original_type}} → {{fallback_type}}",
                            }
                        ],
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(assembly_processing_errors_total[5m])",
                                "legendFormat": "{{error_type}} - {{stage}}",
                            }
                        ],
                    },
                    {
                        "title": "SLO Compliance",
                        "type": "table",
                        "targets": [{"expr": "assembly_slo_compliance", "format": "table"}],
                    },
                ],
            }
        }


class SecurityValidator:
    """安全验证器"""

    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_parts = 1000
        self.max_mates = 5000
        self.max_recursion_depth = 10
        self.timeout = 60  # seconds

    def validate_input(self, file_data: bytes, file_type: str) -> Tuple[bool, Optional[str]]:
        """验证输入文件安全性"""

        # 检查文件大小
        if len(file_data) > self.max_file_size:
            return False, f"File too large: {len(file_data)} bytes (max: {self.max_file_size})"

        # 检查zip bomb
        if file_type in ["zip", "gz", "tar"]:
            if self._check_compression_ratio(file_data) > 100:
                return False, "Potential zip bomb detected"

        # 检查STEP/DXF特定威胁
        if file_type == "step":
            if self._check_step_recursion(file_data):
                return False, "Recursive reference detected in STEP file"

        return True, None

    def validate_assembly_limits(self, assembly: Dict) -> Tuple[bool, Optional[str]]:
        """验证装配体限制"""

        parts = assembly.get("parts", [])
        mates = assembly.get("mates", [])

        if len(parts) > self.max_parts:
            return False, f"Too many parts: {len(parts)} (max: {self.max_parts})"

        if len(mates) > self.max_mates:
            return False, f"Too many mates: {len(mates)} (max: {self.max_mates})"

        # 检查循环引用
        if self._check_circular_references(mates):
            return False, "Circular references detected in assembly"

        return True, None

    def _check_compression_ratio(self, compressed_data: bytes) -> float:
        """检查压缩率（简化实现）"""
        # 实际应该解压一小部分来检查
        return 1.0

    def _check_step_recursion(self, step_data: bytes) -> bool:
        """检查STEP文件递归引用"""
        # 简化实现
        return False

    def _check_circular_references(self, mates: List[Dict]) -> bool:
        """检查循环引用"""
        # 构建图并检查循环
        import networkx as nx

        G = nx.DiGraph()
        for mate in mates:
            G.add_edge(mate.get("part1"), mate.get("part2"))

        try:
            cycles = list(nx.simple_cycles(G))
            return len(cycles) > 0
        except Exception:
            return False


# 使用示例
if __name__ == "__main__":
    # 创建监控器
    monitor = AssemblyMetricsMonitor()

    # 模拟记录指标
    monitor.record_request("urdf", "success", 1250.5)
    monitor.record_analysis_quality(0.82, 0.91, [0.7, 0.8, 0.9, 0.95], "v1.0.0")
    monitor.record_cache_metrics(hits=80, misses=20)
    monitor.record_constraint_fallback("gear", "coupled_revolute", "urdf")
    monitor.record_simulation_result("pybullet", True)

    # 获取SLO报告
    slo_report = monitor.get_slo_report()
    print("SLO Report:")
    print(json.dumps(slo_report, indent=2, default=str))

    # 安全验证
    validator = SecurityValidator()

    # 测试装配体
    test_assembly = {
        "parts": [{"id": f"part_{i}"} for i in range(10)],
        "mates": [{"part1": "part_0", "part2": f"part_{i}"} for i in range(1, 10)],
    }

    is_valid, message = validator.validate_assembly_limits(test_assembly)
    print(f"\nAssembly validation: {is_valid}")
    if message:
        print(f"  Message: {message}")
