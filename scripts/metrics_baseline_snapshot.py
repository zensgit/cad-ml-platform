#!/usr/bin/env python3
"""
Metrics Baseline Snapshot Tool
指标基线快照工具 - 建立指标基线用于漂移检测
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import subprocess
import numpy as np

# Prometheus client
try:
    import requests
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("Warning: requests/prometheus_client not installed, using mock mode")
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricBaseline:
    """指标基线"""
    metric_name: str
    labels: List[str]
    label_values: Dict[str, Set[str]]
    cardinality: int
    value_stats: Dict[str, float]  # mean, std, min, max, p50, p90, p99
    sample_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BaselineSnapshot:
    """基线快照"""
    version: str
    timestamp: datetime
    environment: str
    metrics: Dict[str, MetricBaseline]
    metadata: Dict[str, Any]
    checksum: str = ""


class MetricsBaselineManager:
    """指标基线管理器"""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.project_root = Path(__file__).parent.parent
        self.baseline_dir = self.project_root / "baselines"
        self.baseline_dir.mkdir(exist_ok=True)

        self.prometheus_url = prometheus_url
        self.current_baseline: Optional[BaselineSnapshot] = None
        self.metrics_data: Dict[str, Any] = {}

    def fetch_current_metrics(self, time_range: str = "1h") -> Dict[str, Any]:
        """获取当前指标数据"""
        print(f"📊 Fetching metrics from last {time_range}...")

        if not PROMETHEUS_AVAILABLE:
            return self._mock_metrics()

        metrics = {}

        try:
            # 获取所有指标名称
            response = requests.get(f"{self.prometheus_url}/api/v1/label/__name__/values")
            if response.status_code != 200:
                print(f"⚠️ Failed to fetch metric names: {response.status_code}")
                return self._mock_metrics()

            metric_names = response.json().get("data", [])

            # 过滤系统指标，只保留应用指标
            app_metrics = [m for m in metric_names if not m.startswith(("go_", "process_", "promhttp_"))]

            for metric_name in app_metrics:
                # 获取指标数据
                query = f'{metric_name}[{time_range}]'
                response = requests.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={"query": query}
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        metrics[metric_name] = self._parse_metric_data(metric_name, data)

        except Exception as e:
            print(f"⚠️ Error fetching metrics: {e}")
            return self._mock_metrics()

        self.metrics_data = metrics
        return metrics

    def _parse_metric_data(self, metric_name: str, data: Dict) -> Dict[str, Any]:
        """解析指标数据"""
        result = data.get("data", {}).get("result", [])

        labels = set()
        label_values = defaultdict(set)
        values = []

        for series in result:
            # 提取标签
            metric_labels = series.get("metric", {})
            for label, value in metric_labels.items():
                if label != "__name__":
                    labels.add(label)
                    label_values[label].add(value)

            # 提取值
            series_values = series.get("values", [])
            for timestamp, value in series_values:
                try:
                    values.append(float(value))
                except:
                    pass

        return {
            "labels": list(labels),
            "label_values": {k: list(v) for k, v in label_values.items()},
            "cardinality": len(result),
            "values": values,
            "sample_count": len(values)
        }

    def _mock_metrics(self) -> Dict[str, Any]:
        """生成模拟指标数据"""
        print("📦 Using mock metrics data...")

        mock_data = {
            "http_requests_total": {
                "labels": ["method", "status", "endpoint"],
                "label_values": {
                    "method": ["GET", "POST", "PUT", "DELETE"],
                    "status": ["200", "201", "400", "404", "500"],
                    "endpoint": ["/api/v1/analyze", "/api/v1/ocr", "/health", "/metrics"]
                },
                "cardinality": 80,
                "values": np.random.exponential(100, 1000).tolist(),
                "sample_count": 1000
            },
            "ocr_processing_duration": {
                "labels": ["provider", "model", "status"],
                "label_values": {
                    "provider": ["tesseract", "easyocr", "paddleocr"],
                    "model": ["default", "fast", "accurate"],
                    "status": ["success", "failure"]
                },
                "cardinality": 18,
                "values": np.random.gamma(2, 0.5, 500).tolist(),
                "sample_count": 500
            },
            "vision_analysis_requests": {
                "labels": ["provider", "analysis_type"],
                "label_values": {
                    "provider": ["openai", "anthropic", "google"],
                    "analysis_type": ["object_detection", "scene_understanding", "text_extraction"]
                },
                "cardinality": 9,
                "values": np.random.normal(50, 10, 300).tolist(),
                "sample_count": 300
            },
            "resilience_circuit_state": {
                "labels": ["service", "endpoint"],
                "label_values": {
                    "service": ["ocr", "vision", "api"],
                    "endpoint": ["process", "analyze", "health"]
                },
                "cardinality": 9,
                "values": [0] * 100 + [1] * 5 + [2] * 2,  # Mostly closed
                "sample_count": 107
            },
            "error_code_occurrences": {
                "labels": ["code", "severity"],
                "label_values": {
                    "code": ["OCR001", "OCR002", "VIS001", "VIS002", "API001"],
                    "severity": ["low", "medium", "high", "critical"]
                },
                "cardinality": 20,
                "values": np.random.poisson(5, 200).tolist(),
                "sample_count": 200
            }
        }

        return mock_data

    def create_baseline(self, name: str = "default", environment: str = "production") -> BaselineSnapshot:
        """创建基线快照"""
        print(f"📸 Creating baseline snapshot: {name}")

        # 获取当前指标
        metrics = self.fetch_current_metrics()

        # 创建基线对象
        baselines = {}

        for metric_name, metric_data in metrics.items():
            values = metric_data.get("values", [])

            # 计算统计信息
            if values:
                value_stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p99": float(np.percentile(values, 99))
                }
            else:
                value_stats = {
                    "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "p50": 0.0, "p90": 0.0, "p99": 0.0
                }

            baseline = MetricBaseline(
                metric_name=metric_name,
                labels=metric_data.get("labels", []),
                label_values={k: set(v) for k, v in metric_data.get("label_values", {}).items()},
                cardinality=metric_data.get("cardinality", 0),
                value_stats=value_stats,
                sample_count=metric_data.get("sample_count", 0)
            )

            baselines[metric_name] = baseline

        # 创建快照
        snapshot = BaselineSnapshot(
            version="1.0.0",
            timestamp=datetime.now(),
            environment=environment,
            metrics=baselines,
            metadata={
                "name": name,
                "metrics_count": len(baselines),
                "total_cardinality": sum(b.cardinality for b in baselines.values()),
                "total_samples": sum(b.sample_count for b in baselines.values())
            }
        )

        # 计算校验和
        snapshot.checksum = self._calculate_checksum(snapshot)

        self.current_baseline = snapshot
        return snapshot

    def _calculate_checksum(self, snapshot: BaselineSnapshot) -> str:
        """计算快照校验和"""
        data = {
            "metrics": {
                name: {
                    "labels": sorted(baseline.labels),
                    "cardinality": baseline.cardinality,
                    "stats": baseline.value_stats
                }
                for name, baseline in snapshot.metrics.items()
            },
            "environment": snapshot.environment,
            "version": snapshot.version
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def save_baseline(self, snapshot: Optional[BaselineSnapshot] = None, filename: Optional[str] = None) -> str:
        """保存基线快照"""
        if snapshot is None:
            snapshot = self.current_baseline

        if snapshot is None:
            raise ValueError("No baseline snapshot to save")

        if filename is None:
            timestamp = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_{snapshot.metadata['name']}_{timestamp}.json"

        filepath = self.baseline_dir / filename

        # 序列化快照
        data = {
            "version": snapshot.version,
            "timestamp": snapshot.timestamp.isoformat(),
            "environment": snapshot.environment,
            "checksum": snapshot.checksum,
            "metadata": snapshot.metadata,
            "metrics": {}
        }

        for name, baseline in snapshot.metrics.items():
            data["metrics"][name] = {
                "labels": baseline.labels,
                "label_values": {k: list(v) for k, v in baseline.label_values.items()},
                "cardinality": baseline.cardinality,
                "value_stats": baseline.value_stats,
                "sample_count": baseline.sample_count,
                "timestamp": baseline.timestamp.isoformat()
            }

        # 保存到文件
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Baseline saved to: {filepath}")
        return str(filepath)

    def load_baseline(self, filename: str) -> BaselineSnapshot:
        """加载基线快照"""
        filepath = self.baseline_dir / filename

        if not filepath.exists():
            # 尝试直接路径
            filepath = Path(filename)
            if not filepath.exists():
                raise FileNotFoundError(f"Baseline file not found: {filename}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        # 反序列化
        metrics = {}
        for name, metric_data in data["metrics"].items():
            baseline = MetricBaseline(
                metric_name=name,
                labels=metric_data["labels"],
                label_values={k: set(v) for k, v in metric_data["label_values"].items()},
                cardinality=metric_data["cardinality"],
                value_stats=metric_data["value_stats"],
                sample_count=metric_data["sample_count"],
                timestamp=datetime.fromisoformat(metric_data["timestamp"])
            )
            metrics[name] = baseline

        snapshot = BaselineSnapshot(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            environment=data["environment"],
            metrics=metrics,
            metadata=data["metadata"],
            checksum=data["checksum"]
        )

        # 验证校验和
        calculated = self._calculate_checksum(snapshot)
        if calculated != snapshot.checksum:
            print(f"⚠️ Checksum mismatch! Expected: {snapshot.checksum}, Got: {calculated}")

        self.current_baseline = snapshot
        return snapshot

    def list_baselines(self) -> List[Dict[str, Any]]:
        """列出所有基线"""
        baselines = []

        for filepath in self.baseline_dir.glob("baseline_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                baselines.append({
                    "filename": filepath.name,
                    "name": data["metadata"].get("name", "unknown"),
                    "timestamp": data["timestamp"],
                    "environment": data["environment"],
                    "metrics_count": data["metadata"].get("metrics_count", 0),
                    "checksum": data["checksum"]
                })
            except:
                pass

        # 按时间排序
        baselines.sort(key=lambda x: x["timestamp"], reverse=True)

        return baselines

    def compare_baselines(self, baseline1: BaselineSnapshot, baseline2: BaselineSnapshot) -> Dict[str, Any]:
        """比较两个基线"""
        comparison = {
            "baseline1": {
                "timestamp": baseline1.timestamp.isoformat(),
                "environment": baseline1.environment,
                "metrics_count": len(baseline1.metrics)
            },
            "baseline2": {
                "timestamp": baseline2.timestamp.isoformat(),
                "environment": baseline2.environment,
                "metrics_count": len(baseline2.metrics)
            },
            "differences": {
                "new_metrics": [],
                "removed_metrics": [],
                "changed_metrics": []
            }
        }

        metrics1 = set(baseline1.metrics.keys())
        metrics2 = set(baseline2.metrics.keys())

        # 新增指标
        comparison["differences"]["new_metrics"] = list(metrics2 - metrics1)

        # 删除的指标
        comparison["differences"]["removed_metrics"] = list(metrics1 - metrics2)

        # 变化的指标
        for metric_name in metrics1 & metrics2:
            m1 = baseline1.metrics[metric_name]
            m2 = baseline2.metrics[metric_name]

            changes = []

            # 检查标签变化
            if set(m1.labels) != set(m2.labels):
                changes.append(f"Labels: {m1.labels} → {m2.labels}")

            # 检查基数变化
            cardinality_change = ((m2.cardinality - m1.cardinality) / m1.cardinality * 100) if m1.cardinality > 0 else 0
            if abs(cardinality_change) > 20:
                changes.append(f"Cardinality: {m1.cardinality} → {m2.cardinality} ({cardinality_change:+.1f}%)")

            # 检查值分布变化
            mean_change = ((m2.value_stats["mean"] - m1.value_stats["mean"]) / m1.value_stats["mean"] * 100) if m1.value_stats["mean"] > 0 else 0
            if abs(mean_change) > 50:
                changes.append(f"Mean: {m1.value_stats['mean']:.2f} → {m2.value_stats['mean']:.2f} ({mean_change:+.1f}%)")

            if changes:
                comparison["differences"]["changed_metrics"].append({
                    "metric": metric_name,
                    "changes": changes
                })

        return comparison

    def generate_report(self, baseline: Optional[BaselineSnapshot] = None) -> str:
        """生成基线报告"""
        if baseline is None:
            baseline = self.current_baseline

        if baseline is None:
            return "No baseline loaded"

        lines = []
        lines.append("# Metrics Baseline Report")
        lines.append(f"\n**Environment**: {baseline.environment}")
        lines.append(f"**Timestamp**: {baseline.timestamp.isoformat()}")
        lines.append(f"**Checksum**: {baseline.checksum}\n")

        # 摘要
        lines.append("## 📊 Summary\n")
        lines.append(f"- **Total Metrics**: {baseline.metadata['metrics_count']}")
        lines.append(f"- **Total Cardinality**: {baseline.metadata['total_cardinality']}")
        lines.append(f"- **Total Samples**: {baseline.metadata['total_samples']}\n")

        # 指标详情
        lines.append("## 📈 Metrics Details\n")
        lines.append("| Metric | Labels | Cardinality | Mean | P90 | P99 |")
        lines.append("|--------|--------|-------------|------|-----|-----|")

        for name, metric in sorted(baseline.metrics.items()):
            labels_str = ", ".join(metric.labels[:3])
            if len(metric.labels) > 3:
                labels_str += f" +{len(metric.labels)-3}"

            lines.append(
                f"| {name} | {labels_str} | {metric.cardinality} | "
                f"{metric.value_stats['mean']:.2f} | "
                f"{metric.value_stats['p90']:.2f} | "
                f"{metric.value_stats['p99']:.2f} |"
            )

        # 标签基数Top 10
        lines.append("\n## 🏷️ Label Cardinality Top 10\n")

        cardinality_list = []
        for name, metric in baseline.metrics.items():
            for label, values in metric.label_values.items():
                cardinality_list.append((name, label, len(values)))

        cardinality_list.sort(key=lambda x: x[2], reverse=True)

        lines.append("| Metric | Label | Unique Values |")
        lines.append("|--------|-------|---------------|")

        for metric, label, count in cardinality_list[:10]:
            lines.append(f"| {metric} | {label} | {count} |")

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Metrics baseline snapshot management"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create baseline
    create_parser = subparsers.add_parser("create", help="Create new baseline")
    create_parser.add_argument("--name", default="default", help="Baseline name")
    create_parser.add_argument("--env", default="production", help="Environment")
    create_parser.add_argument("--save", action="store_true", help="Save to file")

    # List baselines
    list_parser = subparsers.add_parser("list", help="List all baselines")

    # Load baseline
    load_parser = subparsers.add_parser("load", help="Load baseline")
    load_parser.add_argument("filename", help="Baseline filename")

    # Compare baselines
    compare_parser = subparsers.add_parser("compare", help="Compare baselines")
    compare_parser.add_argument("baseline1", help="First baseline")
    compare_parser.add_argument("baseline2", help="Second baseline")

    # Report
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--baseline", help="Baseline file")
    report_parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    manager = MetricsBaselineManager()

    if args.command == "create":
        baseline = manager.create_baseline(args.name, args.env)
        if args.save:
            filepath = manager.save_baseline(baseline)
            print(f"Baseline saved: {filepath}")
        else:
            print(manager.generate_report(baseline))

    elif args.command == "list":
        baselines = manager.list_baselines()
        print("\n📚 Available Baselines:\n")
        for b in baselines:
            print(f"- {b['filename']}")
            print(f"  Name: {b['name']}, Env: {b['environment']}")
            print(f"  Created: {b['timestamp']}, Metrics: {b['metrics_count']}\n")

    elif args.command == "load":
        baseline = manager.load_baseline(args.filename)
        print(f"✅ Loaded baseline: {args.filename}")
        print(manager.generate_report(baseline))

    elif args.command == "compare":
        b1 = manager.load_baseline(args.baseline1)
        b2 = manager.load_baseline(args.baseline2)
        comparison = manager.compare_baselines(b1, b2)
        print(json.dumps(comparison, indent=2, default=str))

    elif args.command == "report":
        if args.baseline:
            baseline = manager.load_baseline(args.baseline)
        else:
            baseline = manager.create_baseline()

        report = manager.generate_report(baseline)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to: {args.output}")
        else:
            print(report)

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())