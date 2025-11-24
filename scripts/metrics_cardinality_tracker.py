#!/usr/bin/env python3
"""
Metrics Cardinality Tracker
实时追踪和分析Prometheus指标基数，预测存储成本
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin
import requests
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CardinalityInfo:
    """指标基数信息"""
    metric_name: str
    cardinality: int  # 时间序列数
    labels: Dict[str, int]  # label名 -> 唯一值数量
    sample_labels: List[Dict[str, str]]  # 样本label组合
    bytes_per_sample: int = 16  # 每个样本的存储字节数
    retention_days: int = 15  # 数据保留天数
    samples_per_day: int = 8640  # 每天采样数（10s间隔）

    @property
    def storage_bytes(self) -> int:
        """计算存储占用（字节）"""
        return self.cardinality * self.bytes_per_sample * self.samples_per_day * self.retention_days

    @property
    def storage_mb(self) -> float:
        """存储占用（MB）"""
        return self.storage_bytes / (1024 * 1024)

    @property
    def monthly_cost(self) -> float:
        """月度存储成本（假设 $0.001 per MB per month）"""
        return self.storage_mb * 0.001


@dataclass
class TrendInfo:
    """基数趋势信息"""
    metric_name: str
    current_cardinality: int
    previous_cardinality: int
    growth_rate: float  # 增长率百分比
    growth_absolute: int  # 绝对增长量
    timestamp: str

    @property
    def is_growing(self) -> bool:
        return self.growth_rate > 0

    @property
    def is_exploding(self) -> bool:
        """是否爆炸式增长（>50%）"""
        return self.growth_rate > 50


@dataclass
class LabelAnalysis:
    """Label分析结果"""
    label_name: str
    unique_values: int
    entropy: float  # 信息熵，越高说明分布越均匀
    top_values: List[Tuple[str, int]]  # (值, 出现次数)
    recommendation: str  # 优化建议


class PrometheusClient:
    """Prometheus客户端"""

    def __init__(self, url: str, timeout: int = 30):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def query(self, query: str) -> Dict[str, Any]:
        """执行Prometheus查询"""
        endpoint = urljoin(self.url, '/api/v1/query')
        params = {'query': query}

        try:
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'success':
                raise ValueError(f"Query failed: {data.get('error', 'Unknown error')}")

            return data.get('data', {})
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            raise

    def get_all_metrics(self) -> List[str]:
        """获取所有指标名称"""
        endpoint = urljoin(self.url, '/api/v1/label/__name__/values')

        try:
            response = self.session.get(endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'success':
                return data.get('data', [])
            return []
        except Exception as e:
            logger.error(f"Failed to get metrics list: {e}")
            return []


class MetricsCardinalityTracker:
    """指标基数追踪器"""

    def __init__(self, prometheus_url: str):
        self.prom_client = PrometheusClient(prometheus_url)
        self.cardinality_cache = {}
        self.history = defaultdict(list)  # 历史记录

    def get_metric_cardinality(self, metric_name: str) -> Optional[CardinalityInfo]:
        """
        获取指标的基数信息

        Args:
            metric_name: 指标名称

        Returns:
            CardinalityInfo对象，包含基数和成本信息
        """
        try:
            # 查询该指标的所有时间序列
            query = f'group by(__name__, {metric_name}) ({{__name__="{metric_name}"}})'
            # 简化查询：直接获取指标
            query = f'{metric_name}'

            result = self.prom_client.query(query)

            if not result or 'result' not in result:
                return None

            series_list = result['result']
            cardinality = len(series_list)

            # 分析labels
            label_stats = defaultdict(set)
            sample_labels = []

            for series in series_list[:100]:  # 采样前100个
                labels = series.get('metric', {})
                sample_labels.append(labels)

                for key, value in labels.items():
                    if key != '__name__':
                        label_stats[key].add(value)

            # 转换为label计数
            labels_count = {
                key: len(values) for key, values in label_stats.items()
            }

            info = CardinalityInfo(
                metric_name=metric_name,
                cardinality=cardinality,
                labels=labels_count,
                sample_labels=sample_labels[:10]  # 保留10个样本
            )

            # 缓存结果
            self.cardinality_cache[metric_name] = info

            # 记录历史
            self.history[metric_name].append({
                'timestamp': datetime.now().isoformat(),
                'cardinality': cardinality
            })

            return info

        except Exception as e:
            logger.error(f"Failed to get cardinality for {metric_name}: {e}")
            return None

    def track_cardinality_trends(self, metrics: Optional[List[str]] = None) -> List[TrendInfo]:
        """
        追踪指标基数变化趋势

        Args:
            metrics: 要追踪的指标列表，None表示所有指标

        Returns:
            趋势信息列表
        """
        if metrics is None:
            metrics = self.prom_client.get_all_metrics()

        trends = []

        for metric in metrics:
            try:
                current_info = self.get_metric_cardinality(metric)
                if not current_info:
                    continue

                # 从历史记录获取之前的基数
                history = self.history.get(metric, [])
                if len(history) < 2:
                    continue

                previous_cardinality = history[-2]['cardinality']
                current_cardinality = current_info.cardinality

                # 计算增长
                growth_absolute = current_cardinality - previous_cardinality
                growth_rate = (growth_absolute / max(previous_cardinality, 1)) * 100

                trend = TrendInfo(
                    metric_name=metric,
                    current_cardinality=current_cardinality,
                    previous_cardinality=previous_cardinality,
                    growth_rate=round(growth_rate, 2),
                    growth_absolute=growth_absolute,
                    timestamp=datetime.now().isoformat()
                )

                trends.append(trend)

            except Exception as e:
                logger.error(f"Failed to track trend for {metric}: {e}")

        # 按增长率排序
        trends.sort(key=lambda x: abs(x.growth_rate), reverse=True)

        return trends

    def identify_high_cardinality_labels(self, metric_name: str, threshold: int = 100) -> List[LabelAnalysis]:
        """
        识别高基数的labels

        Args:
            metric_name: 指标名称
            threshold: 高基数阈值

        Returns:
            Label分析结果列表
        """
        info = self.cardinality_cache.get(metric_name) or self.get_metric_cardinality(metric_name)

        if not info:
            return []

        analyses = []

        for label_name, unique_count in info.labels.items():
            # 计算该label的值分布
            value_counts = defaultdict(int)
            for sample in info.sample_labels:
                if label_name in sample:
                    value_counts[sample[label_name]] += 1

            # 计算信息熵
            total = sum(value_counts.values())
            entropy = 0
            if total > 0:
                for count in value_counts.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * (p if p == 1 else p * (1 - p))

            # 获取top值
            top_values = sorted(
                value_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # 生成优化建议
            recommendation = self._generate_label_recommendation(
                label_name, unique_count, entropy, threshold
            )

            analysis = LabelAnalysis(
                label_name=label_name,
                unique_values=unique_count,
                entropy=round(entropy, 3),
                top_values=top_values,
                recommendation=recommendation
            )

            analyses.append(analysis)

        # 按唯一值数量排序
        analyses.sort(key=lambda x: x.unique_values, reverse=True)

        return analyses

    def _generate_label_recommendation(self, label_name: str, unique_count: int,
                                      entropy: float, threshold: int) -> str:
        """生成label优化建议"""
        recommendations = []

        if unique_count > threshold * 10:
            recommendations.append(f"⚠️ 极高基数({unique_count})，建议删除或使用recording rule")
        elif unique_count > threshold:
            recommendations.append(f"⚡ 高基数({unique_count})，建议减少粒度或合并值")

        if entropy > 0.9:
            recommendations.append("📊 分布均匀，可能是ID类型，建议移除")
        elif entropy < 0.3:
            recommendations.append("📈 分布集中，可以考虑只保留top值")

        # 特定label的建议
        if 'id' in label_name.lower() or 'uuid' in label_name.lower():
            recommendations.append("🆔 ID类型label，强烈建议移除")
        elif 'user' in label_name.lower():
            recommendations.append("👤 用户相关label，建议采样或聚合")
        elif 'path' in label_name.lower() or 'url' in label_name.lower():
            recommendations.append("🔗 路径类label，建议规范化或分组")

        return " | ".join(recommendations) if recommendations else "✅ 基数合理"

    def estimate_total_cost(self) -> Dict[str, Any]:
        """估算总体成本"""
        total_cardinality = 0
        total_storage_mb = 0
        total_monthly_cost = 0
        metrics_count = 0

        top_cost_metrics = []

        for metric_name, info in self.cardinality_cache.items():
            total_cardinality += info.cardinality
            total_storage_mb += info.storage_mb
            total_monthly_cost += info.monthly_cost
            metrics_count += 1

            top_cost_metrics.append({
                'metric': metric_name,
                'cardinality': info.cardinality,
                'storage_mb': round(info.storage_mb, 2),
                'monthly_cost': round(info.monthly_cost, 4)
            })

        # 按成本排序，取top 10
        top_cost_metrics.sort(key=lambda x: x['monthly_cost'], reverse=True)

        return {
            'total_metrics': metrics_count,
            'total_cardinality': total_cardinality,
            'total_storage_mb': round(total_storage_mb, 2),
            'total_monthly_cost': round(total_monthly_cost, 2),
            'average_cardinality': round(total_cardinality / max(metrics_count, 1), 2),
            'top_cost_metrics': top_cost_metrics[:10]
        }

    def export_cardinality_data(self, output_file: str = None) -> str:
        """导出基数数据"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }

        for metric_name, info in self.cardinality_cache.items():
            data['metrics'][metric_name] = {
                'cardinality': info.cardinality,
                'labels': info.labels,
                'storage_mb': round(info.storage_mb, 2),
                'monthly_cost': round(info.monthly_cost, 4)
            }

        # 添加汇总
        data['summary'] = self.estimate_total_cost()

        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str


def main():
    """主函数 - 用于测试"""
    import argparse

    parser = argparse.ArgumentParser(description='Prometheus Metrics Cardinality Tracker')
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                       help='Prometheus server URL')
    parser.add_argument('--metric', help='Specific metric to analyze')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top metrics to show')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--analyze-labels', action='store_true',
                       help='Analyze high cardinality labels')

    args = parser.parse_args()

    # 创建追踪器
    tracker = MetricsCardinalityTracker(args.prometheus_url)

    if args.metric:
        # 分析特定指标
        info = tracker.get_metric_cardinality(args.metric)
        if info:
            print(f"\n📊 Metric: {info.metric_name}")
            print(f"   Cardinality: {info.cardinality:,}")
            print(f"   Storage: {info.storage_mb:.2f} MB")
            print(f"   Monthly Cost: ${info.monthly_cost:.4f}")
            print(f"   Labels: {json.dumps(info.labels, indent=4)}")

            if args.analyze_labels:
                print("\n🏷️ Label Analysis:")
                analyses = tracker.identify_high_cardinality_labels(args.metric)
                for analysis in analyses[:5]:
                    print(f"\n   Label: {analysis.label_name}")
                    print(f"   Unique Values: {analysis.unique_values}")
                    print(f"   Entropy: {analysis.entropy}")
                    print(f"   Recommendation: {analysis.recommendation}")
        else:
            print(f"❌ Failed to analyze metric: {args.metric}")
    else:
        # 分析所有指标
        print("🔍 Analyzing all metrics...")
        metrics = tracker.prom_client.get_all_metrics()

        for metric in metrics[:args.top]:
            tracker.get_metric_cardinality(metric)

        # 显示成本汇总
        cost_summary = tracker.estimate_total_cost()
        print(f"\n💰 Cost Summary:")
        print(f"   Total Metrics: {cost_summary['total_metrics']}")
        print(f"   Total Cardinality: {cost_summary['total_cardinality']:,}")
        print(f"   Total Storage: {cost_summary['total_storage_mb']:.2f} MB")
        print(f"   Total Monthly Cost: ${cost_summary['total_monthly_cost']:.2f}")

        print(f"\n📈 Top {args.top} Cost Metrics:")
        for item in cost_summary['top_cost_metrics']:
            print(f"   {item['metric']}: {item['cardinality']:,} series, "
                  f"${item['monthly_cost']:.4f}/month")

    # 导出数据
    if args.output:
        tracker.export_cardinality_data(args.output)
        print(f"\n✅ Data exported to: {args.output}")


if __name__ == "__main__":
    main()