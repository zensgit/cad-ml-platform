#!/usr/bin/env python3
"""
Metrics Auto Optimizer
自动优化高基数指标，生成优化配置
"""

import json
import yaml
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib

# 导入相关模块
try:
    from .metrics_cardinality_tracker import MetricsCardinalityTracker, CardinalityInfo
except ImportError:  # standalone execution
    from metrics_cardinality_tracker import MetricsCardinalityTracker, CardinalityInfo  # type: ignore
try:
    from .cardinality_analysis_report import CardinalityAnalysisReporter, Recommendation
except ImportError:  # standalone
    from cardinality_analysis_report import CardinalityAnalysisReporter, Recommendation  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationRule:
    """优化规则"""
    rule_type: str  # "merge_labels", "drop_labels", "downsample", "recording_rule"
    pattern: str  # 指标名称模式
    config: Dict[str, Any]  # 规则配置
    priority: int = 5  # 优先级
    enabled: bool = True  # 是否启用


@dataclass
class OptimizationResult:
    """优化结果"""
    metric_name: str
    original_cardinality: int
    optimized_cardinality: int
    reduction_percentage: float
    optimization_type: str
    config_changes: List[Dict[str, Any]]
    rollback_config: Dict[str, Any]  # 回滚配置


@dataclass
class PrometheusConfig:
    """Prometheus配置"""
    global_config: Dict[str, Any] = field(default_factory=dict)
    scrape_configs: List[Dict[str, Any]] = field(default_factory=list)
    recording_rules: List[Dict[str, Any]] = field(default_factory=list)
    metric_relabel_configs: List[Dict[str, Any]] = field(default_factory=list)


class MetricsAutoOptimizer:
    """指标自动优化器"""

    def __init__(self, optimization_rules: Optional[List[OptimizationRule]] = None):
        self.rules = optimization_rules or self._get_default_rules()
        self.optimization_history = []
        self.rollback_configs = {}

    def _get_default_rules(self) -> List[OptimizationRule]:
        """获取默认优化规则"""
        return [
            # 合并用户代理标签
            OptimizationRule(
                rule_type="merge_labels",
                pattern="http_.*",
                config={
                    "label_mappings": {
                        "user_agent": "browser_family",
                        "detailed_path": "path_category"
                    }
                },
                priority=7
            ),
            # 删除高基数标签
            OptimizationRule(
                rule_type="drop_labels",
                pattern=".*",
                config={
                    "labels_to_drop": [
                        "request_id", "session_id", "trace_id",
                        "user_id", "uuid", "correlation_id"
                    ]
                },
                priority=9
            ),
            # 降采样高基数指标
            OptimizationRule(
                rule_type="downsample",
                pattern=".*",
                config={
                    "cardinality_threshold": 5000,
                    "retention_policy": {
                        "raw": "1h",
                        "5m": "1d",
                        "1h": "7d",
                        "1d": "30d"
                    }
                },
                priority=5
            ),
            # 创建recording rules
            OptimizationRule(
                rule_type="recording_rule",
                pattern="http_request_duration_seconds.*",
                config={
                    "aggregations": ["p50", "p95", "p99"],
                    "group_by": ["service", "method", "status_class"]
                },
                priority=8
            )
        ]

    def apply_label_merging(self, metric_info: CardinalityInfo,
                          label_mappings: Dict[str, str]) -> OptimizationResult:
        """
        应用标签合并

        Args:
            metric_info: 指标信息
            label_mappings: 标签映射关系

        Returns:
            优化结果
        """
        original_cardinality = metric_info.cardinality
        estimated_reduction = 0
        config_changes = []

        for old_label, new_label in label_mappings.items():
            if old_label in metric_info.labels:
                unique_values = metric_info.labels[old_label]
                # 估算合并后的基数减少
                estimated_reduction += unique_values * 0.7  # 假设减少70%

                config_change = {
                    "action": "relabel",
                    "source_labels": [old_label],
                    "target_label": new_label,
                    "regex": "(.*)",
                    "replacement": self._generate_mapping_function(old_label, new_label)
                }
                config_changes.append(config_change)

        optimized_cardinality = max(1, int(original_cardinality - estimated_reduction))
        reduction_percentage = ((original_cardinality - optimized_cardinality) /
                               original_cardinality * 100)

        # 生成回滚配置
        rollback_config = {
            "metric": metric_info.metric_name,
            "original_labels": list(label_mappings.keys()),
            "timestamp": datetime.now().isoformat()
        }

        return OptimizationResult(
            metric_name=metric_info.metric_name,
            original_cardinality=original_cardinality,
            optimized_cardinality=optimized_cardinality,
            reduction_percentage=round(reduction_percentage, 2),
            optimization_type="label_merging",
            config_changes=config_changes,
            rollback_config=rollback_config
        )

    def _generate_mapping_function(self, old_label: str, new_label: str) -> str:
        """生成标签映射函数"""
        # 简化的映射逻辑
        if "user_agent" in old_label:
            return "${1:browser_family}"
        elif "path" in old_label:
            return "${1:path_category}"
        else:
            return "${1}"

    def apply_label_dropping(self, metric_info: CardinalityInfo,
                           labels_to_drop: List[str]) -> OptimizationResult:
        """
        应用标签删除

        Args:
            metric_info: 指标信息
            labels_to_drop: 要删除的标签列表

        Returns:
            优化结果
        """
        original_cardinality = metric_info.cardinality
        estimated_reduction = 0
        config_changes = []

        for label in labels_to_drop:
            if label in metric_info.labels:
                unique_values = metric_info.labels[label]
                # 删除高基数标签可以大幅减少基数
                estimated_reduction += unique_values * 0.9

                config_change = {
                    "action": "labeldrop",
                    "regex": label
                }
                config_changes.append(config_change)

        optimized_cardinality = max(1, int(original_cardinality - estimated_reduction))
        reduction_percentage = ((original_cardinality - optimized_cardinality) /
                               original_cardinality * 100)

        # 生成回滚配置
        rollback_config = {
            "metric": metric_info.metric_name,
            "dropped_labels": labels_to_drop,
            "timestamp": datetime.now().isoformat()
        }

        return OptimizationResult(
            metric_name=metric_info.metric_name,
            original_cardinality=original_cardinality,
            optimized_cardinality=optimized_cardinality,
            reduction_percentage=round(reduction_percentage, 2),
            optimization_type="label_dropping",
            config_changes=config_changes,
            rollback_config=rollback_config
        )

    def apply_downsampling(self, metric_info: CardinalityInfo,
                         retention_policy: Dict[str, str]) -> Dict[str, Any]:
        """
        应用降采样策略

        Args:
            metric_info: 指标信息
            retention_policy: 保留策略

        Returns:
            降采样配置
        """
        # 生成Prometheus recording rules
        recording_rules = []

        for interval, retention in retention_policy.items():
            if interval == "raw":
                continue

            rule = {
                "record": f"{metric_info.metric_name}:{interval}",
                "expr": f"avg_over_time({metric_info.metric_name}[{interval}])",
                "labels": {
                    "aggregation": interval
                }
            }
            recording_rules.append(rule)

        # 生成远程存储配置
        remote_write_config = {
            "url": "http://prometheus-long-term:9090/api/v1/write",
            "write_relabel_configs": [
                {
                    "source_labels": ["__name__"],
                    "regex": f"{metric_info.metric_name}",
                    "target_label": "__tmp_retention",
                    "replacement": retention_policy.get("raw", "1h")
                }
            ]
        }

        return {
            "metric": metric_info.metric_name,
            "recording_rules": recording_rules,
            "remote_write_config": remote_write_config,
            "estimated_reduction": 50  # 预计减少50%存储
        }

    def create_recording_rules(self, metric_info: CardinalityInfo,
                             aggregations: List[str],
                             group_by: List[str]) -> List[Dict[str, Any]]:
        """
        创建recording rules

        Args:
            metric_info: 指标信息
            aggregations: 聚合类型列表
            group_by: 分组标签列表

        Returns:
            Recording rules列表
        """
        rules = []

        for agg in aggregations:
            if agg in ["p50", "p95", "p99"]:
                # 百分位数
                quantile = float(agg[1:]) / 100
                expr = f'histogram_quantile({quantile}, ' \
                       f'sum(rate({metric_info.metric_name}[5m])) by ({", ".join(group_by)}, le))'
                record_name = f'{metric_info.metric_name}:{agg}:5m'
            elif agg == "rate":
                # 速率
                expr = f'sum(rate({metric_info.metric_name}[5m])) by ({", ".join(group_by)})'
                record_name = f'{metric_info.metric_name}:rate:5m'
            elif agg == "sum":
                # 总和
                expr = f'sum({metric_info.metric_name}) by ({", ".join(group_by)})'
                record_name = f'{metric_info.metric_name}:sum'
            else:
                continue

            rule = {
                "record": record_name,
                "expr": expr,
                "labels": {
                    "aggregation": agg
                }
            }
            rules.append(rule)

        return rules

    def optimize_metrics(self, tracker: MetricsCardinalityTracker,
                        recommendations: List[Recommendation]) -> List[OptimizationResult]:
        """
        批量优化指标

        Args:
            tracker: 基数追踪器
            recommendations: 优化建议列表

        Returns:
            优化结果列表
        """
        results = []

        for rec in recommendations:
            if rec.metric_name == "GLOBAL":
                continue

            metric_info = tracker.cardinality_cache.get(rec.metric_name)
            if not metric_info:
                continue

            # 根据建议类型应用优化
            if "Remove ID label" in rec.action or "remove" in rec.action.lower():
                # 提取要删除的标签
                labels_to_drop = self._extract_labels_from_action(rec.action)
                if labels_to_drop:
                    result = self.apply_label_dropping(metric_info, labels_to_drop)
                    results.append(result)

            elif "Reduce granularity" in rec.action or "merge" in rec.action.lower():
                # 应用标签合并
                label_mappings = self._generate_label_mappings(metric_info)
                if label_mappings:
                    result = self.apply_label_merging(metric_info, label_mappings)
                    results.append(result)

            elif "downsampling" in rec.action.lower():
                # 应用降采样
                config = self.apply_downsampling(metric_info, {"raw": "1h", "5m": "1d"})
                # 转换为OptimizationResult
                result = OptimizationResult(
                    metric_name=metric_info.metric_name,
                    original_cardinality=metric_info.cardinality,
                    optimized_cardinality=int(metric_info.cardinality * 0.5),
                    reduction_percentage=50.0,
                    optimization_type="downsampling",
                    config_changes=[config],
                    rollback_config={"metric": metric_info.metric_name}
                )
                results.append(result)

        # 记录优化历史
        self.optimization_history.extend(results)

        return results

    def _extract_labels_from_action(self, action: str) -> List[str]:
        """从建议动作中提取标签名"""
        labels = []
        # 查找引号中的标签名
        matches = re.findall(r"'([^']+)'", action)
        for match in matches:
            if "_" in match or match.lower() in ["id", "uuid", "user"]:
                labels.append(match)
        return labels

    def _generate_label_mappings(self, metric_info: CardinalityInfo) -> Dict[str, str]:
        """生成标签映射关系"""
        mappings = {}
        for label, unique_count in metric_info.labels.items():
            if unique_count > 100:
                # 为高基数标签生成映射
                if "user" in label:
                    mappings[label] = "user_category"
                elif "path" in label or "url" in label:
                    mappings[label] = "path_pattern"
                elif "agent" in label:
                    mappings[label] = "agent_type"
        return mappings

    def generate_prometheus_config(self, optimizations: List[OptimizationResult]) -> str:
        """
        生成Prometheus配置

        Args:
            optimizations: 优化结果列表

        Returns:
            YAML格式的配置
        """
        config = {
            'global': {
                'evaluation_interval': '30s'
            },
            'metric_relabel_configs': [],
            'rule_files': ['recording_rules.yml']
        }

        # 收集所有配置变更
        for opt in optimizations:
            for change in opt.config_changes:
                if 'action' in change:
                    config['metric_relabel_configs'].append(change)

        # 生成recording rules文件内容
        recording_rules = {
            'groups': [
                {
                    'name': 'cardinality_optimization',
                    'interval': '30s',
                    'rules': []
                }
            ]
        }

        for opt in optimizations:
            if opt.optimization_type == "recording_rule":
                for change in opt.config_changes:
                    if 'recording_rules' in change:
                        recording_rules['groups'][0]['rules'].extend(
                            change['recording_rules']
                        )

        # 生成YAML
        prometheus_config = yaml.dump(config, default_flow_style=False)
        recording_rules_config = yaml.dump(recording_rules, default_flow_style=False)

        return f"# prometheus.yml additions\n{prometheus_config}\n\n" \
               f"# recording_rules.yml\n{recording_rules_config}"

    def generate_optimization_pr(self, optimizations: List[OptimizationResult],
                                repo_path: str = ".") -> Dict[str, Any]:
        """
        生成优化PR

        Args:
            optimizations: 优化结果列表
            repo_path: 仓库路径

        Returns:
            PR信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"metrics-optimization-{timestamp}"

        # 生成配置文件
        config = self.generate_prometheus_config(optimizations)

        # 计算总体影响
        total_reduction = sum(opt.original_cardinality - opt.optimized_cardinality
                            for opt in optimizations)
        total_percentage = sum(opt.reduction_percentage for opt in optimizations) / len(optimizations)

        # 生成PR描述
        pr_description = f"""
## 📊 Metrics Cardinality Optimization

### Summary
- **Metrics Optimized**: {len(optimizations)}
- **Total Series Reduction**: {total_reduction:,}
- **Average Reduction**: {total_percentage:.1f}%
- **Estimated Monthly Savings**: ${total_reduction * 0.001:.2f}

### Optimizations Applied

| Metric | Type | Original | Optimized | Reduction |
|--------|------|----------|-----------|-----------|
"""

        for opt in optimizations[:10]:  # 显示前10个
            pr_description += f"| {opt.metric_name} | {opt.optimization_type} | " \
                            f"{opt.original_cardinality:,} | {opt.optimized_cardinality:,} | " \
                            f"{opt.reduction_percentage:.1f}% |\n"

        pr_description += """

### Configuration Changes

The following configuration changes have been applied:
1. Label consolidation for high-cardinality labels
2. Removal of unnecessary ID-type labels
3. Recording rules for common aggregations
4. Downsampling policies for historical data

### Testing

- [ ] Configuration syntax validated
- [ ] Test environment deployment successful
- [ ] No impact on existing dashboards
- [ ] Alerts continue to function correctly

### Rollback Plan

If issues are encountered, the optimization can be rolled back by:
1. Reverting this PR
2. Restarting Prometheus with the previous configuration
3. Restoration scripts are available in `/rollback/`

---
*Generated by Metrics Auto Optimizer*
"""

        # 创建文件变更
        files_changed = {
            'prometheus/prometheus.yml': config,
            'prometheus/recording_rules.yml': config,
            f'rollback/rollback_{timestamp}.json': json.dumps({
                'optimizations': [asdict(opt) for opt in optimizations],
                'timestamp': datetime.now().isoformat()
            }, indent=2)
        }

        return {
            'branch_name': branch_name,
            'pr_title': f'[Auto] Optimize metrics cardinality - Save ${total_reduction * 0.001:.0f}/month',
            'pr_description': pr_description,
            'files_changed': files_changed,
            'labels': ['metrics', 'optimization', 'automated'],
            'reviewers': ['platform-team']
        }

    def rollback_optimization(self, rollback_config: Dict[str, Any]) -> bool:
        """
        回滚优化

        Args:
            rollback_config: 回滚配置

        Returns:
            是否成功
        """
        try:
            # 这里应该实现实际的回滚逻辑
            # 例如恢复原始的Prometheus配置
            logger.info(f"Rolling back optimization for {rollback_config.get('metric', 'unknown')}")

            # 从历史中移除
            self.optimization_history = [
                opt for opt in self.optimization_history
                if opt.metric_name != rollback_config.get('metric')
            ]

            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def main():
    """主函数 - 用于测试"""
    import argparse

    parser = argparse.ArgumentParser(description='Metrics Auto Optimizer')
    parser.add_argument('--prometheus-url', default='http://localhost:9090')
    parser.add_argument('--apply', action='store_true', help='Apply optimizations')
    parser.add_argument('--generate-config', action='store_true', help='Generate config only')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--pr', action='store_true', help='Generate PR')

    args = parser.parse_args()

    # 创建组件
    tracker = MetricsCardinalityTracker(args.prometheus_url)
    reporter = CardinalityAnalysisReporter(tracker)
    optimizer = MetricsAutoOptimizer()

    print("🔧 Metrics Auto Optimizer")

    # 生成优化建议
    print("\n📊 Generating optimization recommendations...")
    recommendations = reporter.generate_optimization_recommendations()

    if not recommendations:
        print("✅ No optimizations needed!")
        return

    print(f"\n Found {len(recommendations)} optimization opportunities")

    if args.apply:
        # 应用优化
        print("\n🚀 Applying optimizations...")
        results = optimizer.optimize_metrics(tracker, recommendations)

        print(f"\n✅ Applied {len(results)} optimizations:")
        total_savings = 0
        for result in results:
            print(f"   - {result.metric_name}: {result.reduction_percentage:.1f}% reduction")
            total_savings += (result.original_cardinality - result.optimized_cardinality) * 0.001

        print(f"\n💰 Total monthly savings: ${total_savings:.2f}")

        if args.generate_config:
            # 生成配置
            config = optimizer.generate_prometheus_config(results)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(config)
                print(f"\n📝 Configuration saved to: {args.output}")
            else:
                print("\n📝 Generated configuration:")
                print(config)

        if args.pr:
            # 生成PR
            pr_info = optimizer.generate_optimization_pr(results)
            print(f"\n🔀 PR Information:")
            print(f"   Branch: {pr_info['branch_name']}")
            print(f"   Title: {pr_info['pr_title']}")
            print(f"   Files: {len(pr_info['files_changed'])}")

            if args.output:
                pr_file = Path(args.output).with_suffix('.pr.json')
                with open(pr_file, 'w') as f:
                    json.dump(pr_info, f, indent=2)
                print(f"   PR details saved to: {pr_file}")

    else:
        # 只显示建议
        print("\n💡 Top optimization recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. {rec.metric_name}")
            print(f"   Action: {rec.action}")
            print(f"   Priority: {rec.priority}/10")
            print(f"   Expected Savings: ${rec.expected_cost_reduction:.2f}/month")
            print(f"   Difficulty: {rec.implementation_difficulty}")


if __name__ == "__main__":
    main()
