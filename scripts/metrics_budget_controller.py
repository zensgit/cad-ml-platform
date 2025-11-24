#!/usr/bin/env python3
"""
Metrics Budget Controller
控制和管理指标基数预算，防止成本超支
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """预算状态"""
    HEALTHY = "healthy"  # <70% 使用
    WARNING = "warning"  # 70-90% 使用
    CRITICAL = "critical"  # 90-100% 使用
    EXCEEDED = "exceeded"  # >100% 使用


class Decision(Enum):
    """决策类型"""
    ALLOW = "allow"  # 允许
    WARN = "warn"  # 警告但允许
    BLOCK = "block"  # 阻止


@dataclass
class BudgetConfig:
    """预算配置"""
    # 全局预算
    global_max_series: int = 1000000  # 总时间序列上限
    global_max_per_metric: int = 10000  # 单个指标上限
    cost_per_series_per_month: float = 0.001  # 每序列每月成本($)

    # 团队预算（时间序列数）
    team_budgets: Dict[str, int] = field(default_factory=lambda: {
        "platform": 300000,
        "api": 200000,
        "frontend": 100000,
        "ml": 150000
    })

    # 服务预算（时间序列数）
    service_budgets: Dict[str, int] = field(default_factory=lambda: {
        "cad-analyzer": 50000,
        "ocr-service": 30000,
        "api-gateway": 20000,
        "auth-service": 15000
    })

    # 指标优先级（高优先级指标在预算紧张时保留）
    metric_priorities: Dict[str, int] = field(default_factory=lambda: {
        "up": 100,  # 服务存活
        "http_request_duration_seconds": 90,  # 延迟
        "http_requests_total": 85,  # 流量
        "error_rate": 95,  # 错误率
        "cpu_usage": 80,  # 资源使用
        "memory_usage": 80
    })

    # 预算策略
    warning_threshold: float = 0.7  # 警告阈值
    critical_threshold: float = 0.9  # 严重阈值
    auto_block_threshold: float = 1.0  # 自动阻止阈值


@dataclass
class BudgetUsage:
    """预算使用情况"""
    entity: str  # 团队或服务名
    budget: int  # 分配的预算
    used: int  # 已使用
    available: int  # 剩余
    usage_percentage: float  # 使用百分比
    status: BudgetStatus  # 状态
    top_metrics: List[Tuple[str, int]]  # Top消耗指标


@dataclass
class MetricChange:
    """指标变更"""
    metric_name: str
    team: str
    service: str
    estimated_cardinality_change: int  # 预计基数变化
    labels_added: List[str]  # 新增labels
    labels_removed: List[str]  # 删除labels
    reason: str  # 变更原因


@dataclass
class AllocationPlan:
    """预算分配计划"""
    reallocations: Dict[str, Dict[str, int]]  # {from: {to: amount}}
    total_optimized: int  # 优化的总量
    recommendations: List[str]  # 建议


class MetricsBudgetController:
    """指标预算控制器"""

    def __init__(self, budget_config: BudgetConfig = None,
                 state_file: str = "metrics_budget_state.json"):
        self.config = budget_config or BudgetConfig()
        self.state_file = Path(state_file)
        self.current_usage = {}  # 当前使用情况
        self.alerts = []  # 告警列表
        self.decisions = []  # 决策历史

        # 加载状态
        self._load_state()

    def _load_state(self):
        """加载持久化状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_usage = state.get('current_usage', {})
                    self.alerts = state.get('alerts', [])
                    self.decisions = state.get('decisions', [])[-100:]  # 保留最近100条
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        """保存状态"""
        try:
            state = {
                'current_usage': self.current_usage,
                'alerts': self.alerts[-50:],  # 保留最近50条告警
                'decisions': self.decisions[-100:],
                'timestamp': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update_usage(self, usage_data: Dict[str, Dict[str, int]]):
        """
        更新使用情况

        Args:
            usage_data: {team/service: {metric: cardinality}}
        """
        self.current_usage = usage_data
        self._save_state()

    def check_budget(self, entity: str, entity_type: str = "team") -> BudgetUsage:
        """
        检查预算状态

        Args:
            entity: 团队或服务名
            entity_type: "team" 或 "service"

        Returns:
            预算使用情况
        """
        # 获取预算配置
        if entity_type == "team":
            budget = self.config.team_budgets.get(entity, 100000)
        else:
            budget = self.config.service_budgets.get(entity, 10000)

        # 计算当前使用
        usage_data = self.current_usage.get(entity, {})
        used = sum(usage_data.values())
        available = max(0, budget - used)
        usage_percentage = (used / budget * 100) if budget > 0 else 0

        # 确定状态
        if usage_percentage >= 100:
            status = BudgetStatus.EXCEEDED
        elif usage_percentage >= self.config.critical_threshold * 100:
            status = BudgetStatus.CRITICAL
        elif usage_percentage >= self.config.warning_threshold * 100:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.HEALTHY

        # 获取top消耗指标
        top_metrics = sorted(
            usage_data.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        usage = BudgetUsage(
            entity=entity,
            budget=budget,
            used=used,
            available=available,
            usage_percentage=round(usage_percentage, 2),
            status=status,
            top_metrics=top_metrics
        )

        # 生成告警
        if status in [BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED]:
            self._generate_alert(entity, entity_type, usage)

        return usage

    def enforce_budget(self, metric_change: MetricChange) -> Tuple[Decision, str]:
        """
        强制执行预算限制

        Args:
            metric_change: 指标变更请求

        Returns:
            (决策, 原因说明)
        """
        # 检查团队预算
        team_usage = self.check_budget(metric_change.team, "team")
        service_usage = self.check_budget(metric_change.service, "service")

        # 评估变更影响
        impact_assessment = self._assess_change_impact(metric_change)

        # 决策逻辑
        decision, reason = self._make_decision(
            metric_change, team_usage, service_usage, impact_assessment
        )

        # 记录决策
        self.decisions.append({
            'timestamp': datetime.now().isoformat(),
            'metric': metric_change.metric_name,
            'team': metric_change.team,
            'service': metric_change.service,
            'decision': decision.value,
            'reason': reason
        })

        self._save_state()

        return decision, reason

    def _assess_change_impact(self, metric_change: MetricChange) -> Dict[str, Any]:
        """评估变更影响"""
        # 计算预计的成本增加
        cost_increase = (metric_change.estimated_cardinality_change *
                        self.config.cost_per_series_per_month)

        # 评估label变化的影响
        label_impact = "high" if len(metric_change.labels_added) > 3 else "low"

        # 检查指标优先级
        metric_priority = self.config.metric_priorities.get(
            metric_change.metric_name, 50
        )

        return {
            'cost_increase': cost_increase,
            'label_impact': label_impact,
            'metric_priority': metric_priority,
            'cardinality_change': metric_change.estimated_cardinality_change
        }

    def _make_decision(self, metric_change: MetricChange, team_usage: BudgetUsage,
                       service_usage: BudgetUsage, impact: Dict) -> Tuple[Decision, str]:
        """做出决策"""
        reasons = []

        # 检查是否超预算
        if team_usage.status == BudgetStatus.EXCEEDED:
            if impact['metric_priority'] < 80:  # 非高优先级指标
                reasons.append(f"团队{metric_change.team}预算已超支")
                return Decision.BLOCK, "; ".join(reasons)

        if service_usage.status == BudgetStatus.EXCEEDED:
            if impact['metric_priority'] < 80:
                reasons.append(f"服务{metric_change.service}预算已超支")
                return Decision.BLOCK, "; ".join(reasons)

        # 检查变更影响
        if impact['cardinality_change'] > self.config.global_max_per_metric:
            reasons.append(f"单个指标基数增加超过限制({self.config.global_max_per_metric})")
            return Decision.BLOCK, "; ".join(reasons)

        # 警告情况
        if team_usage.status == BudgetStatus.CRITICAL:
            reasons.append(f"团队预算使用率{team_usage.usage_percentage}%，接近上限")
            return Decision.WARN, "; ".join(reasons)

        if impact['label_impact'] == 'high':
            reasons.append("新增多个labels可能导致高基数")
            return Decision.WARN, "; ".join(reasons)

        # 允许
        reasons.append("预算充足，变更影响可控")
        return Decision.ALLOW, "; ".join(reasons)

    def optimize_budget_allocation(self) -> AllocationPlan:
        """
        优化预算分配

        Returns:
            优化后的分配计划
        """
        reallocations = {}
        recommendations = []
        total_optimized = 0

        # 分析各团队使用情况
        team_usages = {}
        for team in self.config.team_budgets:
            team_usages[team] = self.check_budget(team, "team")

        # 识别预算过剩和不足的团队
        surplus_teams = []  # 预算过剩
        deficit_teams = []  # 预算不足

        for team, usage in team_usages.items():
            if usage.usage_percentage < 50:
                surplus_teams.append((team, usage))
            elif usage.usage_percentage > 90:
                deficit_teams.append((team, usage))

        # 重新分配预算
        for surplus_team, surplus_usage in surplus_teams:
            available_to_give = int(surplus_usage.available * 0.3)  # 可转移30%

            for deficit_team, deficit_usage in deficit_teams:
                if available_to_give <= 0:
                    break

                needed = deficit_usage.used - int(deficit_usage.budget * 0.9)
                transfer = min(available_to_give, needed)

                if transfer > 0:
                    if surplus_team not in reallocations:
                        reallocations[surplus_team] = {}
                    reallocations[surplus_team][deficit_team] = transfer
                    available_to_give -= transfer
                    total_optimized += transfer

                    recommendations.append(
                        f"转移 {transfer:,} 序列预算从 {surplus_team} 到 {deficit_team}"
                    )

        # 生成优化建议
        if not reallocations:
            recommendations.append("当前预算分配相对均衡，无需调整")
        else:
            recommendations.append(f"共优化 {total_optimized:,} 时间序列预算")

        # 额外建议
        for team, usage in team_usages.items():
            if usage.top_metrics and usage.top_metrics[0][1] > usage.budget * 0.3:
                recommendations.append(
                    f"⚠️ {team}的指标'{usage.top_metrics[0][0]}'占用"
                    f"{usage.top_metrics[0][1]:,}序列，建议优化"
                )

        return AllocationPlan(
            reallocations=reallocations,
            total_optimized=total_optimized,
            recommendations=recommendations
        )

    def _generate_alert(self, entity: str, entity_type: str, usage: BudgetUsage):
        """生成告警"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'entity': entity,
            'entity_type': entity_type,
            'status': usage.status.value,
            'usage_percentage': usage.usage_percentage,
            'message': f"{entity_type.title()} {entity} 预算使用率 {usage.usage_percentage}%"
        }

        self.alerts.append(alert)
        logger.warning(f"Budget Alert: {alert['message']}")

    def get_global_status(self) -> Dict[str, Any]:
        """获取全局预算状态"""
        total_used = 0
        total_budget = self.config.global_max_series

        # 汇总所有使用
        for usage_data in self.current_usage.values():
            total_used += sum(usage_data.values())

        usage_percentage = (total_used / total_budget * 100) if total_budget > 0 else 0

        # 确定全局状态
        if usage_percentage >= 100:
            global_status = BudgetStatus.EXCEEDED
        elif usage_percentage >= self.config.critical_threshold * 100:
            global_status = BudgetStatus.CRITICAL
        elif usage_percentage >= self.config.warning_threshold * 100:
            global_status = BudgetStatus.WARNING
        else:
            global_status = BudgetStatus.HEALTHY

        # 统计各状态的实体数
        status_counts = {
            BudgetStatus.HEALTHY: 0,
            BudgetStatus.WARNING: 0,
            BudgetStatus.CRITICAL: 0,
            BudgetStatus.EXCEEDED: 0
        }

        for team in self.config.team_budgets:
            usage = self.check_budget(team, "team")
            status_counts[usage.status] += 1

        return {
            'global_budget': total_budget,
            'global_used': total_used,
            'global_available': total_budget - total_used,
            'global_usage_percentage': round(usage_percentage, 2),
            'global_status': global_status.value,
            'global_monthly_cost': round(total_used * self.config.cost_per_series_per_month, 2),
            'status_distribution': {k.value: v for k, v in status_counts.items()},
            'recent_alerts': self.alerts[-5:],
            'recent_decisions': self.decisions[-5:]
        }

    def export_budget_report(self, output_file: str = None) -> str:
        """导出预算报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'global_status': self.get_global_status(),
            'team_budgets': {},
            'service_budgets': {},
            'optimization_plan': None
        }

        # 团队预算状态
        for team in self.config.team_budgets:
            usage = self.check_budget(team, "team")
            report['team_budgets'][team] = asdict(usage)

        # 服务预算状态
        for service in self.config.service_budgets:
            usage = self.check_budget(service, "service")
            report['service_budgets'][service] = asdict(usage)

        # 优化计划
        optimization = self.optimize_budget_allocation()
        report['optimization_plan'] = asdict(optimization)

        json_str = json.dumps(report, indent=2, ensure_ascii=False)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str


def main():
    """主函数 - 用于测试"""
    import argparse

    parser = argparse.ArgumentParser(description='Metrics Budget Controller')
    parser.add_argument('--config', help='Budget configuration file')
    parser.add_argument('--check', help='Check budget for team/service')
    parser.add_argument('--type', choices=['team', 'service'], default='team')
    parser.add_argument('--simulate-change', help='Simulate metric change (JSON)')
    parser.add_argument('--optimize', action='store_true', help='Optimize budget allocation')
    parser.add_argument('--status', action='store_true', help='Show global status')
    parser.add_argument('--output', help='Output report file')

    args = parser.parse_args()

    # 创建控制器
    controller = MetricsBudgetController()

    # 模拟一些使用数据
    sample_usage = {
        'platform': {
            'http_requests_total': 15000,
            'http_request_duration_seconds': 25000,
            'error_rate': 5000,
            'custom_metric_1': 80000
        },
        'api': {
            'api_calls_total': 30000,
            'api_latency': 20000,
            'api_errors': 10000
        },
        'frontend': {
            'page_views': 5000,
            'user_sessions': 3000,
            'js_errors': 2000
        }
    }
    controller.update_usage(sample_usage)

    if args.check:
        # 检查特定实体的预算
        usage = controller.check_budget(args.check, args.type)
        print(f"\n💰 Budget Status for {args.type} '{args.check}':")
        print(f"   Status: {usage.status.value.upper()}")
        print(f"   Budget: {usage.budget:,} series")
        print(f"   Used: {usage.used:,} series ({usage.usage_percentage}%)")
        print(f"   Available: {usage.available:,} series")

        if usage.top_metrics:
            print(f"\n   Top Metrics:")
            for metric, count in usage.top_metrics:
                print(f"   - {metric}: {count:,} series")

    elif args.simulate_change:
        # 模拟指标变更
        change_data = json.loads(args.simulate_change)
        change = MetricChange(**change_data)

        decision, reason = controller.enforce_budget(change)
        print(f"\n🎯 Budget Decision:")
        print(f"   Metric: {change.metric_name}")
        print(f"   Team: {change.team}")
        print(f"   Decision: {decision.value.upper()}")
        print(f"   Reason: {reason}")

    elif args.optimize:
        # 优化预算分配
        plan = controller.optimize_budget_allocation()
        print(f"\n📊 Budget Optimization Plan:")

        if plan.reallocations:
            print(f"   Reallocations:")
            for from_team, transfers in plan.reallocations.items():
                for to_team, amount in transfers.items():
                    print(f"   - {from_team} → {to_team}: {amount:,} series")

        print(f"\n   Recommendations:")
        for rec in plan.recommendations:
            print(f"   - {rec}")

    elif args.status:
        # 显示全局状态
        status = controller.get_global_status()
        print(f"\n🌍 Global Budget Status:")
        print(f"   Status: {status['global_status'].upper()}")
        print(f"   Budget: {status['global_budget']:,} series")
        print(f"   Used: {status['global_used']:,} series ({status['global_usage_percentage']}%)")
        print(f"   Monthly Cost: ${status['global_monthly_cost']:.2f}")

        print(f"\n   Status Distribution:")
        for s, count in status['status_distribution'].items():
            print(f"   - {s}: {count} teams")

    # 导出报告
    if args.output:
        controller.export_budget_report(args.output)
        print(f"\n✅ Report exported to: {args.output}")


if __name__ == "__main__":
    main()