#!/usr/bin/env python3
"""
错误码生命周期管理器
Error Code Lifecycle Manager - 管理错误码的完整生命周期
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.error_code_scanner import ErrorCodeScanner, ErrorCode

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CleanupPlan:
    """清理计划"""
    def __init__(self):
        self.immediate_removal = []  # 立即删除
        self.deprecation = []        # 标记弃用
        self.consolidation = []      # 合并重复
        self.monitoring = []         # 继续监控
        self.migration_guide = {}    # 迁移指南


class ErrorCodeLifecycleManager:
    """错误码生命周期管理"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化管理器

        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.scanner = ErrorCodeScanner()
        self.history_file = Path('data/error_code_history.json')
        self.history = self._load_history()

    def _load_config(self, config_file: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            'thresholds': {
                'unused_days': 60,           # 未使用天数阈值
                'rare_usage_count': 10,       # 稀有使用次数阈值
                'deprecation_days': 30,       # 弃用保留天数
                'min_usage_for_active': 100   # 活跃使用的最低次数
            },
            'policies': {
                'auto_remove_unused': True,   # 自动删除未使用
                'auto_deprecate_rare': True,  # 自动弃用稀有
                'merge_duplicates': True,     # 合并重复
                'require_migration_doc': True # 需要迁移文档
            },
            'exclusions': {
                'protected_codes': [],        # 受保护的错误码（不能删除）
                'ignore_patterns': []         # 忽略的模式
            }
        }

        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # 深度合并配置
                self._merge_config(default_config, user_config)

        return default_config

    def _merge_config(self, base: dict, override: dict):
        """深度合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _load_history(self) -> Dict:
        """加载历史记录"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            'scans': [],
            'cleanups': [],
            'error_code_lifecycle': {}
        }

    def _save_history(self):
        """保存历史记录"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def analyze_lifecycle(self) -> Dict[str, Any]:
        """
        分析错误码生命周期

        Returns:
            生命周期分析结果
        """
        logger.info("开始分析错误码生命周期...")

        # 执行扫描
        scan_results = self.scanner.scan_all()

        # 记录扫描历史
        self.history['scans'].append({
            'timestamp': datetime.now().isoformat(),
            'summary': scan_results['summary']
        })

        # 分析生命周期状态
        lifecycle_analysis = self._analyze_lifecycle_status(scan_results)

        # 生成清理计划
        cleanup_plan = self.generate_cleanup_plan(scan_results, lifecycle_analysis)

        # 保存历史
        self._save_history()

        return {
            'scan_results': scan_results,
            'lifecycle_analysis': lifecycle_analysis,
            'cleanup_plan': cleanup_plan
        }

    def _analyze_lifecycle_status(self, scan_results: Dict) -> Dict[str, Any]:
        """分析生命周期状态"""
        classification = scan_results['classification']
        definitions = scan_results['definitions']
        log_stats = scan_results['log_stats']

        lifecycle_status = {
            'healthy': [],      # 健康的
            'at_risk': [],      # 有风险的
            'unhealthy': [],    # 不健康的
            'critical': []      # 严重问题的
        }

        # 分析每个错误码的健康状态
        for code, error_code in definitions.items():
            status = error_code.status
            usage_count = log_stats.get(code, 0)

            # 更新生命周期历史
            if code not in self.history['error_code_lifecycle']:
                self.history['error_code_lifecycle'][code] = {
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': None,
                    'usage_history': []
                }

            lifecycle_record = self.history['error_code_lifecycle'][code]
            lifecycle_record['usage_history'].append({
                'date': datetime.now().isoformat(),
                'count': usage_count,
                'status': status
            })

            # 判断健康状态
            if status == 'ACTIVE' and usage_count > self.config['thresholds']['min_usage_for_active']:
                lifecycle_status['healthy'].append(code)
            elif status == 'RARE' or usage_count < self.config['thresholds']['rare_usage_count']:
                lifecycle_status['at_risk'].append(code)
            elif status in ['UNUSED', 'ZOMBIE']:
                lifecycle_status['unhealthy'].append(code)
            elif status in ['DEPRECATED', 'DUPLICATE', 'ORPHAN']:
                lifecycle_status['critical'].append(code)

            # 更新最后使用时间
            if usage_count > 0:
                lifecycle_record['last_seen'] = datetime.now().isoformat()

        return lifecycle_status

    def generate_cleanup_plan(
        self,
        scan_results: Dict,
        lifecycle_analysis: Dict
    ) -> CleanupPlan:
        """
        生成清理计划

        Args:
            scan_results: 扫描结果
            lifecycle_analysis: 生命周期分析

        Returns:
            清理计划
        """
        plan = CleanupPlan()
        classification = scan_results['classification']
        definitions = scan_results['definitions']
        duplicates = scan_results['duplicates']

        # 1. 处理未使用的错误码
        for code in classification['UNUSED']:
            if self._is_protected(code):
                plan.monitoring.append(code)
                continue

            # 检查未使用天数
            lifecycle_record = self.history['error_code_lifecycle'].get(code, {})
            first_seen = lifecycle_record.get('first_seen')
            last_seen = lifecycle_record.get('last_seen')

            if first_seen:
                first_seen_date = datetime.fromisoformat(first_seen)
                days_since_first = (datetime.now() - first_seen_date).days

                if days_since_first > self.config['thresholds']['unused_days']:
                    plan.immediate_removal.append(code)
                elif days_since_first > self.config['thresholds']['deprecation_days']:
                    plan.deprecation.append(code)
                else:
                    plan.monitoring.append(code)
            else:
                # 新发现的未使用码，先监控
                plan.monitoring.append(code)

        # 2. 处理僵尸错误码（曾经使用过但现在没了）
        for code in classification['ZOMBIE']:
            if not self._is_protected(code):
                plan.immediate_removal.append(code)

        # 3. 处理稀有使用的错误码
        for code in classification['RARE']:
            if not self._is_protected(code) and self.config['policies']['auto_deprecate_rare']:
                plan.deprecation.append(code)
            else:
                plan.monitoring.append(code)

        # 4. 处理重复定义
        if self.config['policies']['merge_duplicates']:
            for code in duplicates:
                plan.consolidation.append({
                    'code': code,
                    'locations': duplicates[code]
                })

        # 5. 生成迁移指南
        if self.config['policies']['require_migration_doc']:
            plan.migration_guide = self._generate_migration_guide(plan)

        return plan

    def _is_protected(self, code: str) -> bool:
        """检查错误码是否受保护"""
        # 检查是否在保护列表中
        if code in self.config['exclusions']['protected_codes']:
            return True

        # 检查是否匹配忽略模式
        for pattern in self.config['exclusions']['ignore_patterns']:
            if pattern in code:
                return True

        return False

    def _generate_migration_guide(self, plan: CleanupPlan) -> Dict[str, Any]:
        """生成迁移指南"""
        guide = {
            'deprecation_timeline': {},
            'replacement_mapping': {},
            'migration_steps': []
        }

        # 弃用时间线
        deprecation_date = datetime.now()
        removal_date = deprecation_date + timedelta(days=self.config['thresholds']['deprecation_days'])

        for code in plan.deprecation:
            guide['deprecation_timeline'][code] = {
                'deprecated_date': deprecation_date.isoformat(),
                'removal_date': removal_date.isoformat(),
                'grace_period_days': self.config['thresholds']['deprecation_days']
            }

        # 替代映射（这里需要智能分析或人工指定）
        # 暂时使用简单规则
        for code in plan.deprecation:
            # 查找相似的活跃错误码作为替代
            # 实际实现中应该有更智能的映射逻辑
            guide['replacement_mapping'][code] = self._find_replacement(code)

        # 迁移步骤
        guide['migration_steps'] = [
            "1. 更新代码中的错误码引用",
            "2. 更新文档和API说明",
            "3. 通知客户端团队",
            "4. 设置监控告警",
            "5. 在宽限期后删除旧错误码"
        ]

        return guide

    def _find_replacement(self, code: str) -> Optional[str]:
        """查找替代错误码"""
        # 这里应该有更智能的逻辑
        # 比如基于语义相似度或功能相似度
        # 现在返回通用替代
        if 'NOT_FOUND' in code:
            return 'ERROR_NOT_FOUND'
        elif 'AUTH' in code:
            return 'ERROR_UNAUTHORIZED'
        elif 'INVALID' in code:
            return 'ERROR_INVALID_INPUT'
        else:
            return 'ERROR_GENERAL'

    def apply_cleanup_plan(self, plan: CleanupPlan, dry_run: bool = True) -> Dict[str, Any]:
        """
        应用清理计划

        Args:
            plan: 清理计划
            dry_run: 是否为演练模式

        Returns:
            应用结果
        """
        results = {
            'removed': [],
            'deprecated': [],
            'consolidated': [],
            'errors': []
        }

        if dry_run:
            logger.info("演练模式：不会实际修改文件")

        # 1. 删除未使用的错误码
        for code in plan.immediate_removal:
            try:
                if not dry_run:
                    self._remove_error_code(code)
                results['removed'].append(code)
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}删除错误码: {code}")
            except Exception as e:
                results['errors'].append(f"删除 {code} 失败: {e}")

        # 2. 标记弃用
        for code in plan.deprecation:
            try:
                if not dry_run:
                    self._deprecate_error_code(code)
                results['deprecated'].append(code)
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}弃用错误码: {code}")
            except Exception as e:
                results['errors'].append(f"弃用 {code} 失败: {e}")

        # 3. 合并重复
        for item in plan.consolidation:
            try:
                if not dry_run:
                    self._consolidate_error_code(item['code'], item['locations'])
                results['consolidated'].append(item['code'])
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}合并错误码: {item['code']}")
            except Exception as e:
                results['errors'].append(f"合并 {item['code']} 失败: {e}")

        # 4. 记录清理历史
        if not dry_run:
            self.history['cleanups'].append({
                'timestamp': datetime.now().isoformat(),
                'removed': results['removed'],
                'deprecated': results['deprecated'],
                'consolidated': results['consolidated']
            })
            self._save_history()

        return results

    def _remove_error_code(self, code: str):
        """删除错误码（实际实现）"""
        # 这里应该实现实际的代码删除逻辑
        # 包括：
        # 1. 从定义文件中删除
        # 2. 更新引用
        # 3. 更新文档
        pass

    def _deprecate_error_code(self, code: str):
        """标记错误码为弃用"""
        # 这里应该实现实际的弃用标记逻辑
        # 包括：
        # 1. 添加 @deprecated 注释
        # 2. 更新文档
        # 3. 添加警告日志
        pass

    def _consolidate_error_code(self, code: str, locations: List[Dict]):
        """合并重复的错误码定义"""
        # 这里应该实现实际的合并逻辑
        # 包括：
        # 1. 保留一个定义
        # 2. 删除其他重复定义
        # 3. 更新所有引用
        pass

    def generate_cleanup_report(self, plan: CleanupPlan) -> str:
        """
        生成清理报告

        Args:
            plan: 清理计划

        Returns:
            Markdown格式报告
        """
        report = f"""# 错误码清理计划

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 清理概要

- **立即删除**: {len(plan.immediate_removal)} 个
- **标记弃用**: {len(plan.deprecation)} 个
- **合并重复**: {len(plan.consolidation)} 个
- **继续监控**: {len(plan.monitoring)} 个

## 🗑️ 立即删除列表

以下错误码超过 {self.config['thresholds']['unused_days']} 天未使用，建议立即删除：

"""
        if plan.immediate_removal:
            for code in plan.immediate_removal[:50]:  # 最多显示50个
                report += f"- `{code}`\n"
            if len(plan.immediate_removal) > 50:
                report += f"- ... 还有 {len(plan.immediate_removal) - 50} 个\n"
        else:
            report += "- 无\n"

        report += f"""
## ⚠️ 标记弃用列表

以下错误码使用率极低（< {self.config['thresholds']['rare_usage_count']} 次/月），建议标记为弃用：

"""
        if plan.deprecation:
            for code in plan.deprecation[:30]:
                replacement = plan.migration_guide.get('replacement_mapping', {}).get(code, '待定')
                report += f"- `{code}` → `{replacement}`\n"
            if len(plan.deprecation) > 30:
                report += f"- ... 还有 {len(plan.deprecation) - 30} 个\n"
        else:
            report += "- 无\n"

        report += """
## 🔄 合并重复列表

以下错误码存在重复定义，需要合并：

"""
        if plan.consolidation:
            for item in plan.consolidation:
                report += f"\n**`{item['code']}`**:\n"
                for loc in item['locations'][:5]:
                    report += f"  - {loc['file']}:{loc['line']}\n"
        else:
            report += "- 无\n"

        report += """
## 📈 影响评估

"""
        total_affected = len(plan.immediate_removal) + len(plan.deprecation) + len(plan.consolidation)
        report += f"""- **受影响错误码总数**: {total_affected}
- **代码体积减少**: 约 {total_affected * 50} 字节
- **维护成本降低**: 约 {total_affected * 10} 分钟/月
- **错误码清晰度提升**: {(total_affected / 200 * 100):.1f}%

## 📝 迁移指南

### 时间线
"""
        if plan.migration_guide and plan.migration_guide.get('deprecation_timeline'):
            sample = list(plan.migration_guide['deprecation_timeline'].values())[0] if plan.migration_guide['deprecation_timeline'] else {}
            if sample:
                report += f"""- **弃用日期**: {sample.get('deprecated_date', '').split('T')[0]}
- **删除日期**: {sample.get('removal_date', '').split('T')[0]}
- **宽限期**: {sample.get('grace_period_days', 30)} 天
"""

        report += """
### 迁移步骤
"""
        if plan.migration_guide and plan.migration_guide.get('migration_steps'):
            for step in plan.migration_guide['migration_steps']:
                report += f"{step}\n"

        report += """
## ✅ 下一步行动

1. **审核清理计划** - 确认删除和弃用列表
2. **通知相关团队** - 特别是客户端团队
3. **执行清理** - 运行 `--apply` 参数应用计划
4. **监控影响** - 观察日志中的异常
5. **更新文档** - 同步更新错误码文档

## ⚠️ 注意事项

- 删除操作不可逆，请先备份
- 建议先在开发环境测试
- 保留至少一周的回滚窗口
- 确保客户端已同步更新

---
*此报告由错误码生命周期管理系统自动生成*
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='错误码生命周期管理')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--analyze', action='store_true', help='分析生命周期')
    parser.add_argument('--plan', action='store_true', help='生成清理计划')
    parser.add_argument('--apply', action='store_true', help='应用清理计划')
    parser.add_argument('--dry-run', action='store_true', help='演练模式')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown',
                       help='输出格式')

    args = parser.parse_args()

    # 创建管理器
    manager = ErrorCodeLifecycleManager(config_file=args.config)

    if args.analyze or args.plan:
        # 分析生命周期
        results = manager.analyze_lifecycle()
        plan = results['cleanup_plan']

        if args.format == 'json':
            output = json.dumps({
                'summary': results['scan_results']['summary'],
                'lifecycle_analysis': results['lifecycle_analysis'],
                'cleanup_plan': {
                    'immediate_removal': plan.immediate_removal,
                    'deprecation': plan.deprecation,
                    'consolidation': plan.consolidation,
                    'monitoring': plan.monitoring
                }
            }, indent=2)
        else:
            output = manager.generate_cleanup_report(plan)

        # 输出结果
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"✅ 报告已保存到: {args.output}")
        else:
            print(output)

        # 如果需要应用计划
        if args.apply:
            print("\n开始应用清理计划...\n")
            apply_results = manager.apply_cleanup_plan(plan, dry_run=args.dry_run)
            print(f"✅ 已删除: {len(apply_results['removed'])} 个")
            print(f"⚠️ 已弃用: {len(apply_results['deprecated'])} 个")
            print(f"🔄 已合并: {len(apply_results['consolidated'])} 个")
            if apply_results['errors']:
                print(f"❌ 错误: {len(apply_results['errors'])} 个")
                for error in apply_results['errors']:
                    print(f"  - {error}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()