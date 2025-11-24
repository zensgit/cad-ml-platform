#!/usr/bin/env python3
"""
Recording Rules Versioning System
录制规则版本化系统 - 管理和跟踪 Prometheus recording rules 的变更历史

Features:
- 版本控制和历史跟踪
- 自动备份和回滚
- 规则验证和语法检查
- 变更审计日志
- 规则对比和差异分析
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import subprocess
import difflib


class ChangeType(Enum):
    """变更类型"""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class RuleChange:
    """规则变更记录"""
    rule_name: str
    change_type: ChangeType
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    author: str = ""
    message: str = ""
    impact: List[str] = field(default_factory=list)


@dataclass
class RuleVersion:
    """规则版本信息"""
    version_id: str
    version_number: str
    timestamp: datetime
    author: str
    message: str
    file_hash: str
    changes: List[RuleChange] = field(default_factory=list)
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class RecordingRulesVersionManager:
    """录制规则版本管理器"""

    def __init__(self, rules_dir: str = "prometheus/rules",
                 version_dir: str = ".rules-versions"):
        """初始化版本管理器"""
        self.rules_dir = Path(rules_dir)
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(exist_ok=True)

        # 创建必要的子目录
        self.versions_path = self.version_dir / "versions"
        self.backups_path = self.version_dir / "backups"
        self.metadata_path = self.version_dir / "metadata.json"

        self.versions_path.mkdir(exist_ok=True)
        self.backups_path.mkdir(exist_ok=True)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """加载版本元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "current_version": None,
            "versions": [],
            "last_check": None
        }

    def _save_metadata(self):
        """保存版本元数据"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_rules(self, file_path: Path) -> Dict[str, Any]:
        """加载规则文件"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_rules(self, rules: Dict[str, Any], file_path: Path):
        """保存规则文件"""
        with open(file_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

    def _validate_rules(self, rules_file: Path) -> tuple[bool, List[str]]:
        """验证规则语法

        Returns:
            (is_valid, errors)
        """
        errors = []

        try:
            # 加载并验证 YAML 格式
            rules = self._load_rules(rules_file)

            if not isinstance(rules, dict):
                errors.append("Rules must be a dictionary")
                return False, errors

            # 验证必需的字段
            if 'groups' not in rules:
                errors.append("Missing 'groups' field")
                return False, errors

            # 验证每个规则组
            for group_idx, group in enumerate(rules.get('groups', [])):
                if 'name' not in group:
                    errors.append(f"Group {group_idx}: missing 'name' field")

                if 'rules' not in group:
                    errors.append(f"Group {group_idx}: missing 'rules' field")
                    continue

                # 验证每条规则
                for rule_idx, rule in enumerate(group.get('rules', [])):
                    if 'record' not in rule and 'alert' not in rule:
                        errors.append(
                            f"Group '{group.get('name', group_idx)}', "
                            f"Rule {rule_idx}: must have 'record' or 'alert'"
                        )

                    if 'expr' not in rule:
                        errors.append(
                            f"Group '{group.get('name', group_idx)}', "
                            f"Rule {rule_idx}: missing 'expr' field"
                        )

            # 如果有 promtool，使用它进行额外验证
            if shutil.which('promtool'):
                result = subprocess.run(
                    ['promtool', 'check', 'rules', str(rules_file)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    errors.append(f"promtool validation failed: {result.stderr}")

        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return len(errors) == 0, errors

    def _detect_changes(self, old_rules: Dict[str, Any],
                       new_rules: Dict[str, Any]) -> List[RuleChange]:
        """检测规则变更"""
        changes = []

        old_groups = {g['name']: g for g in old_rules.get('groups', [])}
        new_groups = {g['name']: g for g in new_rules.get('groups', [])}

        # 检测删除的组
        for name in old_groups:
            if name not in new_groups:
                changes.append(RuleChange(
                    rule_name=f"group:{name}",
                    change_type=ChangeType.DELETE,
                    old_value=old_groups[name]
                ))

        # 检测新增的组
        for name in new_groups:
            if name not in old_groups:
                changes.append(RuleChange(
                    rule_name=f"group:{name}",
                    change_type=ChangeType.ADD,
                    new_value=new_groups[name]
                ))

        # 检测修改的组
        for name in old_groups:
            if name in new_groups:
                old_group = old_groups[name]
                new_group = new_groups[name]

                # 比较规则
                old_rules_map = {}
                new_rules_map = {}

                for rule in old_group.get('rules', []):
                    key = rule.get('record') or rule.get('alert')
                    if key:
                        old_rules_map[key] = rule

                for rule in new_group.get('rules', []):
                    key = rule.get('record') or rule.get('alert')
                    if key:
                        new_rules_map[key] = rule

                # 检测规则变化
                for rule_name in old_rules_map:
                    if rule_name not in new_rules_map:
                        changes.append(RuleChange(
                            rule_name=f"{name}:{rule_name}",
                            change_type=ChangeType.DELETE,
                            old_value=old_rules_map[rule_name]
                        ))

                for rule_name in new_rules_map:
                    if rule_name not in old_rules_map:
                        changes.append(RuleChange(
                            rule_name=f"{name}:{rule_name}",
                            change_type=ChangeType.ADD,
                            new_value=new_rules_map[rule_name]
                        ))
                    elif old_rules_map[rule_name] != new_rules_map[rule_name]:
                        changes.append(RuleChange(
                            rule_name=f"{name}:{rule_name}",
                            change_type=ChangeType.MODIFY,
                            old_value=old_rules_map[rule_name],
                            new_value=new_rules_map[rule_name]
                        ))

        return changes

    def _analyze_impact(self, changes: List[RuleChange]) -> Dict[str, List[str]]:
        """分析变更影响"""
        impact = {
            "dashboards": [],
            "alerts": [],
            "downstream_rules": [],
            "queries": []
        }

        for change in changes:
            # 分析对仪表板的影响
            if change.change_type in [ChangeType.DELETE, ChangeType.MODIFY]:
                if ":" in change.rule_name:
                    metric_name = change.rule_name.split(":")[-1]
                    impact["dashboards"].append(
                        f"Dashboard using metric '{metric_name}' may be affected"
                    )

            # 分析对告警的影响
            if change.rule_name.startswith("alert:") or \
               (change.old_value and 'alert' in change.old_value):
                impact["alerts"].append(
                    f"Alert '{change.rule_name}' {change.change_type.value}"
                )

            # 分析对下游规则的影响
            if change.change_type == ChangeType.MODIFY:
                if change.old_value and 'expr' in change.old_value:
                    old_expr = change.old_value['expr']
                    if 'record' in change.old_value:
                        record_name = change.old_value['record']
                        impact["downstream_rules"].append(
                            f"Rules using '{record_name}' may need review"
                        )

        return impact

    def create_version(self, author: str = "", message: str = "") -> Optional[str]:
        """创建新版本

        Returns:
            版本 ID 或 None（如果没有变化）
        """
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 收集所有规则文件
        rule_files = list(self.rules_dir.glob("*.yml")) + \
                    list(self.rules_dir.glob("*.yaml"))

        if not rule_files:
            print("No rule files found")
            return None

        # 创建版本目录
        version_path = self.versions_path / version_id
        version_path.mkdir(exist_ok=True)

        all_changes = []

        for rule_file in rule_files:
            # 验证规则
            is_valid, errors = self._validate_rules(rule_file)
            if not is_valid:
                print(f"Validation failed for {rule_file.name}:")
                for error in errors:
                    print(f"  - {error}")
                # 清理并返回
                shutil.rmtree(version_path)
                return None

            # 复制文件到版本目录
            dest_file = version_path / rule_file.name
            shutil.copy2(rule_file, dest_file)

            # 检测变化
            if self.metadata["current_version"]:
                old_version_path = self.versions_path / self.metadata["current_version"]
                old_file = old_version_path / rule_file.name

                if old_file.exists():
                    old_rules = self._load_rules(old_file)
                    new_rules = self._load_rules(rule_file)
                    changes = self._detect_changes(old_rules, new_rules)
                    all_changes.extend(changes)

        if not all_changes and self.metadata["current_version"]:
            # 没有变化，不创建新版本
            shutil.rmtree(version_path)
            print("No changes detected")
            return None

        # 创建版本信息
        version_info = RuleVersion(
            version_id=version_id,
            version_number=f"v{len(self.metadata['versions']) + 1}",
            timestamp=datetime.now(timezone.utc),
            author=author or os.getenv('USER', 'unknown'),
            message=message or "Manual version",
            file_hash=self._calculate_file_hash(rule_files[0]),
            changes=all_changes,
            parent_version=self.metadata["current_version"]
        )

        # 保存版本信息
        version_info_path = version_path / "version_info.json"
        with open(version_info_path, 'w') as f:
            json.dump({
                "version_id": version_info.version_id,
                "version_number": version_info.version_number,
                "timestamp": version_info.timestamp.isoformat(),
                "author": version_info.author,
                "message": version_info.message,
                "file_hash": version_info.file_hash,
                "changes": [
                    {
                        "rule_name": c.rule_name,
                        "change_type": c.change_type.value,
                        "timestamp": c.timestamp.isoformat()
                    }
                    for c in version_info.changes
                ],
                "parent_version": version_info.parent_version
            }, f, indent=2)

        # 更新元数据
        self.metadata["current_version"] = version_id
        self.metadata["versions"].append(version_id)
        self.metadata["last_check"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()

        print(f"Created version {version_info.version_number} ({version_id})")
        return version_id

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        versions = []

        for version_id in self.metadata["versions"]:
            version_path = self.versions_path / version_id
            info_path = version_path / "version_info.json"

            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    versions.append(info)

        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)

    def rollback(self, version_id: str, backup: bool = True) -> bool:
        """回滚到指定版本

        Args:
            version_id: 版本 ID
            backup: 是否备份当前版本

        Returns:
            是否成功
        """
        version_path = self.versions_path / version_id

        if not version_path.exists():
            print(f"Version {version_id} not found")
            return False

        # 备份当前版本
        if backup:
            backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backups_path / backup_id
            backup_path.mkdir(exist_ok=True)

            for rule_file in self.rules_dir.glob("*.y*ml"):
                shutil.copy2(rule_file, backup_path / rule_file.name)

            print(f"Current rules backed up to {backup_id}")

        # 恢复版本文件
        for rule_file in version_path.glob("*.y*ml"):
            dest_file = self.rules_dir / rule_file.name
            shutil.copy2(rule_file, dest_file)

        # 更新元数据
        self.metadata["current_version"] = version_id
        self._save_metadata()

        print(f"Rolled back to version {version_id}")
        return True

    def diff(self, version1: str, version2: str = None) -> str:
        """比较两个版本的差异

        Args:
            version1: 第一个版本 ID
            version2: 第二个版本 ID（默认为当前版本）

        Returns:
            差异报告
        """
        if version2 is None:
            version2 = self.metadata["current_version"]

        if not version2:
            return "No current version to compare"

        v1_path = self.versions_path / version1
        v2_path = self.versions_path / version2

        if not v1_path.exists():
            return f"Version {version1} not found"
        if not v2_path.exists():
            return f"Version {version2} not found"

        diff_output = []
        diff_output.append(f"Comparing {version1} -> {version2}\n")
        diff_output.append("=" * 50 + "\n")

        # 比较每个文件
        all_files = set()
        all_files.update(f.name for f in v1_path.glob("*.y*ml"))
        all_files.update(f.name for f in v2_path.glob("*.y*ml"))

        for file_name in sorted(all_files):
            if file_name == "version_info.json":
                continue

            file1 = v1_path / file_name
            file2 = v2_path / file_name

            if not file1.exists():
                diff_output.append(f"\n+ NEW FILE: {file_name}\n")
                continue

            if not file2.exists():
                diff_output.append(f"\n- DELETED FILE: {file_name}\n")
                continue

            # 比较文件内容
            with open(file1, 'r') as f:
                lines1 = f.readlines()
            with open(file2, 'r') as f:
                lines2 = f.readlines()

            diff = difflib.unified_diff(
                lines1, lines2,
                fromfile=f"{version1}/{file_name}",
                tofile=f"{version2}/{file_name}",
                lineterm=''
            )

            diff_lines = list(diff)
            if diff_lines:
                diff_output.append(f"\n📝 {file_name}:\n")
                diff_output.extend(diff_lines)

        return "".join(diff_output)

    def auto_commit(self, check_interval: int = 3600) -> bool:
        """自动提交变更（如果有）

        Args:
            check_interval: 检查间隔（秒）

        Returns:
            是否创建了新版本
        """
        last_check = self.metadata.get("last_check")

        if last_check:
            last_check_time = datetime.fromisoformat(last_check)
            time_since_check = (datetime.now(timezone.utc) - last_check_time).seconds

            if time_since_check < check_interval:
                print(f"Last check was {time_since_check}s ago, skipping")
                return False

        # 检查变化并自动提交
        version_id = self.create_version(
            author="auto-commit",
            message="Automatic version checkpoint"
        )

        return version_id is not None

    def export_version(self, version_id: str, output_dir: str) -> bool:
        """导出版本到指定目录

        Args:
            version_id: 版本 ID
            output_dir: 输出目录

        Returns:
            是否成功
        """
        version_path = self.versions_path / version_id

        if not version_path.exists():
            print(f"Version {version_id} not found")
            return False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 复制规则文件
        for rule_file in version_path.glob("*.y*ml"):
            shutil.copy2(rule_file, output_path / rule_file.name)

        # 复制版本信息
        info_file = version_path / "version_info.json"
        if info_file.exists():
            shutil.copy2(info_file, output_path / "version_info.json")

        print(f"Exported version {version_id} to {output_dir}")
        return True

    def generate_report(self) -> Dict[str, Any]:
        """生成版本管理报告"""
        versions = self.list_versions()

        report = {
            "total_versions": len(versions),
            "current_version": self.metadata.get("current_version"),
            "last_check": self.metadata.get("last_check"),
            "versions": [],
            "statistics": {
                "total_changes": 0,
                "adds": 0,
                "modifies": 0,
                "deletes": 0
            },
            "recent_activity": []
        }

        # 统计变化
        for version in versions[:10]:  # 最近10个版本
            changes = version.get("changes", [])

            version_summary = {
                "version_id": version["version_id"],
                "version_number": version["version_number"],
                "timestamp": version["timestamp"],
                "author": version["author"],
                "message": version["message"],
                "change_count": len(changes)
            }

            report["versions"].append(version_summary)

            # 统计变化类型
            for change in changes:
                change_type = change.get("change_type", "")
                report["statistics"]["total_changes"] += 1

                if change_type == "add":
                    report["statistics"]["adds"] += 1
                elif change_type == "modify":
                    report["statistics"]["modifies"] += 1
                elif change_type == "delete":
                    report["statistics"]["deletes"] += 1

        # 最近活动
        if versions:
            recent = versions[0]
            report["recent_activity"] = {
                "last_version": recent["version_id"],
                "last_author": recent["author"],
                "last_message": recent["message"],
                "last_timestamp": recent["timestamp"]
            }

        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Recording Rules Version Management System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init 命令
    init_parser = subparsers.add_parser("init", help="Initialize versioning")
    init_parser.add_argument("--rules-dir", default="prometheus/rules",
                            help="Rules directory")

    # commit 命令
    commit_parser = subparsers.add_parser("commit", help="Create new version")
    commit_parser.add_argument("-m", "--message", required=True,
                              help="Version message")
    commit_parser.add_argument("-a", "--author", help="Author name")

    # list 命令
    list_parser = subparsers.add_parser("list", help="List versions")
    list_parser.add_argument("-n", "--number", type=int, default=10,
                            help="Number of versions to show")

    # rollback 命令
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to version")
    rollback_parser.add_argument("version", help="Version ID")
    rollback_parser.add_argument("--no-backup", action="store_true",
                                help="Don't backup current version")

    # diff 命令
    diff_parser = subparsers.add_parser("diff", help="Compare versions")
    diff_parser.add_argument("version1", help="First version")
    diff_parser.add_argument("version2", nargs="?", help="Second version (default: current)")

    # export 命令
    export_parser = subparsers.add_parser("export", help="Export version")
    export_parser.add_argument("version", help="Version ID")
    export_parser.add_argument("-o", "--output", required=True, help="Output directory")

    # report 命令
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--format", choices=["json", "markdown"], default="markdown")

    # auto-commit 命令
    auto_parser = subparsers.add_parser("auto-commit", help="Auto commit if changed")
    auto_parser.add_argument("--interval", type=int, default=3600,
                            help="Check interval in seconds")

    args = parser.parse_args()

    # 创建管理器
    manager = RecordingRulesVersionManager()

    try:
        if args.command == "init":
            manager = RecordingRulesVersionManager(rules_dir=args.rules_dir)
            print(f"Initialized versioning for {args.rules_dir}")

        elif args.command == "commit":
            version_id = manager.create_version(
                author=args.author or os.getenv('USER', 'unknown'),
                message=args.message
            )
            if version_id:
                print(f"Created version: {version_id}")
            else:
                print("No changes to commit")

        elif args.command == "list":
            versions = manager.list_versions()

            if not versions:
                print("No versions found")
                return 0

            print(f"\n📋 Recording Rules Versions (showing {min(args.number, len(versions))})\n")
            print(f"{'Version':<10} {'ID':<20} {'Author':<15} {'Message':<40} {'Changes':<10}")
            print("-" * 105)

            for version in versions[:args.number]:
                print(f"{version['version_number']:<10} "
                     f"{version['version_id']:<20} "
                     f"{version['author']:<15} "
                     f"{version['message'][:37] + '...' if len(version['message']) > 40 else version['message']:<40} "
                     f"{len(version.get('changes', [])):<10}")

        elif args.command == "rollback":
            success = manager.rollback(
                args.version,
                backup=not args.no_backup
            )
            if success:
                print(f"Successfully rolled back to {args.version}")

        elif args.command == "diff":
            diff_output = manager.diff(args.version1, args.version2)
            print(diff_output)

        elif args.command == "export":
            success = manager.export_version(args.version, args.output)
            if success:
                print(f"Exported to {args.output}")

        elif args.command == "report":
            report = manager.generate_report()

            if args.format == "json":
                print(json.dumps(report, indent=2))
            else:
                # Markdown 格式
                print("# Recording Rules Version Report\n")
                print(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                print("## Summary\n")
                print(f"- **Total Versions**: {report['total_versions']}")
                print(f"- **Current Version**: {report['current_version']}")
                print(f"- **Last Check**: {report['last_check']}")

                print("\n## Change Statistics\n")
                stats = report['statistics']
                print(f"- **Total Changes**: {stats['total_changes']}")
                print(f"- **Additions**: {stats['adds']}")
                print(f"- **Modifications**: {stats['modifies']}")
                print(f"- **Deletions**: {stats['deletes']}")

                if report['recent_activity']:
                    print("\n## Recent Activity\n")
                    recent = report['recent_activity']
                    print(f"- **Last Version**: {recent['last_version']}")
                    print(f"- **Last Author**: {recent['last_author']}")
                    print(f"- **Last Message**: {recent['last_message']}")
                    print(f"- **Last Update**: {recent['last_timestamp']}")

                if report['versions']:
                    print("\n## Recent Versions\n")
                    print("| Version | ID | Author | Message | Changes |")
                    print("|---------|----|---------|---------|---------")
                    for v in report['versions'][:5]:
                        print(f"| {v['version_number']} | {v['version_id']} | "
                             f"{v['author']} | {v['message'][:30]}... | {v['change_count']} |")

        elif args.command == "auto-commit":
            created = manager.auto_commit(check_interval=args.interval)
            if created:
                print("Auto-commit created new version")
            else:
                print("No changes to auto-commit")

        else:
            parser.print_help()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())