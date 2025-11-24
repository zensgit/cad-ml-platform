#!/usr/bin/env python3
"""
SBOM Comparison Tool
比较两个 SBOM 文件，生成依赖变更报告
"""

import argparse
import json
import sys
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass


@dataclass
class PackageChange:
    """包变更信息"""
    name: str
    change_type: str  # added, removed, updated, license_changed
    old_version: str = ""
    new_version: str = ""
    old_license: str = ""
    new_license: str = ""


class SBOMComparator:
    """SBOM 比较器"""

    def __init__(self):
        self.changes: List[PackageChange] = []

    def compare(self, base_sbom_path: str, current_sbom_path: str) -> Dict[str, Any]:
        """比较两个 SBOM 文件"""

        # 加载 SBOM
        with open(base_sbom_path, 'r') as f:
            base_sbom = json.load(f)

        with open(current_sbom_path, 'r') as f:
            current_sbom = json.load(f)

        # 提取包信息
        base_packages = self._extract_packages(base_sbom)
        current_packages = self._extract_packages(current_sbom)

        # 比较包
        self._compare_packages(base_packages, current_packages)

        # 生成报告
        return self._generate_report()

    def _extract_packages(self, sbom: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """提取包信息到字典"""
        packages = {}

        # CycloneDX 格式
        if "components" in sbom:
            for component in sbom["components"]:
                if component.get("type") == "library":
                    name = component.get("name", "")
                    packages[name] = {
                        "version": component.get("version", ""),
                        "license": self._extract_license(component)
                    }

        # SPDX 格式
        elif "packages" in sbom:
            for package in sbom["packages"]:
                name = package.get("name", "")
                if name and not name.startswith("SPDXRef-"):
                    packages[name] = {
                        "version": package.get("version", ""),
                        "license": package.get("licenseDeclared", "")
                    }

        return packages

    def _extract_license(self, component: Dict[str, Any]) -> str:
        """提取许可证信息"""
        licenses = component.get("licenses", [])
        if licenses and isinstance(licenses, list):
            if isinstance(licenses[0], dict):
                license_info = licenses[0].get("license", {})
                if isinstance(license_info, dict):
                    return license_info.get("id", "UNKNOWN")
                return str(license_info)
        return "UNKNOWN"

    def _compare_packages(
        self,
        base_packages: Dict[str, Dict[str, str]],
        current_packages: Dict[str, Dict[str, str]]
    ):
        """比较包差异"""

        base_names = set(base_packages.keys())
        current_names = set(current_packages.keys())

        # 新增的包
        added = current_names - base_names
        for name in added:
            self.changes.append(PackageChange(
                name=name,
                change_type="added",
                new_version=current_packages[name]["version"],
                new_license=current_packages[name]["license"]
            ))

        # 删除的包
        removed = base_names - current_names
        for name in removed:
            self.changes.append(PackageChange(
                name=name,
                change_type="removed",
                old_version=base_packages[name]["version"],
                old_license=base_packages[name]["license"]
            ))

        # 更新的包
        common = base_names & current_names
        for name in common:
            base_pkg = base_packages[name]
            current_pkg = current_packages[name]

            # 版本变更
            if base_pkg["version"] != current_pkg["version"]:
                self.changes.append(PackageChange(
                    name=name,
                    change_type="updated",
                    old_version=base_pkg["version"],
                    new_version=current_pkg["version"],
                    old_license=base_pkg["license"],
                    new_license=current_pkg["license"]
                ))

            # 仅许可证变更
            elif base_pkg["license"] != current_pkg["license"]:
                self.changes.append(PackageChange(
                    name=name,
                    change_type="license_changed",
                    old_version=base_pkg["version"],
                    new_version=current_pkg["version"],
                    old_license=base_pkg["license"],
                    new_license=current_pkg["license"]
                ))

    def _generate_report(self) -> Dict[str, Any]:
        """生成比较报告"""

        # 按变更类型分组
        added = [c for c in self.changes if c.change_type == "added"]
        removed = [c for c in self.changes if c.change_type == "removed"]
        updated = [c for c in self.changes if c.change_type == "updated"]
        license_changed = [c for c in self.changes if c.change_type == "license_changed"]

        # 检测安全相关变更
        security_updates = self._detect_security_updates(updated)
        license_risks = self._detect_license_risks(added + updated + license_changed)

        report = {
            "summary": {
                "total_changes": len(self.changes),
                "added": len(added),
                "removed": len(removed),
                "updated": len(updated),
                "license_changed": len(license_changed),
                "security_related": len(security_updates),
                "license_risks": len(license_risks)
            },
            "changes": {
                "added": self._format_changes(added),
                "removed": self._format_changes(removed),
                "updated": self._format_changes(updated),
                "license_changed": self._format_changes(license_changed)
            },
            "security_updates": security_updates,
            "license_risks": license_risks
        }

        return report

    def _format_changes(self, changes: List[PackageChange]) -> List[Dict[str, str]]:
        """格式化变更列表"""
        formatted = []
        for change in sorted(changes, key=lambda x: x.name):
            item = {"name": change.name}

            if change.change_type == "added":
                item["version"] = change.new_version
                item["license"] = change.new_license

            elif change.change_type == "removed":
                item["version"] = change.old_version
                item["license"] = change.old_license

            elif change.change_type in ["updated", "license_changed"]:
                item["old_version"] = change.old_version
                item["new_version"] = change.new_version
                item["old_license"] = change.old_license
                item["new_license"] = change.new_license

            formatted.append(item)

        return formatted

    def _detect_security_updates(self, updated: List[PackageChange]) -> List[Dict[str, str]]:
        """检测安全相关的更新"""
        security_packages = {
            "cryptography", "pyyaml", "urllib3", "requests",
            "werkzeug", "jinja2", "django", "flask"
        }

        security_updates = []
        for change in updated:
            if change.name.lower() in security_packages:
                security_updates.append({
                    "name": change.name,
                    "old_version": change.old_version,
                    "new_version": change.new_version,
                    "recommendation": "Review security changelog for this update"
                })

        return security_updates

    def _detect_license_risks(self, changes: List[PackageChange]) -> List[Dict[str, str]]:
        """检测许可证风险"""
        risky_licenses = {
            "GPL-3.0", "GPL-2.0", "AGPL-3.0", "AGPL-2.0",
            "LGPL-3.0", "LGPL-2.1"
        }

        license_risks = []
        for change in changes:
            new_license = change.new_license
            if new_license in risky_licenses:
                license_risks.append({
                    "name": change.name,
                    "license": new_license,
                    "risk": "Copyleft license may affect your project",
                    "recommendation": "Review license compatibility"
                })

        return license_risks

    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成 Markdown 格式的报告"""
        lines = []

        # 摘要
        summary = report["summary"]
        if summary["total_changes"] == 0:
            lines.append("✅ No dependency changes detected")
            return "\n".join(lines)

        lines.append("### 📊 Summary")
        lines.append("")
        lines.append(f"Total Changes: **{summary['total_changes']}**")
        lines.append("")

        if summary["added"] > 0:
            lines.append(f"- ➕ Added: {summary['added']}")
        if summary["removed"] > 0:
            lines.append(f"- ➖ Removed: {summary['removed']}")
        if summary["updated"] > 0:
            lines.append(f"- 🔄 Updated: {summary['updated']}")
        if summary["license_changed"] > 0:
            lines.append(f"- 📜 License Changed: {summary['license_changed']}")

        # 安全警告
        if summary["security_related"] > 0:
            lines.append("")
            lines.append(f"⚠️ **Security-related updates: {summary['security_related']}**")

        if summary["license_risks"] > 0:
            lines.append("")
            lines.append(f"⚠️ **License risks detected: {summary['license_risks']}**")

        # 详细变更
        changes = report["changes"]

        # 新增包
        if changes["added"]:
            lines.append("")
            lines.append("### ➕ Added Dependencies")
            lines.append("")
            lines.append("| Package | Version | License |")
            lines.append("|---------|---------|---------|")
            for pkg in changes["added"]:
                lines.append(f"| {pkg['name']} | {pkg['version']} | {pkg['license']} |")

        # 删除包
        if changes["removed"]:
            lines.append("")
            lines.append("### ➖ Removed Dependencies")
            lines.append("")
            lines.append("| Package | Version | License |")
            lines.append("|---------|---------|---------|")
            for pkg in changes["removed"]:
                lines.append(f"| {pkg['name']} | {pkg['version']} | {pkg['license']} |")

        # 更新包
        if changes["updated"]:
            lines.append("")
            lines.append("### 🔄 Updated Dependencies")
            lines.append("")
            lines.append("| Package | Old Version | New Version | License |")
            lines.append("|---------|-------------|-------------|---------|")
            for pkg in changes["updated"]:
                version_change = f"{pkg['old_version']} → {pkg['new_version']}"
                license_info = pkg['new_license']
                if pkg['old_license'] != pkg['new_license']:
                    license_info = f"{pkg['old_license']} → {pkg['new_license']}"
                lines.append(f"| {pkg['name']} | {pkg['old_version']} | {pkg['new_version']} | {license_info} |")

        # 许可证变更
        if changes["license_changed"]:
            lines.append("")
            lines.append("### 📜 License Changes")
            lines.append("")
            lines.append("| Package | Version | Old License | New License |")
            lines.append("|---------|---------|-------------|-------------|")
            for pkg in changes["license_changed"]:
                lines.append(f"| {pkg['name']} | {pkg['new_version']} | {pkg['old_license']} | {pkg['new_license']} |")

        # 安全更新
        if report["security_updates"]:
            lines.append("")
            lines.append("### 🔒 Security Updates")
            lines.append("")
            for update in report["security_updates"]:
                lines.append(f"- **{update['name']}** {update['old_version']} → {update['new_version']}")
                lines.append(f"  - {update['recommendation']}")

        # 许可证风险
        if report["license_risks"]:
            lines.append("")
            lines.append("### ⚠️ License Risks")
            lines.append("")
            for risk in report["license_risks"]:
                lines.append(f"- **{risk['name']}** ({risk['license']})")
                lines.append(f"  - {risk['risk']}")
                lines.append(f"  - {risk['recommendation']}")

        return "\n".join(lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Compare two SBOM files")
    parser.add_argument("--base", required=True, help="Base SBOM file")
    parser.add_argument("--current", required=True, help="Current SBOM file")
    parser.add_argument("--output", default="sbom-diff.md", help="Output file")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown",
                       help="Output format")

    args = parser.parse_args()

    # 创建比较器
    comparator = SBOMComparator()

    try:
        # 执行比较
        print(f"Comparing SBOMs...")
        print(f"  Base: {args.base}")
        print(f"  Current: {args.current}")

        report = comparator.compare(args.base, args.current)

        # 生成输出
        if args.format == "json":
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            markdown = comparator.generate_markdown_report(report)
            with open(args.output, 'w') as f:
                f.write(markdown)

        print(f"Comparison report saved to {args.output}")

        # 打印摘要
        summary = report["summary"]
        print(f"\nChanges detected: {summary['total_changes']}")
        if summary["security_related"] > 0:
            print(f"⚠️  Security-related updates: {summary['security_related']}")
        if summary["license_risks"] > 0:
            print(f"⚠️  License risks: {summary['license_risks']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())