#!/usr/bin/env python3
"""
Supply Chain Risk Assessment
供应链风险评估脚本 - 分析 SBOM 中的安全风险
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RiskFactor:
    """风险因素"""
    name: str
    severity: str  # critical, high, medium, low
    description: str
    recommendation: str


@dataclass
class PackageRisk:
    """包风险评估"""
    package_name: str
    version: str
    risk_score: float = 0.0
    risk_factors: List[RiskFactor] = field(default_factory=list)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    license_risk: Optional[str] = None
    maintenance_risk: Optional[str] = None
    popularity_score: float = 0.0


class SupplyChainRiskAnalyzer:
    """供应链风险分析器"""

    # 高风险许可证
    HIGH_RISK_LICENSES = {
        "GPL-3.0", "GPL-2.0", "AGPL-3.0", "AGPL-2.0",
        "LGPL-3.0", "LGPL-2.1", "LGPL-2.0"
    }

    # 中等风险许可证
    MEDIUM_RISK_LICENSES = {
        "MPL-2.0", "MPL-1.1", "EPL-2.0", "EPL-1.0",
        "CDDL-1.0", "CPL-1.0"
    }

    # 已知有问题的包
    PROBLEMATIC_PACKAGES = {
        "requests": {"min_version": "2.31.0", "reason": "CVE-2023-32681"},
        "urllib3": {"min_version": "2.0.0", "reason": "Multiple CVEs"},
        "cryptography": {"min_version": "41.0.0", "reason": "Security updates"},
        "pyyaml": {"min_version": "6.0.1", "reason": "CVE-2020-14343"},
        "jinja2": {"min_version": "3.1.2", "reason": "Security fixes"},
        "werkzeug": {"min_version": "3.0.0", "reason": "Security updates"},
    }

    def __init__(self):
        self.package_risks: Dict[str, PackageRisk] = {}
        self.total_risk_score: float = 0.0

    def analyze_sbom(self, sbom_path: str) -> Dict[str, Any]:
        """分析 SBOM 文件"""
        with open(sbom_path, 'r') as f:
            sbom = json.load(f)

        # 提取包信息
        packages = self._extract_packages(sbom)

        # 分析每个包
        for package in packages:
            risk = self._analyze_package(package)
            self.package_risks[package['name']] = risk
            self.total_risk_score += risk.risk_score

        # 生成报告
        return self._generate_report()

    def _extract_packages(self, sbom: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 SBOM 中提取包信息"""
        packages = []

        # CycloneDX 格式
        if "components" in sbom:
            for component in sbom["components"]:
                if component.get("type") == "library":
                    packages.append({
                        "name": component.get("name", ""),
                        "version": component.get("version", ""),
                        "license": self._extract_license(component),
                        "purl": component.get("purl", "")
                    })

        # SPDX 格式
        elif "packages" in sbom:
            for package in sbom["packages"]:
                packages.append({
                    "name": package.get("name", ""),
                    "version": package.get("version", ""),
                    "license": package.get("licenseDeclared", ""),
                    "purl": ""
                })

        return packages

    def _extract_license(self, component: Dict[str, Any]) -> str:
        """提取许可证信息"""
        licenses = component.get("licenses", [])
        if licenses and isinstance(licenses, list):
            if isinstance(licenses[0], dict):
                return licenses[0].get("license", {}).get("id", "UNKNOWN")
        return "UNKNOWN"

    def _analyze_package(self, package: Dict[str, Any]) -> PackageRisk:
        """分析单个包的风险"""
        risk = PackageRisk(
            package_name=package["name"],
            version=package["version"]
        )

        # 1. 许可证风险分析
        self._analyze_license_risk(package, risk)

        # 2. 已知漏洞检查
        self._check_known_vulnerabilities(package, risk)

        # 3. 维护状态检查
        self._check_maintenance_status(package, risk)

        # 4. 依赖深度分析
        self._analyze_dependency_depth(package, risk)

        # 5. 版本过时检查
        self._check_version_outdated(package, risk)

        # 计算总风险分数
        risk.risk_score = self._calculate_risk_score(risk)

        return risk

    def _analyze_license_risk(self, package: Dict[str, Any], risk: PackageRisk):
        """分析许可证风险"""
        license_name = package.get("license", "UNKNOWN")

        if license_name in self.HIGH_RISK_LICENSES:
            risk.license_risk = "HIGH"
            risk.risk_factors.append(RiskFactor(
                name="High Risk License",
                severity="high",
                description=f"Package uses {license_name} which may have viral effects",
                recommendation="Review license compatibility with your project"
            ))
            risk.risk_score += 30

        elif license_name in self.MEDIUM_RISK_LICENSES:
            risk.license_risk = "MEDIUM"
            risk.risk_factors.append(RiskFactor(
                name="Medium Risk License",
                severity="medium",
                description=f"Package uses {license_name} which requires attribution",
                recommendation="Ensure proper attribution is maintained"
            ))
            risk.risk_score += 15

        elif license_name == "UNKNOWN":
            risk.license_risk = "UNKNOWN"
            risk.risk_factors.append(RiskFactor(
                name="Unknown License",
                severity="medium",
                description="Package license is not specified",
                recommendation="Manually review the package license"
            ))
            risk.risk_score += 20

    def _check_known_vulnerabilities(self, package: Dict[str, Any], risk: PackageRisk):
        """检查已知漏洞"""
        pkg_name = package["name"].lower()
        pkg_version = package["version"]

        # 检查问题包列表
        if pkg_name in self.PROBLEMATIC_PACKAGES:
            issue = self.PROBLEMATIC_PACKAGES[pkg_name]
            min_version = issue["min_version"]

            if self._compare_versions(pkg_version, min_version) < 0:
                risk.risk_factors.append(RiskFactor(
                    name="Known Vulnerability",
                    severity="critical",
                    description=f"{issue['reason']} - current: {pkg_version}, required: >={min_version}",
                    recommendation=f"Update {pkg_name} to version {min_version} or higher"
                ))
                risk.risk_score += 50

    def _check_maintenance_status(self, package: Dict[str, Any], risk: PackageRisk):
        """检查维护状态"""
        # 这里可以调用 PyPI API 获取实际数据
        # 为演示目的，使用模拟逻辑

        pkg_name = package["name"]

        # 检查是否是已知的废弃包
        deprecated_packages = {"nose", "pycrypto", "python-jose-cryptodome"}
        if pkg_name.lower() in deprecated_packages:
            risk.maintenance_risk = "DEPRECATED"
            risk.risk_factors.append(RiskFactor(
                name="Deprecated Package",
                severity="high",
                description=f"{pkg_name} is deprecated and no longer maintained",
                recommendation=f"Replace {pkg_name} with an actively maintained alternative"
            ))
            risk.risk_score += 40

    def _analyze_dependency_depth(self, package: Dict[str, Any], risk: PackageRisk):
        """分析依赖深度"""
        # 包含特定高风险依赖的包
        high_risk_deps = {"setuptools", "pip", "wheel"}

        if package["name"].lower() in high_risk_deps:
            risk.risk_factors.append(RiskFactor(
                name="Core Dependency",
                severity="low",
                description=f"{package['name']} is a core Python dependency",
                recommendation="Keep this package updated to latest stable version"
            ))
            risk.risk_score += 5

    def _check_version_outdated(self, package: Dict[str, Any], risk: PackageRisk):
        """检查版本是否过时"""
        # 检查版本格式，识别预发布版本
        version = package["version"]
        if any(x in version.lower() for x in ["alpha", "beta", "rc", "dev"]):
            risk.risk_factors.append(RiskFactor(
                name="Pre-release Version",
                severity="medium",
                description=f"Using pre-release version: {version}",
                recommendation="Consider using a stable release version"
            ))
            risk.risk_score += 15

    def _calculate_risk_score(self, risk: PackageRisk) -> float:
        """计算风险分数"""
        # 基于风险因素的严重性调整分数
        severity_multipliers = {
            "critical": 2.0,
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5
        }

        adjusted_score = risk.risk_score
        for factor in risk.risk_factors:
            multiplier = severity_multipliers.get(factor.severity, 1.0)
            adjusted_score *= multiplier

        # 限制最大分数为 100
        return min(adjusted_score, 100.0)

    def _compare_versions(self, v1: str, v2: str) -> int:
        """比较版本号"""
        try:
            from packaging import version
            return -1 if version.parse(v1) < version.parse(v2) else 1
        except:
            # 简单字符串比较作为后备
            return -1 if v1 < v2 else 1

    def _generate_report(self) -> Dict[str, Any]:
        """生成风险报告"""
        # 按风险分数排序
        high_risk_packages = [
            risk for risk in self.package_risks.values()
            if risk.risk_score >= 50
        ]
        medium_risk_packages = [
            risk for risk in self.package_risks.values()
            if 20 <= risk.risk_score < 50
        ]
        low_risk_packages = [
            risk for risk in self.package_risks.values()
            if risk.risk_score < 20
        ]

        # 统计风险因素
        risk_factor_counts = {}
        for risk in self.package_risks.values():
            for factor in risk.risk_factors:
                risk_factor_counts[factor.name] = risk_factor_counts.get(factor.name, 0) + 1

        report = {
            "summary": {
                "total_packages": len(self.package_risks),
                "high_risk_count": len(high_risk_packages),
                "medium_risk_count": len(medium_risk_packages),
                "low_risk_count": len(low_risk_packages),
                "average_risk_score": (
                    self.total_risk_score / len(self.package_risks)
                    if self.package_risks else 0
                ),
                "generated_at": datetime.utcnow().isoformat()
            },
            "high_risk_packages": self._format_packages(high_risk_packages),
            "medium_risk_packages": self._format_packages(medium_risk_packages),
            "risk_factor_distribution": risk_factor_counts,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _format_packages(self, packages: List[PackageRisk]) -> List[Dict[str, Any]]:
        """格式化包信息"""
        formatted = []
        for pkg in sorted(packages, key=lambda x: x.risk_score, reverse=True)[:10]:
            formatted.append({
                "name": pkg.package_name,
                "version": pkg.version,
                "risk_score": round(pkg.risk_score, 2),
                "risk_factors": [
                    {
                        "name": factor.name,
                        "severity": factor.severity,
                        "description": factor.description
                    }
                    for factor in pkg.risk_factors
                ],
                "recommendations": [
                    factor.recommendation
                    for factor in pkg.risk_factors
                ]
            })
        return formatted

    def _generate_recommendations(self) -> List[str]:
        """生成总体建议"""
        recommendations = []

        # 基于风险分析生成建议
        high_risk_count = sum(
            1 for risk in self.package_risks.values()
            if risk.risk_score >= 50
        )

        if high_risk_count > 0:
            recommendations.append(
                f"⚠️  CRITICAL: {high_risk_count} high-risk packages detected. "
                "Immediate action required."
            )

        # 许可证建议
        high_license_risks = sum(
            1 for risk in self.package_risks.values()
            if risk.license_risk == "HIGH"
        )
        if high_license_risks > 0:
            recommendations.append(
                f"Review {high_license_risks} packages with restrictive licenses"
            )

        # 维护建议
        deprecated_count = sum(
            1 for risk in self.package_risks.values()
            if risk.maintenance_risk == "DEPRECATED"
        )
        if deprecated_count > 0:
            recommendations.append(
                f"Replace {deprecated_count} deprecated packages with maintained alternatives"
            )

        # 通用建议
        recommendations.extend([
            "Regularly update dependencies to latest stable versions",
            "Implement automated vulnerability scanning in CI/CD pipeline",
            "Review and approve all new dependencies before adding",
            "Maintain an allowlist of approved packages and licenses"
        ])

        return recommendations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Analyze supply chain risks in SBOM"
    )
    parser.add_argument(
        "--sbom",
        required=True,
        help="Path to SBOM file (CycloneDX or SPDX format)"
    )
    parser.add_argument(
        "--output",
        default="risk-report.json",
        help="Output file for risk report"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Risk score threshold for failing the check"
    )

    args = parser.parse_args()

    # 创建分析器
    analyzer = SupplyChainRiskAnalyzer()

    try:
        # 分析 SBOM
        print(f"Analyzing SBOM: {args.sbom}")
        report = analyzer.analyze_sbom(args.sbom)

        # 保存报告
        if args.format == "json":
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            # Markdown 格式
            with open(args.output, 'w') as f:
                f.write("# Supply Chain Risk Report\n\n")
                f.write(f"Generated: {report['summary']['generated_at']}\n\n")
                f.write("## Summary\n")
                for key, value in report['summary'].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n## High Risk Packages\n")
                for pkg in report['high_risk_packages']:
                    f.write(f"\n### {pkg['name']} v{pkg['version']}\n")
                    f.write(f"Risk Score: {pkg['risk_score']}\n")
                    for factor in pkg['risk_factors']:
                        f.write(f"- {factor['description']}\n")

        print(f"Risk report saved to {args.output}")

        # 打印摘要
        print("\n=== Risk Assessment Summary ===")
        print(f"Total Packages: {report['summary']['total_packages']}")
        print(f"High Risk: {report['summary']['high_risk_count']}")
        print(f"Medium Risk: {report['summary']['medium_risk_count']}")
        print(f"Low Risk: {report['summary']['low_risk_count']}")
        print(f"Average Risk Score: {report['summary']['average_risk_score']:.2f}")

        # 检查是否超过阈值
        if report['summary']['average_risk_score'] > args.threshold:
            print(f"\n❌ FAILED: Average risk score ({report['summary']['average_risk_score']:.2f}) "
                  f"exceeds threshold ({args.threshold})")
            return 1

        print("\n✅ PASSED: Risk assessment within acceptable limits")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())