#!/usr/bin/env python3
"""
SPDX SBOM Generator
生成 SPDX 格式的软件物料清单
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import pkg_resources


def get_package_info() -> List[Dict[str, Any]]:
    """获取已安装包的信息"""
    packages = []

    for dist in pkg_resources.working_set:
        try:
            # 获取包元数据
            metadata = dist.get_metadata_lines('METADATA') if dist.has_metadata('METADATA') else []

            # 解析许可证
            license_info = "NOASSERTION"
            for line in metadata:
                if line.startswith('License:'):
                    license_info = line.split(':', 1)[1].strip()
                    break

            # 获取文件列表
            files = []
            if dist.has_metadata('RECORD'):
                for line in dist.get_metadata_lines('RECORD'):
                    file_path = line.split(',')[0]
                    files.append(file_path)

            package = {
                "name": dist.project_name,
                "version": dist.version,
                "license": license_info,
                "location": dist.location,
                "files": files,
                "download_location": f"https://pypi.org/project/{dist.project_name}/{dist.version}/",
            }

            packages.append(package)

        except Exception as e:
            print(f"Warning: Failed to process package {dist.project_name}: {e}", file=sys.stderr)

    return packages


def calculate_file_hash(file_path: str) -> str:
    """计算文件的 SHA256 哈希"""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""


def generate_spdx_document(packages: List[Dict[str, Any]], project_name: str) -> Dict[str, Any]:
    """生成 SPDX 文档"""

    creation_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # SPDX 文档结构
    spdx_doc = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{project_name} SBOM",
        "documentNamespace": f"https://github.com/org/cad-ml-platform/sbom-{creation_time}",
        "creationInfo": {
            "created": creation_time,
            "creators": ["Tool: CAD-ML-SBOM-Generator-1.0"],
            "licenseListVersion": "3.20"
        },
        "packages": []
    }

    # 添加主项目包
    main_package = {
        "SPDXID": "SPDXRef-Package",
        "name": project_name,
        "downloadLocation": "https://github.com/org/cad-ml-platform",
        "filesAnalyzed": False,
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "NOASSERTION"
    }
    spdx_doc["packages"].append(main_package)

    # 添加依赖包
    for idx, pkg in enumerate(packages):
        spdx_package = {
            "SPDXID": f"SPDXRef-Package-{idx}",
            "name": pkg["name"],
            "version": pkg["version"],
            "downloadLocation": pkg["download_location"],
            "filesAnalyzed": False,
            "licenseConcluded": map_to_spdx_license(pkg["license"]),
            "licenseDeclared": pkg["license"],
            "copyrightText": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{pkg['name']}@{pkg['version']}"
                }
            ]
        }
        spdx_doc["packages"].append(spdx_package)

    # 添加关系
    spdx_doc["relationships"] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package"
        }
    ]

    for idx in range(len(packages)):
        spdx_doc["relationships"].append({
            "spdxElementId": "SPDXRef-Package",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": f"SPDXRef-Package-{idx}"
        })

    return spdx_doc


def map_to_spdx_license(license_str: str) -> str:
    """映射到 SPDX 许可证标识符"""
    license_mapping = {
        "MIT License": "MIT",
        "MIT": "MIT",
        "Apache Software License": "Apache-2.0",
        "Apache 2.0": "Apache-2.0",
        "Apache License 2.0": "Apache-2.0",
        "BSD License": "BSD-3-Clause",
        "BSD": "BSD-3-Clause",
        "3-Clause BSD License": "BSD-3-Clause",
        "GNU General Public License v3": "GPL-3.0",
        "GPLv3": "GPL-3.0",
        "GNU Lesser General Public License v3": "LGPL-3.0",
        "LGPLv3": "LGPL-3.0",
        "Python Software Foundation License": "PSF-2.0",
        "ISC License": "ISC",
        "ISC": "ISC",
    }

    # 尝试匹配
    for key, value in license_mapping.items():
        if key.lower() in license_str.lower():
            return value

    # 如果没有匹配，返回原始值或 NOASSERTION
    if license_str and license_str != "UNKNOWN":
        return license_str
    return "NOASSERTION"


def validate_spdx_document(doc: Dict[str, Any]) -> List[str]:
    """验证 SPDX 文档"""
    errors = []

    # 检查必需字段
    required_fields = ["spdxVersion", "dataLicense", "SPDXID", "name",
                      "documentNamespace", "creationInfo"]
    for field in required_fields:
        if field not in doc:
            errors.append(f"Missing required field: {field}")

    # 检查包
    if "packages" not in doc or not doc["packages"]:
        errors.append("No packages found in SBOM")

    # 检查关系
    if "relationships" not in doc or not doc["relationships"]:
        errors.append("No relationships defined in SBOM")

    return errors


def generate_summary_report(packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成摘要报告"""

    # 统计许可证
    license_counts = {}
    for pkg in packages:
        license_name = pkg.get("license", "UNKNOWN")
        license_counts[license_name] = license_counts.get(license_name, 0) + 1

    # 识别高风险许可证
    high_risk_licenses = ["GPL-3.0", "AGPL-3.0", "GPL-2.0", "AGPL-2.0"]
    risky_packages = [
        pkg for pkg in packages
        if any(risk in pkg.get("license", "") for risk in high_risk_licenses)
    ]

    summary = {
        "total_packages": len(packages),
        "license_distribution": license_counts,
        "high_risk_packages": risky_packages,
        "generation_time": datetime.utcnow().isoformat(),
        "statistics": {
            "unique_licenses": len(license_counts),
            "packages_with_unknown_license": sum(
                1 for pkg in packages
                if pkg.get("license") in ["UNKNOWN", "NOASSERTION", None]
            )
        }
    }

    return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate SPDX SBOM for Python project")
    parser.add_argument("--output", default="sbom-spdx.json", help="Output file name")
    parser.add_argument("--format", choices=["json", "yaml", "xml"], default="json",
                       help="Output format")
    parser.add_argument("--project-name", default="CAD ML Platform",
                       help="Project name")
    parser.add_argument("--validate", action="store_true",
                       help="Validate generated SBOM")
    parser.add_argument("--summary", action="store_true",
                       help="Generate summary report")

    args = parser.parse_args()

    print("Collecting package information...")
    packages = get_package_info()
    print(f"Found {len(packages)} packages")

    print("Generating SPDX document...")
    spdx_doc = generate_spdx_document(packages, args.project_name)

    if args.validate:
        print("Validating SPDX document...")
        errors = validate_spdx_document(spdx_doc)
        if errors:
            print("Validation errors found:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("Validation passed")

    # 保存 SBOM
    print(f"Writing SBOM to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(spdx_doc, f, indent=2)

    # 生成摘要
    if args.summary:
        summary = generate_summary_report(packages)
        summary_file = args.output.replace('.json', '-summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary report saved to {summary_file}")

        # 打印摘要
        print("\n=== SBOM Summary ===")
        print(f"Total Packages: {summary['total_packages']}")
        print(f"Unique Licenses: {summary['statistics']['unique_licenses']}")
        print(f"Unknown Licenses: {summary['statistics']['packages_with_unknown_license']}")
        if summary['high_risk_packages']:
            print(f"\n⚠️  High Risk Packages ({len(summary['high_risk_packages'])}):")
            for pkg in summary['high_risk_packages'][:5]:
                print(f"  - {pkg['name']} ({pkg['license']})")

    print("SBOM generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())