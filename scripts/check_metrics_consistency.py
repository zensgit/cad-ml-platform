#!/usr/bin/env python3
"""验证metrics定义与__all__导出一致性.

Ensures all defined Prometheus metrics are properly exported in __all__.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Set, List


def extract_metric_definitions(file_path: Path) -> Set[str]:
    """提取Counter/Histogram/Gauge定义的变量名."""
    with open(file_path) as f:
        content = f.read()

    # 正则匹配 metric_name = Counter(...), Histogram(...), Gauge(...)
    pattern = r'^(\w+)\s*=\s*(?:Counter|Histogram|Gauge)\('
    matches = re.finditer(pattern, content, re.MULTILINE)
    return {m.group(1) for m in matches}


def extract_all_exports(file_path: Path) -> Set[str]:
    """提取__all__列表中的导出项."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    # Handle both list literals and other structures
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        exports = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                exports.append(elt.value)
                            elif isinstance(elt, ast.Str):  # Python <3.8
                                exports.append(elt.s)
                        return set(exports)
    return set()


def main() -> int:
    """主函数，检查metrics一致性."""
    metrics_file = Path("src/utils/analysis_metrics.py")

    if not metrics_file.exists():
        print(f"❌ File not found: {metrics_file}")
        return 1

    print("🔍 Checking metrics consistency...")
    print()

    # 提取定义和导出
    defined = extract_metric_definitions(metrics_file)
    exported = extract_all_exports(metrics_file)

    print(f"📊 Found {len(defined)} metric definitions")
    print(f"📦 Found {len(exported)} exports in __all__")
    print()

    # 检查不一致
    missing_exports = defined - exported
    extra_exports = exported - defined

    # 过滤掉非metric的导出（如函数、类等）
    # 假设metric名称都以小写字母开头且包含下划线
    extra_exports_filtered = {
        e for e in extra_exports
        if e.islower() and '_' in e
    }

    exit_code = 0

    if missing_exports:
        print("❌ Metrics defined but NOT exported in __all__:")
        for metric in sorted(missing_exports):
            print(f"   - {metric}")
        print()
        exit_code = 1

    if extra_exports_filtered:
        print("⚠️  Metrics in __all__ but NOT defined:")
        for metric in sorted(extra_exports_filtered):
            print(f"   - {metric}")
        print()
        # This is a warning, not blocking

    if not missing_exports and not extra_exports_filtered:
        print(f"✅ All {len(defined)} metrics are properly exported")
        print()

    # List all metrics for reference
    if defined:
        print("📋 All defined metrics:")
        for metric in sorted(defined):
            print(f"   ✓ {metric}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
