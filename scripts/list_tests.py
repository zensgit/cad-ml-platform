#!/usr/bin/env python3
"""
Generate test statistics for TEST_MAP.md

Outputs:
1. Pytest node count per file (execution units)
2. Function count per file (def test_* count)

Usage:
    python3 scripts/list_tests.py
    python3 scripts/list_tests.py --markdown  # Output as markdown table
"""

import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def get_pytest_nodes(test_dir: str) -> dict:
    """Get pytest node count per file."""
    result = subprocess.run(
        ["pytest", test_dir, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    counts = defaultdict(int)
    for line in result.stdout.strip().split("\n"):
        if "::" in line:
            file_path = line.split("::")[0]
            counts[file_path] += 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def get_function_count(test_dir: str) -> dict:
    """Count test functions per file (def test_* / async def test_*)."""
    project_root = Path(__file__).parent.parent
    test_path = project_root / test_dir

    counts = {}
    for py_file in sorted(test_path.glob("test_*.py")):
        count = 0
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Match def test_ or async def test_ with any indentation
                    stripped = line.lstrip()
                    if stripped.startswith("def test_") or stripped.startswith("async def test_"):
                        count += 1
        except Exception:
            count = 0
        rel_path = f"{test_dir}/{py_file.name}"
        counts[rel_path] = count

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def main():
    markdown_mode = "--markdown" in sys.argv

    modules = [
        ("tests/vision", "Vision"),
        ("tests/ocr", "OCR")
    ]

    if markdown_mode:
        print("# Test Statistics (Auto-generated)")
        print(f"\nGenerated: {subprocess.run(['date', '+%Y-%m-%d %H:%M'], capture_output=True, text=True).stdout.strip()}")
        print()

    for test_dir, module_name in modules:
        pytest_nodes = get_pytest_nodes(test_dir)
        func_counts = get_function_count(test_dir)

        total_nodes = sum(pytest_nodes.values())
        total_funcs = sum(func_counts.values())

        if markdown_mode:
            print(f"## {module_name} Module")
            print(f"\n**Total**: {total_nodes} pytest nodes, {total_funcs} test functions\n")
            print("| File | Pytest Nodes | Test Functions | Notes |")
            print("|------|-------------|----------------|-------|")

            for file_path in pytest_nodes:
                file_name = Path(file_path).name
                nodes = pytest_nodes.get(file_path, 0)
                funcs = func_counts.get(file_path, 0)

                # Note if parametrized (nodes > funcs)
                note = ""
                if nodes > funcs:
                    note = f"parametrized ({nodes - funcs} expanded)"

                print(f"| `{file_name}` | {nodes} | {funcs} | {note} |")

            print()
        else:
            print(f"=== {module_name} Module ===")
            print(f"Total: {total_nodes} pytest nodes, {total_funcs} test functions")
            print()
            print(f"{'File':<45} {'Nodes':>8} {'Funcs':>8}")
            print("-" * 65)

            for file_path in pytest_nodes:
                file_name = Path(file_path).name
                nodes = pytest_nodes.get(file_path, 0)
                funcs = func_counts.get(file_path, 0)
                print(f"{file_name:<45} {nodes:>8} {funcs:>8}")

            print()

    # Summary
    if markdown_mode:
        print("## Summary")
        print("\n**口径说明**:")
        print("- **Pytest Nodes**: pytest --collect-only 的执行单元数（包含参数化展开）")
        print("- **Test Functions**: def test_* / async def test_* 的函数定义数")
        print("- 当 Nodes > Functions 时，表示存在 @pytest.mark.parametrize 展开")
    else:
        print("=== Summary ===")
        print("Pytest Nodes: Execution units (includes parametrize expansion)")
        print("Test Functions: Function definitions (def test_*)")
        print("Nodes > Functions indicates @pytest.mark.parametrize usage")


if __name__ == "__main__":
    main()
