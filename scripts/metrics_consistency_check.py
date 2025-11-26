"""
Metrics consistency checker.

Validates that all metric variables defined in src/utils/analysis_metrics.py
are exported in __all__, and reports any extras or misses.

Usage:
  python scripts/metrics_consistency_check.py
"""

from __future__ import annotations

import ast
from pathlib import Path


def parse_metrics_file(path: Path) -> tuple[set[str], list[str]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    defined: set[str] = set()
    exported: list[str] = []

    # Collect metric variable names assigned at module level
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Only single-name targets
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                # Heuristic: metrics are typically assigned via Counter/Histogram/Gauge calls
                if isinstance(node.value, (ast.Call, ast.Name)):
                    defined.add(name)

        # Find __all__ list literal and capture its string elements
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        exported.append(elt.value)

    # Exclude dunder and helper names commonly present
    excluded_prefixes = {"_", "ast", "Path"}
    defined = {n for n in defined if n and not n.startswith(tuple(excluded_prefixes))}

    return defined, exported


def main() -> None:
    metrics_path = Path("src/utils/analysis_metrics.py")
    defined, exported = parse_metrics_file(metrics_path)
    exported_set = set(exported)

    # Only consider names that look like metrics (lowercase with underscores)
    defined_metrics = {n for n in defined if n.islower() and "__" not in n}

    missing = sorted(n for n in defined_metrics if n not in exported_set)
    extras = sorted(n for n in exported if n not in defined_metrics)

    result = {
        "defined_count": len(defined_metrics),
        "exported_count": len(exported),
        "missing_in___all__": missing,
        "extras_in___all__": extras,
    }
    print(result)

    # Non-zero exit when inconsistencies found (useful in CI)
    if missing or extras:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

