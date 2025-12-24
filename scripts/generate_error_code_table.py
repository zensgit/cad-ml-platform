"""
Generate a Markdown table of error codes and sources from errors_extended.py.

Usage:
  python3 scripts/generate_error_code_table.py [output_path]

If output_path is omitted, writes to docs/ERROR_CODES.md.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def parse_errors(py_path: Path) -> tuple[list[str], dict[str, str]]:
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))

    error_codes: list[str] = []
    source_map: dict[str, str] = {}

    # Find ErrorCode enum members and ERROR_SOURCE_MAPPING
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ErrorCode":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    error_codes.append(name)
        mapping_node = None
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "ERROR_SOURCE_MAPPING"
            ):
                mapping_node = node.value
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "ERROR_SOURCE_MAPPING"
            ):
                mapping_node = node.value
        if mapping_node and isinstance(mapping_node, ast.Dict):
            for k, v in zip(mapping_node.keys, mapping_node.values):
                # Keys like ErrorCode.SOMETHING, values like ErrorSource.INPUT
                if isinstance(k, ast.Attribute) and isinstance(v, ast.Attribute):
                    code_name = k.attr
                    source_name = v.attr
                    source_map[code_name] = source_name

    # Deduplicate while preserving order
    seen = set()
    ordered_codes = []
    for c in error_codes:
        if c not in seen:
            ordered_codes.append(c)
            seen.add(c)

    return ordered_codes, source_map


def render_markdown(codes: list[str], source_map: dict[str, str]) -> str:
    lines = []
    lines.append("# Error Codes Reference")
    lines.append("")
    lines.append("Auto-generated from src/core/errors_extended.py. Do not edit by hand.")
    lines.append("")
    lines.append("| Code | Source |")
    lines.append("|------|--------|")
    for c in codes:
        src = source_map.get(c, "-")
        lines.append(f"| {c} | {src} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors_py = root / "src/core/errors_extended.py"
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (root / "docs/ERROR_CODES.md")
    codes, src_map = parse_errors(errors_py)
    md = render_markdown(codes, src_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
