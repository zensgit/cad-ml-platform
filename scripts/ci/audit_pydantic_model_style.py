#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PATTERNS = (
    "dict_model_config",
    "mutable_literal_default",
    "mutable_field_default",
    "non_optional_none_default",
)


@dataclass(frozen=True)
class Finding:
    pattern: str
    path: str
    line: int
    snippet: str


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.is_file():
                yield path


def _is_mutable_expr(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"list", "dict", "set"}
    return False


def _is_optional_annotation(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return True
    text = ast.unparse(annotation)
    return (
        "Optional[" in text
        or "| None" in text
        or text == "Any"
        or text.endswith(".Any")
        or text == "None"
    )


def _is_basemodel_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


def collect_findings(roots: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []

    for path in _iter_python_files(roots):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
            lines = text.splitlines()
        except (OSError, SyntaxError):
            continue

        rel_path = path.as_posix()
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_basemodel_class(node):
                continue

            for stmt in node.body:
                # model_config style
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "model_config"
                    and isinstance(stmt.value, ast.Dict)
                ):
                    line_no = getattr(stmt, "lineno", 1)
                    snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                    findings.append(
                        Finding(
                            pattern="dict_model_config",
                            path=rel_path,
                            line=line_no,
                            snippet=snippet,
                        )
                    )
                    continue

                # field: foo: T = None with non-optional T
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is None
                    and stmt.target.id != "model_config"
                    and not _is_optional_annotation(stmt.annotation)
                ):
                    line_no = getattr(stmt, "lineno", 1)
                    snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                    findings.append(
                        Finding(
                            pattern="non_optional_none_default",
                            path=rel_path,
                            line=line_no,
                            snippet=snippet,
                        )
                    )

                # mutable defaults via annotation assignment
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id != "model_config"
                    and _is_mutable_expr(stmt.value)
                ):
                    line_no = getattr(stmt, "lineno", 1)
                    snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                    findings.append(
                        Finding(
                            pattern="mutable_literal_default",
                            path=rel_path,
                            line=line_no,
                            snippet=snippet,
                        )
                    )

                # mutable defaults via plain assignment
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id != "model_config"
                    and _is_mutable_expr(stmt.value)
                ):
                    line_no = getattr(stmt, "lineno", 1)
                    snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                    findings.append(
                        Finding(
                            pattern="mutable_literal_default",
                            path=rel_path,
                            line=line_no,
                            snippet=snippet,
                        )
                    )

                # mutable defaults via Field(default=...)
                value = None
                if isinstance(stmt, ast.AnnAssign):
                    value = stmt.value
                elif isinstance(stmt, ast.Assign):
                    value = stmt.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "Field"
                ):
                    for kw in value.keywords:
                        if kw.arg == "default" and _is_mutable_expr(kw.value):
                            line_no = getattr(stmt, "lineno", 1)
                            snippet = (
                                lines[line_no - 1].strip() if line_no <= len(lines) else ""
                            )
                            findings.append(
                                Finding(
                                    pattern="mutable_field_default",
                                    path=rel_path,
                                    line=line_no,
                                    snippet=snippet,
                                )
                            )
                            break
    return findings


def summarize_counts(findings: Iterable[Finding]) -> dict[str, int]:
    counts = {name: 0 for name in PATTERNS}
    for item in findings:
        counts[item.pattern] = counts.get(item.pattern, 0) + 1
    return counts


def build_baseline_payload(roots: list[str], counts: dict[str, int]) -> dict[str, object]:
    normalized = {name: int(counts.get(name, 0)) for name in PATTERNS}
    return {
        "roots": roots,
        "patterns": list(PATTERNS),
        "counts": normalized,
        "total_findings": sum(normalized.values()),
    }


def load_baseline(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_counts = data.get("counts", {})
    if not isinstance(raw_counts, dict):
        return {name: 0 for name in PATTERNS}
    return {name: int(raw_counts.get(name, 0)) for name in PATTERNS}


def find_regressions(current: dict[str, int], baseline: dict[str, int]) -> dict[str, tuple[int, int]]:
    regressions: dict[str, tuple[int, int]] = {}
    for name in PATTERNS:
        base = int(baseline.get(name, 0))
        now = int(current.get(name, 0))
        if now > base:
            regressions[name] = (base, now)
    return regressions


def _print_summary(counts: dict[str, int]) -> None:
    print("Pydantic model style audit summary")
    print("----------------------------------")
    for name in PATTERNS:
        print(f"{name}: {counts.get(name, 0)}")
    print(f"total_findings: {sum(counts.values())}")


def _print_findings(findings: list[Finding], limit: int) -> None:
    if not findings:
        return
    print("")
    print(f"Sample findings (limit {limit})")
    print("-------------------------------")
    for item in findings[:limit]:
        print(f"{item.pattern}: {item.path}:{item.line}: {item.snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pydantic model style consistency.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["src"],
        help="Root paths to scan (default: src)",
    )
    parser.add_argument(
        "--baseline",
        default="config/pydantic_model_style_baseline.json",
        help="Baseline json path for regression checks",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Fail when current counts exceed baseline counts",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write baseline file with current counts",
    )
    parser.add_argument(
        "--show-findings-limit",
        type=int,
        default=25,
        help="Number of findings to print",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    findings = collect_findings(roots)
    counts = summarize_counts(findings)
    _print_summary(counts)
    _print_findings(findings, args.show_findings_limit)

    baseline_path = Path(args.baseline)
    if args.write_baseline:
        payload = build_baseline_payload(args.roots, counts)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("")
        print(f"Baseline written: {baseline_path.as_posix()}")

    if not args.check_regression:
        return 0

    if not baseline_path.exists():
        print("")
        print(f"ERROR: baseline not found: {baseline_path.as_posix()}")
        return 2

    baseline_counts = load_baseline(baseline_path)
    regressions = find_regressions(counts, baseline_counts)
    if regressions:
        print("")
        print("Regression detected:")
        for name in PATTERNS:
            if name not in regressions:
                continue
            base, now = regressions[name]
            print(f"- {name}: baseline={base}, current={now}")
        return 1

    print("")
    print("No pydantic model-style regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
