#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PATTERNS: dict[str, re.Pattern[str]] = {
    "pydantic_v1_import": re.compile(r"\bfrom\s+pydantic\.v1\s+import\b"),
    "validator_decorator": re.compile(r"@\s*validator\("),
    "root_validator_decorator": re.compile(r"@\s*root_validator\("),
    "class_config": re.compile(r"^\s*class\s+Config\s*:\s*$"),
    "parse_obj_call": re.compile(r"\.parse_obj\("),
    "parse_raw_call": re.compile(r"\.parse_raw\("),
    "from_orm_call": re.compile(r"\.from_orm\("),
    "fields_attr": re.compile(r"\.__fields__\b"),
}


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


def collect_findings(
    roots: Iterable[Path],
    patterns: dict[str, re.Pattern[str]] | None = None,
) -> list[Finding]:
    use_patterns = patterns or PATTERNS
    findings: list[Finding] = []

    for path in _iter_python_files(roots):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel_path = path.as_posix()
        for line_no, line in enumerate(lines, start=1):
            for name, regex in use_patterns.items():
                if regex.search(line):
                    findings.append(
                        Finding(
                            pattern=name,
                            path=rel_path,
                            line=line_no,
                            snippet=line.strip(),
                        )
                    )
    return findings


def summarize_counts(findings: Iterable[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in PATTERNS}
    for item in findings:
        counts[item.pattern] = counts.get(item.pattern, 0) + 1
    return counts


def build_baseline_payload(roots: list[str], counts: dict[str, int]) -> dict[str, object]:
    normalized = {name: int(counts.get(name, 0)) for name in PATTERNS}
    return {
        "roots": roots,
        "patterns": list(PATTERNS.keys()),
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
    print("Pydantic v2 compatibility audit summary")
    print("--------------------------------------")
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
    parser = argparse.ArgumentParser(description="Audit pydantic v2 compatibility risk patterns.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["src"],
        help="Root paths to scan (default: src)",
    )
    parser.add_argument(
        "--baseline",
        default="config/pydantic_v2_audit_baseline.json",
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
    print("No pydantic v2 compatibility regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
