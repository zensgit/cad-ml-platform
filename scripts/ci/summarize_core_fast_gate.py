#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


PASS_RE = re.compile(r"(\d+ passed(?:, \d+ deselected)? in [0-9.]+s)")
MARKERS = {
    "tolerance suite": re.compile(r"\bmake test-tolerance\b"),
    "openapi-contract suite": re.compile(r"\bmake validate-openapi\b"),
    "service-mesh suite": re.compile(r"\bmake test-service-mesh\b"),
    "provider-core suite": re.compile(r"\bmake test-provider-core\b"),
    "provider-contract suite": re.compile(r"\bmake test-provider-contract\b"),
}


def _extract_suite_result(lines: list[str], marker_re: re.Pattern[str]) -> str:
    marker_idx = None
    for i, line in enumerate(lines):
        if marker_re.search(line):
            marker_idx = i
            break
    if marker_idx is None:
        return "N/A"
    for line in lines[marker_idx + 1 :]:
        if re.search(r"\bmake\s+\S+", line):
            break
        m = PASS_RE.search(line)
        if m:
            return m.group(1)
    return "N/A"


def _bool_mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def build_summary(log_text: str, title: str) -> str:
    lines = log_text.splitlines()
    iso_ok = any("ISO286 deviations:" in line for line in lines) and any(
        line.strip() == "OK" for line in lines
    )
    hole_ok = any("All required hole symbols present:" in line for line in lines)

    suite_results = {
        suite: _extract_suite_result(lines, marker_re)
        for suite, marker_re in MARKERS.items()
    }

    out: list[str] = []
    out.append(f"## {title}")
    out.append("")
    out.append("| Check | Status | Evidence |")
    out.append("|---|---|---|")
    out.append(
        f"| ISO286 deviations validator | {_bool_mark(iso_ok)} | `ISO286 deviations ... OK` |"
    )
    out.append(
        f"| ISO286 hole symbols validator | {_bool_mark(hole_ok)} | `All required hole symbols present` |"
    )
    for suite, result in suite_results.items():
        out.append(
            f"| {suite} | {_bool_mark(result != 'N/A')} | `{result}` |"
        )
    out.append("")
    out.append("Tail:")
    out.append("```text")
    tail = lines[-30:] if lines else ["<empty log>"]
    out.extend(tail)
    out.append("```")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize core-fast-gate log into Markdown.")
    parser.add_argument("--log-file", required=True, help="Path to core-fast-gate log file")
    parser.add_argument("--title", required=True, help="Markdown title suffix")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"## {args.title}\n\nNo core-fast-gate log found.\n")
        return 0

    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(build_summary(text, args.title), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
