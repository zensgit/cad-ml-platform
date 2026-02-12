#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


COUNT_RE = re.compile(r"^(?P<name>[a-z0-9_]+):\s+(?P<count>\d+)\s*$")


def _bool_mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def build_summary(log_text: str, title: str) -> str:
    counts: dict[str, int] = {}
    total_findings: int | None = None
    no_regression = "No pydantic model-style regressions detected." in log_text
    regression_detected = "Regression detected:" in log_text

    for line in log_text.splitlines():
        m = COUNT_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        value = int(m.group("count"))
        if name == "total_findings":
            total_findings = value
        else:
            counts[name] = value

    out: list[str] = []
    out.append(f"## {title}")
    out.append("")
    out.append("| Check | Status | Evidence |")
    out.append("|---|---|---|")
    out.append(
        f"| Regression gate | {_bool_mark(no_regression and not regression_detected)} | "
        f"`{'no regressions' if no_regression and not regression_detected else 'regression detected or missing signal'}` |"
    )
    out.append(
        f"| Total findings | {_bool_mark((total_findings or 0) == 0)} | `{total_findings if total_findings is not None else 'N/A'}` |"
    )
    for key in sorted(counts):
        out.append(f"| {key} | {_bool_mark(counts[key] == 0)} | `{counts[key]}` |")
    out.append("")
    out.append("Tail:")
    out.append("```text")
    lines = log_text.splitlines()
    tail = lines[-20:] if lines else ["<empty log>"]
    out.extend(tail)
    out.append("```")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize pydantic style audit log into markdown table."
    )
    parser.add_argument("--log-file", required=True, help="Path to audit log")
    parser.add_argument("--title", required=True, help="Section title")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"## {args.title}\n\nNo pydantic style audit log found.\n")
        return 0

    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(build_summary(text, args.title), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
