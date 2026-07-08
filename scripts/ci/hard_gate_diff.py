#!/usr/bin/env python3
"""Diff-scoped hard gate for dead-code (vulture) and duplicate-code (pylint).

Phase 0 slice A4. The repo's existing dead-code / duplicate-code checks
(code-quality.yml) run over the WHOLE, already-bloated tree and end in ``|| true``
-- so they can neither be made blocking (they'd red every PR on pre-existing
debt) nor stop the fleet re-introducing new debt.

This gate is the missing MECHANIC: it fails ONLY on violations located on lines
the current PR actually added or changed, so it never punishes a PR for
pre-existing debt in files it didn't touch. That is what makes it safe to arm as
a *required* check (an owner action -- this script does not touch branch
protection).

Two modes:
  * dry-run (default): report what it WOULD block, exit 0. Arm it here first.
  * enforce (HARD_GATE_ENFORCE=1): exit 1 on any new violation.

The finding-producers (vulture, pylint) are shelled out and are best-effort: if a
tool is missing the gate reports "tool unavailable" and does NOT fail (a missing
linter must not mask or fabricate a violation). The load-bearing, testable logic
is the diff-line filter, exercised by scripts/ci/test_hard_gate_diff.py.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    file: str          # repo-relative path
    line: int          # 1-indexed
    message: str
    source: str        # "vulture" | "duplicate-code"


# ---------------------------------------------------------------------------
# The load-bearing logic: which lines did this PR add/change?
# ---------------------------------------------------------------------------
def changed_lines(base_ref: str, run=subprocess.run) -> dict[str, set[int]]:
    """Map repo-relative *.py path -> set of added/changed line numbers vs base.

    Parses ``git diff --unified=0`` hunk headers (``@@ -a,b +c,d @@``); ``c..c+d-1``
    are the new-file lines this diff introduced. Deletion-only hunks (``d == 0``)
    contribute nothing.
    """
    out = run(
        ["git", "diff", "--unified=0", f"{base_ref}...HEAD", "--", "*.py"],
        capture_output=True, text=True, check=False,
    ).stdout
    result: dict[str, set[int]] = {}
    cur: str | None = None
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            result.setdefault(cur, set())
        elif line.startswith("@@") and cur is not None:
            m = hunk.match(line)
            if not m:
                continue
            start = int(m.group(1))
            count = 1 if m.group(2) is None else int(m.group(2))
            for i in range(start, start + count):
                result[cur].add(i)
    return {f: lines for f, lines in result.items() if lines}


def new_violations(findings: list[Finding], changed: dict[str, set[int]]) -> list[Finding]:
    """Keep only findings whose (file, line) was added/changed by this PR."""
    return [f for f in findings if f.line in changed.get(f.file, ())]


# ---------------------------------------------------------------------------
# Finding-producers (best-effort; absence never fails the gate)
# ---------------------------------------------------------------------------
def _tool_available(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def run_vulture(paths: list[str], run=subprocess.run) -> tuple[list[Finding], bool]:
    if not _tool_available("vulture") or not paths:
        return [], _tool_available("vulture")
    out = run(["vulture", *paths, "--min-confidence", "80"],
              capture_output=True, text=True, check=False).stdout
    findings = []
    # vulture line format: path:line: message (NN% confidence)
    pat = re.compile(r"^(.+?):(\d+): (.+)$")
    for line in out.splitlines():
        m = pat.match(line)
        if m:
            findings.append(Finding(m.group(1), int(m.group(2)), m.group(3), "vulture"))
    return findings, True


def run_duplicate(paths: list[str], run=subprocess.run) -> tuple[list[Finding], bool]:
    if not _tool_available("pylint") or not paths:
        return [], _tool_available("pylint")
    out = run(["pylint", *paths, "--disable=all", "--enable=duplicate-code",
               "--min-similarity-lines=10", "--output-format=text"],
              capture_output=True, text=True, check=False).stdout
    findings = []
    pat = re.compile(r"^(.+?):(\d+):\d+: (R0801.+)$")
    for line in out.splitlines():
        m = pat.match(line)
        if m:
            findings.append(Finding(m.group(1), int(m.group(2)), m.group(3), "duplicate-code"))
    return findings, True


def main() -> int:
    base = os.environ.get("HARD_GATE_BASE", "origin/main")
    enforce = os.environ.get("HARD_GATE_ENFORCE") == "1"

    changed = changed_lines(base)
    changed_files = sorted(changed)
    if not changed_files:
        print("hard-gate: no changed .py lines vs", base, "-> nothing to check")
        return 0

    vult, vult_ok = run_vulture(changed_files)
    dup, dup_ok = run_duplicate(changed_files)
    new = new_violations(vult + dup, changed)

    print(f"hard-gate: mode={'ENFORCE' if enforce else 'dry-run'} base={base}")
    print(f"  changed .py files: {len(changed_files)}")
    for tool, ok in [("vulture", vult_ok), ("duplicate-code", dup_ok)]:
        if not ok:
            print(f"  ::warning:: {tool} unavailable -- not run (gate does not fabricate a pass/fail)")
    if not new:
        print("  no NEW dead-code / duplicate-code on changed lines. OK")
        return 0

    print(f"  {len(new)} NEW violation(s) on changed lines:")
    for f in new:
        sev = "error" if enforce else "warning"
        print(f"    ::{sev}::{f.file}:{f.line}: [{f.source}] {f.message}")

    if enforce:
        print("hard-gate: FAIL (enforce mode)")
        return 1
    print("hard-gate: dry-run -- would block in enforce mode; not failing yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
