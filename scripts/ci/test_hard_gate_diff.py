#!/usr/bin/env python3
"""Tests for the A4 hard-gate diff filter -- the load-bearing, must-be-correct part.

Runnable with plain `python3` (no pytest, no heavy deps) so it executes anywhere,
including local py3.9. Exercises the two invariants that make the gate safe to arm:

  1. a violation on a line THIS PR changed is caught (gate can fail);
  2. a violation on a line the PR did NOT touch is ignored (no punishing
     pre-existing debt -> never reds an unrelated PR).

Plus the git-diff hunk parser, driven by a fake `run` so no real repo is needed.
"""
import sys
import types

sys.path.insert(0, ".")
from scripts.ci.hard_gate_diff import Finding, changed_lines, new_violations  # noqa: E402

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


# --- invariant 1 & 2: the diff-line filter -------------------------------------
changed = {"src/core/foo.py": {10, 11, 12}, "src/core/bar.py": {5}}

on_changed = Finding("src/core/foo.py", 11, "unused function 'x'", "vulture")
on_unchanged = Finding("src/core/foo.py", 400, "unused function 'legacy'", "vulture")  # pre-existing debt
other_file = Finding("src/core/untouched.py", 3, "unused var", "vulture")
dup_on_changed = Finding("src/core/bar.py", 5, "R0801 similar lines", "duplicate-code")

kept = new_violations([on_changed, on_unchanged, other_file, dup_on_changed], changed)

check("catches a violation on a CHANGED line (gate can fail)", on_changed in kept)
check("catches a duplicate on a CHANGED line", dup_on_changed in kept)
check("IGNORES pre-existing debt on an UNCHANGED line (no 误伤)", on_unchanged not in kept)
check("IGNORES a violation in an UNTOUCHED file (no 误伤)", other_file not in kept)
check("exactly the two changed-line findings survive", len(kept) == 2)

# --- the hunk parser: added lines only, deletions contribute nothing -----------
FAKE_DIFF = (
    "diff --git a/src/core/foo.py b/src/core/foo.py\n"
    "--- a/src/core/foo.py\n"
    "+++ b/src/core/foo.py\n"
    "@@ -10,0 +11,3 @@\n"       # added 3 lines starting at 11
    "+def new_a(): ...\n"
    "+def new_b(): ...\n"
    "+def new_c(): ...\n"
    "@@ -50,2 +54,0 @@\n"       # pure deletion -> no new lines
    "diff --git a/src/core/bar.py b/src/core/bar.py\n"
    "--- a/src/core/bar.py\n"
    "+++ b/src/core/bar.py\n"
    "@@ -5 +5 @@\n"             # single-line change at 5 (no count = 1)
    "+x = 2\n"
)


def fake_run(cmd, capture_output, text, check):  # noqa: ARG001
    return types.SimpleNamespace(stdout=FAKE_DIFF)


parsed = changed_lines("origin/main", run=fake_run)
check("hunk parser: foo.py added lines == {11,12,13}", parsed.get("src/core/foo.py") == {11, 12, 13})
check("hunk parser: bar.py changed line == {5}", parsed.get("src/core/bar.py") == {5})
check("hunk parser: deletion-only hunk adds nothing (no line 50/51)",
      50 not in parsed.get("src/core/foo.py", set()))

print("\nOBSERVED-RED demonstration (the filter genuinely discriminates):")
print(f"  same violation on changed line 11 -> caught={on_changed in kept}")
print(f"  same violation on unchanged line 400 -> caught={on_unchanged in kept}  (must be False)")

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nALL PASS -- diff filter + hunk parser correct")
sys.exit(0)
