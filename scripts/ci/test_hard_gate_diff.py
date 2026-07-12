#!/usr/bin/env python3
"""Tests for the A4 hard-gate diff filter -- the load-bearing, must-be-correct part.

Runnable with plain `python3` (no pytest, no heavy deps) so it executes anywhere,
including local py3.9. Exercises the two invariants that make the gate safe to arm:

  1. a violation on a line THIS PR changed is caught (gate can fail);
  2. a violation on a line the PR did NOT touch is ignored (no punishing
     pre-existing debt -> never reds an unrelated PR).

Plus the git-diff hunk parser, driven by a fake `run` so no real repo is needed.
"""
import os
import sys
import types

sys.path.insert(0, ".")
from scripts.ci import hard_gate_diff as hg  # noqa: E402
from scripts.ci.hard_gate_diff import (  # noqa: E402
    Finding,
    HardGateError,
    changed_lines,
    new_violations,
)

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
    # command-aware: the base-resolves probe (rev-parse) succeeds, the diff returns hunks.
    if "rev-parse" in cmd:
        return types.SimpleNamespace(stdout="deadbeef\n", stderr="", returncode=0)
    return types.SimpleNamespace(stdout=FAKE_DIFF, stderr="", returncode=0)


parsed = changed_lines("origin/main", run=fake_run)
check("hunk parser: foo.py added lines == {11,12,13}", parsed.get("src/core/foo.py") == {11, 12, 13})
check("hunk parser: bar.py changed line == {5}", parsed.get("src/core/bar.py") == {5})
check("hunk parser: deletion-only hunk adds nothing (no line 50/51)",
      50 not in parsed.get("src/core/foo.py", set()))

# --- fail-closed invariant 1: an UNRESOLVABLE base must raise, not return {} -----
def run_base_missing(cmd, capture_output, text, check):  # noqa: ARG001
    if "rev-parse" in cmd:                       # base does not resolve
        return types.SimpleNamespace(stdout="", stderr="", returncode=128)
    raise AssertionError("must not run `git diff` once the base is known-unresolvable")


try:
    changed_lines("origin/gone", run=run_base_missing)
    check("unresolvable base FAILS CLOSED (raises, not {} = pass-everything)", False)
except HardGateError:
    check("unresolvable base FAILS CLOSED (raises, not {} = pass-everything)", True)

# --- fail-closed invariant 2: a `git diff` failure must raise --------------------
def run_diff_fails(cmd, capture_output, text, check):  # noqa: ARG001
    if "rev-parse" in cmd:
        return types.SimpleNamespace(stdout="ok\n", stderr="", returncode=0)
    return types.SimpleNamespace(stdout="", stderr="fatal: bad object", returncode=128)


try:
    changed_lines("origin/main", run=run_diff_fails)
    check("git diff failure FAILS CLOSED (raises)", False)
except HardGateError:
    check("git diff failure FAILS CLOSED (raises)", True)

# --- fail-closed invariant 3: enforce mode + missing tool must raise -------------
_orig_avail = hg._tool_available
_orig_cl = hg.changed_lines
try:
    hg._tool_available = lambda name: False                     # no vulture / pylint
    hg.changed_lines = lambda base: {"src/x.py": {1}}           # pretend a .py changed
    os.environ["HARD_GATE_ENFORCE"] = "1"
    os.environ["HARD_GATE_BASE"] = "origin/main"
    try:
        hg.main()
        check("enforce mode + missing finding-producer FAILS CLOSED (raises)", False)
    except HardGateError:
        check("enforce mode + missing finding-producer FAILS CLOSED (raises)", True)
finally:
    hg._tool_available = _orig_avail
    hg.changed_lines = _orig_cl
    os.environ.pop("HARD_GATE_ENFORCE", None)
    os.environ.pop("HARD_GATE_BASE", None)

print("\nOBSERVED-RED demonstration (the filter genuinely discriminates):")
print(f"  same violation on changed line 11 -> caught={on_changed in kept}")
print(f"  same violation on unchanged line 400 -> caught={on_unchanged in kept}  (must be False)")

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nALL PASS -- diff filter + hunk parser correct")
sys.exit(0)
