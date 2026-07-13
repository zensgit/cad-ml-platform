#!/usr/bin/env python3
"""Tests for the A4 hard-gate diff filter -- the load-bearing, must-be-correct part.

Runnable with plain `python3` (no pytest, no heavy deps) so it executes anywhere,
including local py3.9. Exercises the two invariants that make the gate safe to arm:

  1. a violation on a line THIS PR changed is caught (gate can fail);
  2. a violation on a line the PR did NOT touch is ignored (no punishing
     pre-existing debt -> never reds an unrelated PR).

Plus the git-diff hunk parser, driven by a fake `run` so no real repo is needed.
"""
import json
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

# --- producer malfunction (tool present but crashed) FAILS CLOSED ----------------
def _crash(rc, stderr=""):
    def _r(cmd, capture_output, text, check):  # noqa: ARG001
        return types.SimpleNamespace(stdout="", stderr=stderr, returncode=rc)
    return _r


hg._tool_available = lambda name: True  # pretend the tools exist so we reach the run
try:
    for label, prod, run in [
        ("vulture bad-args (rc=2)", hg.run_vulture, _crash(2, "usage")),
        ("vulture traceback", hg.run_vulture, _crash(1, "Traceback (most recent call last):")),
        ("pylint usage-error (rc=32)", hg.run_duplicate, _crash(32, "bad flag")),
        ("pylint fatal (rc=1)", hg.run_duplicate, _crash(1, "fatal")),
    ]:
        try:
            prod(["src/x.py"], run=run)
            check(f"producer malfunction FAILS CLOSED: {label}", False)
        except HardGateError:
            check(f"producer malfunction FAILS CLOSED: {label}", True)

    # normal exit codes must NOT be misread as a malfunction
    ok_clean = hg.run_duplicate(["src/x.py"], run=_crash(0))  # rc=0, empty JSON handled
    check("pylint rc=0 (clean) is NOT a malfunction", ok_clean == ([], True))

    # multi-location parse: pylint anchors R0801 on the UNCHANGED file (order-dependent), the
    # finding must STILL reach the CHANGED file or the changed-line filter silently drops it.
    def run_anchor_on_unchanged(cmd, capture_output, text, check):  # noqa: ARG001
        msg = "Similar lines in 2 files\n==orig:[1:12]\n==copy_new:[1:12]"
        payload = json.dumps([{"symbol": "duplicate-code", "path": "src/core/orig.py",
                               "line": 1, "message": msg}])
        return types.SimpleNamespace(stdout=payload, stderr="", returncode=8)

    dup_findings, _ = hg.run_duplicate(["src/core/orig.py", "src/core/copy_new.py"],
                                       run=run_anchor_on_unchanged)
    dup_files = {f.file for f in dup_findings}
    check("multi-location parse emits BOTH files (anchor-order irrelevant)",
          {"src/core/orig.py", "src/core/copy_new.py"} <= dup_files)
    kept_dup = new_violations(dup_findings, {"src/core/copy_new.py": set(range(1, 13))})
    check("changed-line filter keeps the dup on the CHANGED file despite anchor on the unchanged",
          any(f.file == "src/core/copy_new.py" for f in kept_dup))
finally:
    hg._tool_available = _orig_avail

# --- test files are OUT of scope -------------------------------------------------
check("is_test_path: test_*.py excluded", hg.is_test_path("scripts/ci/test_x.py"))
check("is_test_path: tests/ dir excluded", hg.is_test_path("tests/unit/x.py"))
check("is_test_path: conftest.py excluded", hg.is_test_path("conftest.py"))
check("is_test_path: production file NOT excluded", not hg.is_test_path("src/core/foo.py"))

# --- REAL-TOOL golden (reviewer-required): new file copying an UNCHANGED file -> RED ---
if hg._tool_available("vulture") and hg._tool_available("pylint"):
    import subprocess
    import tempfile

    d = tempfile.mkdtemp()

    def _git(*a):
        return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)

    _git("init", "-q"); _git("config", "user.email", "t@t"); _git("config", "user.name", "t")
    os.makedirs(f"{d}/src/core")
    # >= min-similarity-lines=10 (well clear of the threshold), so pylint reliably flags it.
    _body = (
        "def compute_totals(rows):\n"
        "    total = 0\n"
        "    count = 0\n"
        "    invalid = 0\n"
        "    skipped = 0\n"
        "    for r in rows:\n"
        "        if r is None:\n"
        "            invalid += 1\n"
        "            continue\n"
        "        if r.get('skip'):\n"
        "            skipped += 1\n"
        "            continue\n"
        "        total += r['amount']\n"
        "        count += 1\n"
        "    average = total / count if count else 0\n"
        "    return total, count, invalid, skipped, average\n"
    )
    open(f"{d}/src/core/orig.py", "w").write(_body)
    _git("add", "-A"); _git("commit", "-qm", "base")
    _base = _git("rev-parse", "HEAD").stdout.strip()
    open(f"{d}/src/core/copy_new.py", "w").write(_body)   # NEW file, byte-copy of UNCHANGED orig
    _git("add", "-A"); _git("commit", "-qm", "dup")
    _res = subprocess.run(
        [sys.executable, hg.__file__], cwd=d,
        env={**os.environ, "HARD_GATE_BASE": _base, "HARD_GATE_ENFORCE": "1"},
        capture_output=True, text=True,
    )
    check("REAL-TOOL golden: new file copying an UNCHANGED file -> RED (exit 1, R0801)",
          _res.returncode == 1 and "R0801" in _res.stdout)
else:
    print("  SKIP real-tool golden (vulture/pylint not importable in this env; CI installs them)")

print("\nOBSERVED-RED demonstration (the filter genuinely discriminates):")
print(f"  same violation on changed line 11 -> caught={on_changed in kept}")
print(f"  same violation on unchanged line 400 -> caught={on_unchanged in kept}  (must be False)")

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nALL PASS -- diff filter + hunk parser correct")
sys.exit(0)
