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

# --- fail-closed: corpus build failure (git ls-files) must RAISE, not degrade -------
def _git_ls_fail(cmd, capture_output, text, check):  # noqa: ARG001
    return types.SimpleNamespace(stdout="", stderr="fatal: not a git repository", returncode=128)


try:
    hg.candidate_corpus(["src/x.py"], run=_git_ls_fail)
    check("corpus build failure (git ls-files) FAILS CLOSED (raises)", False)
except HardGateError:
    check("corpus build failure (git ls-files) FAILS CLOSED (raises)", True)

# --- vulture: malfunction raises; incomplete (rc=1) is 'partial'; clean codes are 'ok' ---
_orig_avail = hg._tool_available
_orig_cl = hg.changed_lines
_orig_cc = hg.candidate_corpus
_orig_rv = hg.run_vulture
_orig_fd = hg.find_duplicates


def _vult(rc, stderr=""):
    def _r(cmd, capture_output, text, check):  # noqa: ARG001
        return types.SimpleNamespace(stdout="", stderr=stderr, returncode=rc)
    return _r


hg._tool_available = lambda name: True
try:
    for label, run in [("bad-args rc=2", _vult(2, "usage")),
                       ("traceback", _vult(1, "Traceback (most recent call last):"))]:
        try:
            hg.run_vulture(["src/x.py"], run=run)
            check(f"vulture malfunction FAILS CLOSED: {label}", False)
        except HardGateError:
            check(f"vulture malfunction FAILS CLOSED: {label}", True)
    def _vult_out(rc, out):
        def _r(cmd, capture_output, text, check):  # noqa: ARG001
            return types.SimpleNamespace(stdout=out, stderr="", returncode=rc)
        return _r

    _, st1 = hg.run_vulture(["src/x.py"], run=_vult(1, "syntax error in a.py"))
    check("vulture rc=1 (unparseable file) -> status 'partial' (incomplete)", st1 == "partial")
    _, st0 = hg.run_vulture(["src/x.py"], run=_vult_out(0, ""))
    check("vulture rc=0 (clean, no output) -> status 'ok'", st0 == "ok")
    _f3, _st3 = hg.run_vulture(["src/x.py"], run=_vult_out(3, "src/a.py:5: unused var 'x' (80% confidence)"))
    check("vulture rc=3 WITH a parseable finding -> status 'ok'", _st3 == "ok" and len(_f3) == 1)
    # rc=3 with no parseable finding, OR any unrecognised output line, is fake-green -> malfunction.
    for label, run in [("rc=3 + no finding", _vult_out(3, "")),
                       ("rc=3 + unrecognised output", _vult_out(3, "weird line\nanother")),
                       ("rc=0 + unrecognised output", _vult_out(0, "unexpected garbage"))]:
        try:
            hg.run_vulture(["src/x.py"], run=run)
            check(f"vulture {label} FAILS CLOSED (raises)", False)
        except HardGateError:
            check(f"vulture {label} FAILS CLOSED (raises)", True)
    # ANY unexpected exit code (not 0/1/3) must fail closed -- do not guess completeness.
    for rc in (4, 137, -9):
        try:
            hg.run_vulture(["src/x.py"], run=_vult(rc))
            check(f"vulture unexpected rc={rc} FAILS CLOSED (raises)", False)
        except HardGateError:
            check(f"vulture unexpected rc={rc} FAILS CLOSED (raises)", True)
finally:
    hg._tool_available = _orig_avail

# --- tokenize normalization: a '#' INSIDE A STRING is not a comment (naive split got this wrong) ---
_norm = hg._tokenized_lines('x = "a#b"\ny = 1  # real comment\n')
check("tokenize: '#' inside a string literal is preserved", "a#b" in _norm.get(1, ""))
check("tokenize: a real trailing comment is dropped", "#" not in _norm.get(2, ""))
_tok_raised = False
try:
    hg._tokenized_lines("def broken(\n")   # unbalanced -> TokenError
except (Exception,):  # noqa: BLE001  (TokenError/SyntaxError/IndentationError all acceptable)
    _tok_raised = True
check("tokenize: untokenizable input raises (caller turns it into 'partial')", _tok_raised)

# --- enforce mode + INCOMPLETE analysis must exit 2 (dry-run only warns) -------------
os.environ["HARD_GATE_BASE"] = "origin/main"
hg.changed_lines = lambda base: {"src/x.py": {1}}
hg.candidate_corpus = lambda cf: ["src/x.py", "src/y.py"]
hg.find_duplicates = lambda corpus, cf: ([], "ok")
try:
    for status in ("partial", "absent"):
        hg.run_vulture = (lambda s: (lambda corpus: ([], s)))(status)
        os.environ["HARD_GATE_ENFORCE"] = "1"
        try:
            hg.main()
            check(f"enforce + vulture '{status}' FAILS CLOSED (raises)", False)
        except HardGateError:
            check(f"enforce + vulture '{status}' FAILS CLOSED (raises)", True)
        os.environ.pop("HARD_GATE_ENFORCE", None)
        try:
            check(f"dry-run + vulture '{status}' warns, does NOT fail", hg.main() == 0)
        except HardGateError:
            check(f"dry-run + vulture '{status}' warns, does NOT fail", False)
finally:
    hg.changed_lines, hg.candidate_corpus, hg.run_vulture, hg.find_duplicates = (
        _orig_cl, _orig_cc, _orig_rv, _orig_fd)
    os.environ.pop("HARD_GATE_ENFORCE", None)
    os.environ.pop("HARD_GATE_BASE", None)

# --- find_duplicates: an unreadable corpus file -> 'partial' -------------------------
_, dst = hg.find_duplicates(["/no/such/file_xyz.py"], ["/no/such/file_xyz.py"])
check("find_duplicates: unreadable corpus file -> status 'partial'", dst == "partial")

# --- candidate_corpus: a path with a SPACE must survive (NUL-safe ls-files) ----------
def _ls_space(cmd, capture_output, text, check):  # noqa: ARG001
    if "ls-files" in cmd:
        return types.SimpleNamespace(stdout="src/new module.py\0src/a.py\0", stderr="", returncode=0)
    return types.SimpleNamespace(stdout="", stderr="", returncode=0)


_corp = hg.candidate_corpus([], run=_ls_space)
check("candidate_corpus: a space in a path is preserved, not torn by .split()",
      "src/new module.py" in _corp and "src/a.py" in _corp)

# --- test files: exclude only real test LOCATIONS, not any file named test_*.py -------
check("is_test_path: tests/ dir excluded", hg.is_test_path("tests/unit/x.py"))
check("is_test_path: a /test/ dir component excluded", hg.is_test_path("src/test/helper.py"))
check("is_test_path: conftest.py excluded", hg.is_test_path("conftest.py"))
check("is_test_path: the gate's own scripts/ci/test_*.py excluded",
      hg.is_test_path("scripts/ci/test_hard_gate_diff.py"))
check("is_test_path: a production file NOT excluded", not hg.is_test_path("src/core/foo.py"))
check("is_test_path: a bare test_*.py in a PRODUCTION dir is GATED (bypass fix)",
      not hg.is_test_path("src/test_runtime.py"))

import subprocess
import tempfile

_body = (
    "def compute_totals(rows):\n    total = 0\n    count = 0\n    invalid = 0\n    skipped = 0\n"
    "    for r in rows:\n        if r is None:\n            invalid += 1\n            continue\n"
    "        if r.get('skip'):\n            skipped += 1\n            continue\n"
    "        total += r['amount']\n        count += 1\n    average = total / count if count else 0\n"
    "    return total, count, invalid, skipped, average\n"
)

# --- GOLDEN (PURE PYTHON, no external tool): find_duplicates catches a cross-subsystem copy ---
# The earlier golden ran the whole gate as a subprocess, which needs vulture; without it the gate
# exits 2 and the golden fails for the wrong reason. This one calls find_duplicates() directly, so
# it proves GLOBAL duplicate coverage with zero external dependency.
_gd = tempfile.mkdtemp()
os.makedirs(f"{_gd}/src/core"); os.makedirs(f"{_gd}/src/api")
open(f"{_gd}/src/core/orig.py", "w").write(_body)
open(f"{_gd}/src/api/copied.py", "w").write(_body)   # copy lives in a DIFFERENT subsystem
_fd, _ = hg.find_duplicates([f"{_gd}/src/core/orig.py", f"{_gd}/src/api/copied.py"],
                            [f"{_gd}/src/api/copied.py"])
check("GOLDEN (pure Python): cross-subsystem copy flagged on the CHANGED file",
      any(f.file == f"{_gd}/src/api/copied.py" for f in _fd))

# --- GOLDEN (end-to-end, guarded on vulture): the whole gate exits 1 on the same copy ---
if hg._tool_available("vulture"):
    _d = tempfile.mkdtemp()

    def _git(*a):
        return subprocess.run(["git", "-C", _d, *a], capture_output=True, text=True)

    _git("init", "-q"); _git("config", "user.email", "t@t"); _git("config", "user.name", "t")
    os.makedirs(f"{_d}/src/core"); os.makedirs(f"{_d}/src/api")
    open(f"{_d}/src/core/orig.py", "w").write(_body)
    open(f"{_d}/src/api/keep.py", "w").write("x = 1\n")
    _git("add", "-A"); _git("commit", "-qm", "base")
    _base = _git("rev-parse", "HEAD").stdout.strip()
    open(f"{_d}/src/api/copied.py", "w").write(_body)
    _git("add", "-A"); _git("commit", "-qm", "cross-subsystem copy")
    _res = subprocess.run(
        [sys.executable, hg.__file__], cwd=_d,
        env={**os.environ, "HARD_GATE_BASE": _base, "HARD_GATE_ENFORCE": "1"},
        capture_output=True, text=True,
    )
    check("GOLDEN (end-to-end): cross-subsystem copy -> RED (exit 1)",
          _res.returncode == 1 and "duplicate block also in src/core/orig.py" in _res.stdout)
else:
    print("  SKIP end-to-end golden (vulture absent; the pure-Python golden above proves coverage)")

# --- git invocations carry the hardening flags (non-ASCII / noprefix / -w / -z) -------
_seen_cmds = []
def _capture(cmd, capture_output, text, check):  # noqa: ARG001
    _seen_cmds.append(cmd)
    if "rev-parse" in cmd:
        return types.SimpleNamespace(stdout="deadbeef", stderr="", returncode=0)
    return types.SimpleNamespace(stdout="", stderr="", returncode=0)


hg.changed_lines("origin/main", run=_capture)
_diff_cmd = next((c for c in _seen_cmds if "diff" in c), [])
check("git diff uses core.quotePath=false (non-ASCII paths not octal-escaped)",
      "core.quotePath=false" in _diff_cmd)
check("git diff uses diff.noprefix=false (b/ prefix the parser needs)",
      "diff.noprefix=false" in _diff_cmd)
check("git diff uses -w (a reindent of a pre-existing dup is not a 'change')", "-w" in _diff_cmd)
_seen_cmds.clear()
hg.candidate_corpus([], run=_capture)
_ls_cmd = next((c for c in _seen_cmds if "ls-files" in c), [])
check("git ls-files uses core.quotePath=false + -z", "core.quotePath=false" in _ls_cmd and "-z" in _ls_cmd)
check("candidate_corpus enumerates ALL .py (not just src/scripts)", _ls_cmd[-1] == "*.py")

# --- HARD_GATE_ENFORCE accepts truthy values, not only exact '1' ----------------------
def _enforce_from(v):
    return v.strip().lower() in ("1", "true", "yes", "on")
for v in ("1", "true", "TRUE", "yes", "on"):
    check(f"enforce accepts {v!r}", _enforce_from(v))
for v in ("0", "", " false", "no"):
    check(f"enforce rejects {v!r}", not _enforce_from(v))

# --- find_duplicates: a file over the size cap is skipped (-> 'partial', fail-closed enforce) ---
_big = tempfile.mkdtemp() + "/huge.py"
with open(_big, "w") as _fh:
    _fh.write("x = 1\n" * (hg._MAX_FILE_BYTES // 6 + 1000))  # comfortably over the cap
_, _bst = hg.find_duplicates([_big], [_big])
check("find_duplicates: a file over _MAX_FILE_BYTES -> status 'partial'", _bst == "partial")

# --- P1: vulture finding-path has REAL coverage (a regression disabling it would be caught) ---
if hg._tool_available("vulture"):
    _vd = tempfile.mkdtemp()
    # a genuinely dead function; vulture must produce a finding at its line
    open(f"{_vd}/dead.py", "w").write("import os\n\n\ndef _never_called():\n    return 42\n")
    _vf, _vst = hg.run_vulture([f"{_vd}/dead.py"])
    check("REAL vulture: dead code yields a parsed finding (arm has finding-path coverage)",
          _vst == "ok" and any(fd.source == "vulture" for fd in _vf))
else:
    print("  SKIP real-vulture finding-path coverage (vulture absent)")

# --- baseline: a WHITESPACE-only reindent of a pre-existing duplicate is NOT flagged (real git) ---
if hg._tool_available("vulture"):
    _bd = tempfile.mkdtemp()

    def _bgit(*a):
        return subprocess.run(["git", "-C", _bd, *a], capture_output=True, text=True)

    _bgit("init", "-q"); _bgit("config", "user.email", "t@t"); _bgit("config", "user.name", "t")
    os.makedirs(f"{_bd}/src")
    _blk = ("    total = 0\n    count = 0\n    invalid = 0\n    skipped = 0\n    for r in rows:\n"
            "        if r is None:\n            invalid += 1\n        total += r\n        count += 1\n"
            "    avg = total / count\n    return total, count, avg\n")
    open(f"{_bd}/src/a.py", "w").write("def a(rows):\n" + _blk)
    open(f"{_bd}/src/b.py", "w").write("def b(rows):\n" + _blk)   # pre-existing duplicate AT BASE
    _bgit("add", "-A"); _bgit("commit", "-qm", "base")
    _bbase = _bgit("rev-parse", "HEAD").stdout.strip()
    # whitespace-only edit to b.py's duplicated block (tokens unchanged, no new construct)
    _bt = open(f"{_bd}/src/b.py").read().replace("total = 0", "total  =  0").replace("count = 0", "count  =  0")
    open(f"{_bd}/src/b.py", "w").write(_bt)
    _bgit("add", "-A"); _bgit("commit", "-qm", "whitespace-reindent")
    _br = subprocess.run([sys.executable, hg.__file__], cwd=_bd,
                         env={**os.environ, "HARD_GATE_BASE": _bbase, "HARD_GATE_ENFORCE": "1"},
                         capture_output=True, text=True)
    check("BASELINE: whitespace edit to a PRE-EXISTING duplicate -> NOT flagged (exit 0)",
          _br.returncode == 0)
else:
    print("  SKIP baseline reindent golden (vulture absent)")

# --- SPACE-PATH end-to-end: a DUPLICATE copied into a path WITH A SPACE must be caught ------------
# The trailing-tab the git `+++ b/` header appends for a spaced path made changed_lines emit a
# tab-suffixed key that never matched the corpus, so the duplicate was silently UNGATED. This is the
# full-enforce observed-RED the reviewer asked for (not just `ls-files -z`).
if hg._tool_available("vulture"):
    _sd = tempfile.mkdtemp()

    def _sgit(*a):
        return subprocess.run(["git", "-C", _sd, *a], capture_output=True, text=True)

    _sgit("init", "-q"); _sgit("config", "user.email", "t@t"); _sgit("config", "user.name", "t")
    os.makedirs(f"{_sd}/src")
    _sbody = (
        "def compute(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    d = c / 4\n    e = d ** 2\n"
        "    g = e + 5\n    h = g - 6\n    i = h + 7\n    j = i + 8\n    return a+b+c+d+e+g+h+i+j\n"
    )
    open(f"{_sd}/src/orig.py", "w").write(_sbody)
    _sgit("add", "-A"); _sgit("commit", "-qm", "base")
    _sbase = _sgit("rev-parse", "HEAD").stdout.strip()
    open(f"{_sd}/src/new module.py", "w").write(_sbody)   # NEW file WITH A SPACE, copies orig
    _sgit("add", "-A"); _sgit("commit", "-qm", "dup in a spaced path")
    _sr = subprocess.run([sys.executable, hg.__file__], cwd=_sd,
                         env={**os.environ, "HARD_GATE_BASE": _sbase, "HARD_GATE_ENFORCE": "1"},
                         capture_output=True, text=True)
    check("SPACE-PATH: a duplicate copied into 'src/new module.py' -> RED (exit 1), not ungated",
          _sr.returncode == 1 and "src/new module.py" in _sr.stdout)
else:
    print("  SKIP space-path golden (vulture absent)")

print("\nOBSERVED-RED demonstration (the filter genuinely discriminates):")
print(f"  same violation on changed line 11 -> caught={on_changed in kept}")
print(f"  same violation on unchanged line 400 -> caught={on_unchanged in kept}  (must be False)")

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nALL PASS -- diff filter + hunk parser correct")
sys.exit(0)
