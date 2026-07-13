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

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass


class HardGateError(RuntimeError):
    """A gate MALFUNCTION (not a code finding).

    Raised when the gate cannot do its job — e.g. the diff base does not resolve. A
    malfunction must fail CLOSED regardless of dry-run/enforce, because the whole point of
    this gate is that a broken gate must not silently report a pass.
    """


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
    # FAIL CLOSED on an unresolvable base. If the base ref does not exist (e.g. the CI
    # fetch failed), `git diff` would print nothing and exit non-zero — which the old code
    # read as an EMPTY diff, i.e. "zero changed lines", i.e. a green gate on everything.
    # An unknown base is a malfunction, not "no changes": refuse to run.
    probe = run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise HardGateError(
            f"diff base {base_ref!r} does not resolve to a commit. Refusing to run: an "
            "unresolvable base sees zero changed lines and would pass the gate on everything. "
            "(In CI this usually means the base fetch failed — it must not be swallowed with "
            "`|| true`.)"
        )
    proc = run(
        ["git", "diff", "--unified=0", f"{base_ref}...HEAD", "--", "*.py"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise HardGateError(
            f"`git diff {base_ref}...HEAD` failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or '<no stderr>'}"
        )
    out = proc.stdout
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
# Corpus: producers MUST scan the whole candidate tree, not just changed files.
#
# A new file b.py that copies an UNCHANGED file a.py is only detectable if the duplicate
# producer sees BOTH (pylint compares the files given to it). Likewise vulture over
# changed-files-only would false-POSITIVE on a symbol defined on a changed line but used in
# an unchanged file. So producers run over the full corpus, and findings are filtered to
# changed lines AFTERWARDS -- pre-existing debt on unchanged lines is never reported.
# ---------------------------------------------------------------------------
def is_test_path(path: str) -> bool:
    """Test scaffolding is deliberately OUT of scope: test helpers are legitimately 'unused'
    (fixtures, fakes) and tests intentionally duplicate. Dead-code/duplicate analysis of them
    is noise, and would red a PR for touching its own test file. The gate guards PRODUCTION
    code re-inflation, not test code."""
    base = os.path.basename(path)
    parts = path.split("/")
    return (
        base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"
        or "tests" in parts or "test" in parts
    )


CORPUS_CAP = 250  # keep pylint duplicate-code well under a required-gate time budget


def _scope_prefixes(changed_files: list[str], depth: int) -> set[str]:
    """Path prefixes of the changed files at `depth` components (or the file's dir if shallower).
    e.g. depth=3: src/core/vision/x.py -> 'src/core/vision'; src/ml/x.py -> 'src/ml'."""
    out = set()
    for f in changed_files:
        if not f.endswith(".py") or is_test_path(f):
            continue
        parts = f.split("/")
        out.add("/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1]) or ".")
    return out


def candidate_corpus(changed_files: list[str], run=subprocess.run) -> list[str]:
    """The tree the producers scan, so a new file copying an EXISTING file is detectable.

    Scanning the literal full tree (~1200 .py) makes pylint duplicate-code time out (>150s),
    which a required gate cannot afford. So the corpus is bounded to the changed files' scope,
    tightened until it fits CORPUS_CAP: subsystem (depth-3, e.g. src/core/vision) first, else
    the changed files' immediate directories. This catches the realistic threat -- re-injecting
    a copy of a nearby existing file -- while staying fast; a cross-scope duplicate is rare and,
    being dry-run until armed, visible to the owner first. The scope is always PRINTED (never a
    silent cap) so a dropped area is auditable."""
    changed_py = [f for f in changed_files if f.endswith(".py") and not is_test_path(f)]

    def _corpus_for(prefixes: set[str]) -> list[str]:
        tracked = run(["git", "ls-files", "--", *sorted(prefixes)],
                      capture_output=True, text=True, check=False).stdout.split()
        c = {p for p in tracked if p.endswith(".py") and not is_test_path(p)}
        c.update(changed_py)
        return sorted(c)

    for depth, label in ((3, "subsystem(depth-3)"), (99, "changed-dirs")):
        prefixes = _scope_prefixes(changed_files, depth)
        corpus = _corpus_for(prefixes) if prefixes else changed_py
        if len(corpus) <= CORPUS_CAP:
            print(f"  corpus scope: {label}, {len(corpus)} .py files "
                  f"({', '.join(sorted(prefixes)[:4])}{' …' if len(prefixes) > 4 else ''})")
            return corpus
    # Even the changed files' own dirs exceed the cap: scan only the changed files + their
    # exact directory, and say so loudly -- a duplicate outside these dirs is not checked.
    print(f"  ::warning:: corpus > {CORPUS_CAP} even at changed-dirs scope; "
          "checking changed files' directories only -- cross-directory duplicates NOT gated.")
    return corpus


# ---------------------------------------------------------------------------
# Finding-producers. Two distinct not-OK states, handled differently:
#   * tool ABSENT (not installed)  -> (·, available=False); main() warns (dry-run) or
#     fails closed (enforce). Not a crash.
#   * tool present but MALFUNCTIONS (crash / bad-args / unexpected exit) -> raise
#     HardGateError -> exit 2 in BOTH modes. A broken producer is not a pass.
# The old code did neither: it ignored the return code, so a crashed tool returned
# findings=[] / ok=True and the gate went green.
# ---------------------------------------------------------------------------
def _tool_available(name: str) -> bool:
    # Availability = "the module is importable", checked against the SAME interpreter we invoke
    # it with (`python -m <name>`), not "a console script is on PATH". pip can install the
    # module without putting its script on PATH, and `python -m` runs it either way.
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run_vulture(corpus: list[str], run=subprocess.run) -> tuple[list[Finding], bool]:
    if not _tool_available("vulture"):
        return [], False
    if not corpus:
        return [], True
    proc = run([sys.executable, "-m", "vulture", *corpus, "--min-confidence", "80"],
               capture_output=True, text=True, check=False)
    # vulture exit codes: 0=clean, 3=dead code found (both expected). 2=bad CLI args and any
    # traceback are a real malfunction. 1=some input file could not be parsed -- for a
    # whole-tree scan that means a PRE-EXISTING broken file, which must NOT red every PR:
    # warn and use the findings from the files that did parse.
    if proc.returncode == 2 or "Traceback (most recent call last)" in proc.stderr:
        raise HardGateError(
            f"vulture malfunctioned (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:300]}"
        )
    if proc.returncode == 1 and proc.stderr.strip():
        print(f"  ::warning:: vulture: partial run -- {proc.stderr.strip().splitlines()[0][:160]}")
    findings = []
    pat = re.compile(r"^(.+?):(\d+): (.+)$")
    for line in proc.stdout.splitlines():
        m = pat.match(line)
        if m:
            findings.append(Finding(m.group(1), int(m.group(2)), m.group(3), "vulture"))
    return findings, True


# pylint exit code is a bit-mask: 1=fatal, 2=error, 4=warning, 8=refactor, 16=convention,
# 32=usage error. duplicate-code (R0801) is a refactor (bit 8). A fatal (1) or usage (32)
# is a malfunction; an error (2) means a file could not be processed -> warn, don't red.
_PYLINT_FATAL, _PYLINT_USAGE = 1, 32


def _pylint_module_map(corpus: list[str]) -> dict[str, list[str]]:
    """pylint reports duplicate locations as `==<module>:[a:b]` using module names, not paths.
    Map a module name back to the corpus path(s) whose stem matches. Ambiguous stems (same
    basename in two dirs) map to BOTH -- deliberately conservative (fail toward reporting)."""
    m: dict[str, list[str]] = {}
    for p in corpus:
        stem = os.path.splitext(os.path.basename(p))[0]
        m.setdefault(stem, []).append(p)
    return m


def run_duplicate(corpus: list[str], run=subprocess.run) -> tuple[list[Finding], bool]:
    if not _tool_available("pylint"):
        return [], False
    if not corpus:
        return [], True
    proc = run([sys.executable, "-m", "pylint", *corpus, "--disable=all",
                "--enable=duplicate-code", "--min-similarity-lines=10",
                "--output-format=json"],
               capture_output=True, text=True, check=False)
    if (proc.returncode & _PYLINT_FATAL) or (proc.returncode & _PYLINT_USAGE):
        raise HardGateError(
            f"pylint malfunctioned (rc={proc.returncode}, "
            f"fatal={bool(proc.returncode & _PYLINT_FATAL)} usage={bool(proc.returncode & _PYLINT_USAGE)}): "
            f"{proc.stderr.strip()[:300] or '<see json>'}"
        )
    try:
        messages = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise HardGateError(f"pylint produced unparseable JSON: {exc}") from exc

    modmap = _pylint_module_map(corpus)
    loc_re = re.compile(r"==([\w.]+):\[(\d+):(\d+)\]")
    findings: list[Finding] = []
    for msg in messages:
        if msg.get("symbol") != "duplicate-code":
            continue
        summary = msg.get("message", "").splitlines()[0] if msg.get("message") else "R0801 duplicate"
        seen: set[tuple[str, int]] = set()
        # Every involved location must be emitted -- pylint's anchor is order-dependent, so
        # relying on it alone would silently drop a duplicate anchored on the UNCHANGED file.
        for mod, lo, hi in loc_re.findall(msg.get("message", "")):
            stem = mod.split(".")[-1]
            for path in modmap.get(stem, []):
                for ln in range(int(lo), int(hi) + 1):
                    if (path, ln) not in seen:
                        seen.add((path, ln))
                        findings.append(Finding(path, ln, f"R0801: {summary}", "duplicate-code"))
        # Fallback: if the message body had no parseable ranges, keep the JSON anchor.
        if not any(loc_re.finditer(msg.get("message", ""))) and msg.get("path"):
            findings.append(Finding(msg["path"], int(msg.get("line", 1)),
                                    f"R0801: {summary}", "duplicate-code"))
    return findings, True


def main() -> int:
    base = os.environ.get("HARD_GATE_BASE", "origin/main")
    enforce = os.environ.get("HARD_GATE_ENFORCE") == "1"

    changed_all = changed_lines(base)
    # Test files are out of scope (see is_test_path). Drop them from the changed set and say so.
    dropped = sorted(f for f in changed_all if is_test_path(f))
    changed = {f: v for f, v in changed_all.items() if not is_test_path(f)}
    if dropped:
        print(f"hard-gate: {len(dropped)} test file(s) out of scope (not gated): "
              f"{', '.join(dropped[:5])}{' …' if len(dropped) > 5 else ''}")
    changed_files = sorted(changed)
    if not changed_files:
        print("hard-gate: no changed non-test .py lines vs", base, "-> nothing to check")
        return 0

    # Producers scan the FULL corpus so cross-file duplicates/usage are visible; findings are
    # filtered to changed lines afterwards. (Scanning changed-files-only misses a new file that
    # copies an unchanged one -- the main "agent re-injects duplicate code" case.)
    corpus = candidate_corpus(changed_files)
    vult, vult_ok = run_vulture(corpus)
    dup, dup_ok = run_duplicate(corpus)
    new = new_violations(vult + dup, changed)

    print(f"hard-gate: mode={'ENFORCE' if enforce else 'dry-run'} base={base}")
    print(f"  changed .py files: {len(changed_files)}  |  corpus scanned: {len(corpus)} .py files")
    missing = [tool for tool, ok in [("vulture", vult_ok), ("duplicate-code", dup_ok)] if not ok]
    for tool in missing:
        print(f"  ::warning:: {tool} unavailable -- not run (gate does not fabricate a pass/fail)")
    # In ENFORCE mode a missing finding-producer means the gate CANNOT verify the changed
    # lines. That is a malfunction, not a pass: fail closed. (In dry-run it's only a warning.)
    if enforce and missing:
        raise HardGateError(
            f"finding-producer(s) unavailable in enforce mode: {', '.join(missing)}. "
            "The gate cannot certify changed lines without them; refusing to report a pass."
        )
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
    try:
        sys.exit(main())
    except HardGateError as exc:
        # A malfunction fails CLOSED in BOTH modes (dry-run and enforce) — a broken gate
        # must never be mistaken for a green one. Exit 2 to distinguish it from a finding (1).
        print(f"::error::hard-gate MALFUNCTION (fail-closed): {exc}", file=sys.stderr)
        sys.exit(2)
