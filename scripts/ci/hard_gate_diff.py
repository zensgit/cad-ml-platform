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

import hashlib
import io
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


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


def candidate_corpus(changed_files: list[str], run=subprocess.run,
                     roots: tuple[str, ...] = ("src", "scripts")) -> list[str]:
    """The FULL production tree (all tracked non-test .py under `roots`) + the changed files.

    No subsystem bound: a new file copying an EXISTING file must be caught wherever the source
    lives (src/core/a copied into src/api/b included), so there is NO cross-scope coverage gap.
    A full-tree scan is affordable because duplicate detection is a fingerprint index
    (`find_duplicates`, O(total_lines)), not pylint's superlinear all-pairs.

    FAIL-CLOSED: if `git ls-files` cannot enumerate the tree, raise -- a partial corpus would
    silently shrink coverage (a duplicate whose source was dropped would pass)."""
    proc = run(["git", "ls-files", "--", *roots], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise HardGateError(
            f"`git ls-files` failed (rc={proc.returncode}): {proc.stderr.strip()[:200]} -- "
            "cannot build the corpus; refusing to run against a partial tree."
        )
    corpus = {p for p in proc.stdout.split() if p.endswith(".py") and not is_test_path(p)}
    corpus.update(f for f in changed_files if f.endswith(".py") and not is_test_path(f))
    return sorted(corpus)


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


def run_vulture(corpus: list[str], run=subprocess.run) -> tuple[list[Finding], str]:
    """Returns (findings, status): status ∈ {"ok","absent","partial"}.
      "absent"  = vulture not installed.
      "partial" = ran, but some input could not be parsed (rc=1) -- INCOMPLETE analysis.
    A hard malfunction (bad args rc=2, or a traceback) raises. main() turns absent/partial into
    a warning (dry-run) or a fail-closed exit 2 (enforce): a required gate must not pass what it
    could not fully analyse."""
    if not _tool_available("vulture"):
        return [], "absent"
    if not corpus:
        return [], "ok"
    proc = run([sys.executable, "-m", "vulture", *corpus, "--min-confidence", "80"],
               capture_output=True, text=True, check=False)
    # vulture exit codes (allow-list, fail-closed on anything else):
    #   0 = clean, 3 = dead code found  -> complete run ('ok')
    #   1 = some file could not be parsed -> INCOMPLETE ('partial'; fail-closed in enforce)
    #   ANYTHING ELSE (2 bad-args, a traceback, a signal-kill like 137/OOM, an unknown code)
    #     -> malfunction: a required gate must not guess. RAISE.
    if "Traceback (most recent call last)" in proc.stderr:
        raise HardGateError(f"vulture crashed: {proc.stderr.strip()[:300]}")
    if proc.returncode in (0, 3):
        status = "ok"
    elif proc.returncode == 1:
        status = "partial"
        if proc.stderr.strip():
            print(f"  ::warning:: vulture partial: {proc.stderr.strip().splitlines()[0][:160]}")
    else:
        raise HardGateError(
            f"vulture returned an unexpected exit code {proc.returncode} -- refusing to guess "
            f"whether the run was complete: {(proc.stderr or proc.stdout).strip()[:300]}"
        )
    findings = []
    pat = re.compile(r"^(.+?):(\d+): (.+)$")
    for line in proc.stdout.splitlines():
        m = pat.match(line)
        if m:
            findings.append(Finding(m.group(1), int(m.group(2)), m.group(3), "vulture"))
    return findings, status


# Duplicate detection: a TOKENIZED-line FINGERPRINT INDEX over the whole corpus, NOT pylint.
# pylint duplicate-code is superlinear -> the full tree times out (>150s), forcing a subsystem
# bound that left cross-scope duplicates ungated. The index is O(total_lines) (~0.6s on this
# repo), so it covers the ENTIRE tree with no scope gap, and is deterministic + tool-version-
# independent. Normalization is by `tokenize`, not naive text: a comment character inside a
# string ("a#b") must NOT be treated as a comment, which a `split("#")` heuristic got wrong.
DUP_MIN_LINES = 10

_SKIP_TOK = frozenset({
    tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
    tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
})


def _tokenized_lines(text: str) -> dict[int, str]:
    """physical-line-number -> the significant tokens on that line, joined with NUL.

    Uses the real tokenizer, so comments are dropped correctly (including a `#` inside a string
    literal, which naive splitting mishandled) and whitespace is irrelevant. Identifier names
    are KEPT (a verbatim/whitespace-different copy is the threat; pylint's R0801 does not rename
    either). Raises tokenize.TokenError / IndentationError / SyntaxError on untokenizable input;
    the caller treats that as an unparseable file (incomplete analysis)."""
    per_line: dict[int, list[str]] = {}
    reader = io.StringIO(text).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type in _SKIP_TOK:
            continue
        s = tok.string.strip()
        if s:
            per_line.setdefault(tok.start[0], []).append(s)
    return {ln: "\x00".join(toks) for ln, toks in per_line.items()}


def find_duplicates(corpus: list[str], changed_files: list[str],
                    min_lines: int = DUP_MIN_LINES) -> tuple[list[Finding], str]:
    """Emit a finding for every changed-file window of `min_lines` consecutive significant
    (tokenized) lines whose fingerprint ALSO appears in a DIFFERENT file (a cross-file copy).
    Findings carry the changed file's real line numbers; main() filters to changed lines."""
    fp_files: dict[str, set[str]] = {}
    file_windows: dict[str, list[tuple[str, list[int]]]] = {}
    unparseable: list[str] = []
    for f in corpus:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="strict")
            norm = _tokenized_lines(text)
        except (OSError, UnicodeDecodeError, tokenize.TokenError, SyntaxError, IndentationError):
            unparseable.append(f)
            continue
        nontrivial = sorted(norm.items())  # (line_no, normalized) for lines with tokens
        wins: list[tuple[str, list[int]]] = []
        for j in range(len(nontrivial) - min_lines + 1):
            window = nontrivial[j:j + min_lines]
            fp = hashlib.sha256("\n".join(n for _, n in window).encode()).hexdigest()
            fp_files.setdefault(fp, set()).add(f)
            wins.append((fp, [ln for ln, _ in window]))
        file_windows[f] = wins

    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()  # one finding per (file, line); overlapping windows dedupe
    for f in changed_files:
        for fp, orig_lines in file_windows.get(f, []):
            others = fp_files.get(fp, set()) - {f}
            if others:
                other = sorted(others)[0]
                for ln in orig_lines:
                    if (f, ln) not in seen:
                        seen.add((f, ln))
                        findings.append(Finding(f, ln, f"duplicate block also in {other}",
                                                "duplicate-code"))

    status = "ok"
    if unparseable:
        # A corpus file we could not tokenize is analysed INCOMPLETELY -> 'partial'
        # (fail-closed in enforce). If one of the CHANGED files is unparseable, that is the PR's
        # own syntax error and other CI catches it; here it just means we can't gate that file.
        status = "partial"
        print(f"  ::warning:: duplicate-index: {len(unparseable)} corpus file(s) unparseable "
              f"(e.g. {unparseable[0]})")
    return findings, status


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
    corpus = candidate_corpus(changed_files)  # raises (fail-closed) if git can't enumerate
    vult, vstatus = run_vulture(corpus)
    dup, dstatus = find_duplicates(corpus, changed_files)
    new = new_violations(vult + dup, changed)

    print(f"hard-gate: mode={'ENFORCE' if enforce else 'dry-run'} base={base}")
    print(f"  changed .py files: {len(changed_files)}  |  corpus scanned: {len(corpus)} .py files")

    # Any producer that is absent or could only PARTIALLY analyse the corpus means the gate
    # could not fully verify the change. That is fail-closed in ENFORCE (a required gate must
    # not report a pass it could not stand behind); a warning in dry-run.
    incomplete = [(n, s) for n, s in (("vulture", vstatus), ("duplicate-index", dstatus))
                  if s != "ok"]
    for name, status in incomplete:
        print(f"  ::warning:: {name} analysis {status} -- corpus not fully verified")
    if enforce and incomplete:
        raise HardGateError(
            "incomplete analysis in enforce mode ("
            + ", ".join(f"{n}={s}" for n, s in incomplete)
            + "). A required gate must not pass what it could not fully analyse."
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
