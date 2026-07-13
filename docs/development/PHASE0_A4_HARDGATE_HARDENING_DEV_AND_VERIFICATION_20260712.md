# Phase 0 · A4 — Hard-gate hardening (Dev & Verification)

**Date**: 2026-07-12 · **PR**: #505 (branch `phase0-a4-hardgate-mechanics-20260708`)
**Rigor**: L2 (CI mechanics — not L3 auth/model) · **Grounded on**: `origin/main@8ff94175`
**Model tier**: spec-complete mechanical CI work → **Sonnet-5 tier** per `PRODUCT_STRATEGY.md`
policy; executed on **Opus 4.8** (Fable 5 at daily cap). Stated explicitly per the tiering rule.

## Why this slice now

The diff-scoped hard gate was sound in DRY-RUN but **could not safely be made a required
check** — three defects would make an *armed* gate either wedge PRs or silently pass. These are
the exact three the review named. This is **not** L3 and **not** on any routine-contested branch,
so it is completable independent of the two owner P0s (stop routine; independent reviewer).

## Desired → Achieved

| # | Defect (before) | Desired | Achieved | Verified by |
|---|---|---|---|---|
| 1 | `hard-gate.yml` had `paths:` filter → as a *required* check, PRs touching no matched file never trigger the job, so the required context is never reported and the PR **waits forever**. | Job runs on **every** PR to main; path-scoping happens *inside* the gate (it only inspects changed `.py` lines), so an irrelevant PR finds nothing and passes. | `paths:` removed; gate returns `{}`→"nothing to check"→pass for a no-`.py`-change PR. | self-test: *"legit empty: no .py change → {} (non-source PR passes, doesn't wait forever)"* |
| 2 | `git fetch origin <base> --depth=1 \|\| true` → a base-fetch failure was **swallowed**, and `--depth=1` truncated the history the merge-base needs. | Base-fetch failure **fails the step**; full history for `base...HEAD`. | `\|\| true` and `--depth=1` removed; fetch is `+refs/heads/<base>:refs/remotes/origin/<base>` (checkout already did `fetch-depth: 0`). | workflow diff |
| 3 | `changed_lines()` ran `git diff` with `check=False` and **ignored the return code**. An unresolvable base printed nothing → read as an **empty diff** → "zero changed lines" → gate **passes everything** (fail-open). | An unresolvable base / failed diff **fails closed** (raises), distinct from a legitimately empty diff. | New `rev-parse --verify` probe raises `HardGateError` on an unresolvable base; `git diff` return code checked and raises on failure. A malfunction exits **2** in *both* dry-run and enforce. | self-test RED1/RED2 + end-to-end `exit=2` |
| 3b | (related) enforce mode + a missing finding-producer → only a warning → could **pass without actually checking**. | In enforce mode, a missing producer is a malfunction → fail closed. | `main()` raises `HardGateError` when `enforce and missing`. | self-test *"enforce mode + missing finding-producer FAILS CLOSED"* |

## Key design point: malfunction ≠ finding

A **finding** (dead code on a changed line) fails only in *enforce* mode (dry-run reports and
exits 0). A **malfunction** (unknown base, failed diff, missing tool in enforce) fails **closed in
both modes** and exits `2` (distinct from a finding's `1`). Rationale: the entire purpose of this
gate is that *a broken gate must never be mistaken for a green one*. This is the same lesson as the
L3 eval-integrity gate (#509): a gate that can't do its job must not report success.

## Verification (executed, not read — with a positive control)

Self-test `scripts/ci/test_hard_gate_diff.py` (plain `python3`, runs on 3.9+; **no `|| true`** in
CI so a broken filter reds):

```
PASS  catches a violation on a CHANGED line (gate can fail)          # gate CAN fail (positive control)
PASS  IGNORES pre-existing debt on an UNCHANGED line (no 误伤)
PASS  hunk parser: foo.py added lines == {11,12,13}
PASS  unresolvable base FAILS CLOSED (raises, not {} = pass-everything)   # defect 3
PASS  git diff failure FAILS CLOSED (raises)                              # defect 3
PASS  enforce mode + missing finding-producer FAILS CLOSED (raises)       # defect 3b
ALL PASS
```

End-to-end against a real git repo: `HARD_GATE_BASE=origin/nonexistent python3 hard_gate_diff.py`
→ **`exit=2`**, stderr `::error::hard-gate MALFUNCTION (fail-closed)`. Normal path (resolvable base,
one changed line) correctly isolates the changed line; a no-`.py`-change PR yields `{}` and passes.

## Round 2 — two more load-bearing defects (second review)

The first round fixed the wedge + base fail-open. A second review found two **false-negatives**
that would let an *armed* gate pass code it should block. Both reproduced with **real tools**.

### Desired → Achieved (round 2)

| # | Defect (before) | Achieved | Verified by (executed) |
|---|---|---|---|
| 4 | **Duplicate false-negative.** Producers were handed only the *changed* files. A new `b.py` copying an **unchanged** `a.py` → pylint never sees `a.py` → no R0801. This is the main "agent re-injects duplicate code" case. | Producers scan a **corpus** (changed files + their subsystem), then findings are filtered to changed lines. R0801 is parsed from **all** involved locations (JSON `==module:[range]`), not just pylint's order-dependent anchor — so a duplicate anchored on the *unchanged* file still reaches the changed one. | real-tool golden: new file copying an UNCHANGED file → **exit 1, R0801**; unit: "multi-location parse emits BOTH files"; "changed-line filter keeps the dup despite anchor on the unchanged". Repro of the *old* miss: only-`b.py`→0 findings, `a.py`+`b.py`→1. |
| 5 | **Producer execution failure fails open.** `run_vulture`/`run_duplicate` ignored the subprocess return code + stderr, always returning `ok=True`. A tool that crashed / hit bad args / emitted unparseable output → `findings=[]` → **green**. | Exit codes are interpreted: vulture `0/3` expected, `2`/traceback → malfunction; pylint bit-mask — `8`=dup (expected), bit `1`(fatal)/`32`(usage) → malfunction; unparseable JSON → malfunction. A malfunction raises `HardGateError` → **exit 2 in both modes**. Distinct from "tool absent" (warn/enforce-fail). | unit: vulture rc=2 / traceback, pylint rc=32 / rc=1 / bad-JSON all → raise; rc=0 clean NOT misread as malfunction. |

### Supporting changes
- **Corpus is bounded, not the literal full tree.** The whole tree (~1200 `.py`) makes pylint
  duplicate-code **time out (>150 s)** — unaffordable for a required gate. Scope = the changed
  files' **subsystem** (depth-3, e.g. `src/core/vision`), tightened to changed-dirs if it exceeds
  `CORPUS_CAP=250`. Measured 2.6–17.7 s across `src/api`, `src/ml`, `src/core/vision`, `scripts/ci`.
  The scope is **always printed** (never a silent cap).
- **Test files are out of scope** (`is_test_path`): test helpers are legitimately "unused" and tests
  intentionally duplicate — analysing them is noise (the review saw 6 vulture findings in the gate's
  own test file). Now excluded and logged; the gate self-applies **clean** on this PR.
- **Producers invoked via `python -m`** (import-based availability), robust when pip installs the
  module without a console script on PATH.
- **Workflow:** producers **pinned** (`vulture==2.16`, `pylint==3.3.9`) and installed **before** the
  self-test so its real-tool golden runs; the check name is now **`Hard Gate (diff-scoped)`** —
  mode-agnostic, so arming (removing "dry-run") will not orphan a required status context.

## Round 3 — close the cross-scope gap + full fail-closed (third review)

Round 2 bounded the corpus to a subsystem for speed and **logged** the cross-subsystem gap. A
third review was right that for a *required* gate a known coverage gap is a malfunction, not a
warning, and that several producer/corpus failure modes still degraded silently. Both closed by
**replacing pylint with a stdlib fingerprint index** — which removes the very cost that forced the
bound.

### Desired → Achieved (round 3)

| # | Defect (before) | Achieved | Verified by (executed) |
|---|---|---|---|
| 6 | **Duplicate gap was only closed within a subsystem.** The corpus was bounded (pylint timed out on the full tree), so `src/core/a` copied into `src/api/b` could pass. | **Global** duplicate detection via a normalized-line **fingerprint index** over the *whole* production tree (no subsystem bound): a window of `DUP_MIN_LINES=10` consecutive non-trivial normalized lines whose fingerprint also appears in a **different** file is a cross-file duplicate. O(total-lines), **~0.6 s**; full gate **6.4 s** end-to-end over 1167 files (was >150 s → timeout). Deterministic, tool-version-independent. | **golden: cross-SUBSYSTEM copy (`src/api` copies `src/core`) → exit 1** ("duplicate block also in src/core/orig.py"). Runs with no external tool. |
| 7 | **Corpus/producer failures degraded silently.** `git ls-files` return code ignored → a failure shrank the corpus to changed-files-only. vulture `rc=1` (a file couldn't parse) only warned, even in enforce. | `candidate_corpus` **raises** on `git ls-files` failure. Producers return a 3-state status (`ok`/`absent`/`partial`); **any non-`ok` in ENFORCE → `HardGateError` → exit 2** (dry-run warns). A required gate never passes what it could not fully analyse. | unit: git-ls-files failure → raise; vulture rc=1 → `partial`; enforce+`partial`/`absent` → raise; dry-run+same → warn+continue; unreadable corpus file → `partial`. |

### Supporting changes
- **pylint removed entirely** from duplicate detection — with it go the JSON parsing, order-dependent
  anchor, module→path mapping, exit-code bit-mask, and version pinning. Only **vulture** remains an
  external producer (still pinned `==2.16`); the fingerprint index is stdlib (`hashlib`).
- **Fingerprint semantics documented:** normalized = trailing-comment stripped + all whitespace
  removed; blank/comment-only lines skipped. Catches verbatim / whitespace- or comment-different
  copies (the realistic re-injection); a rename-heavy paraphrase is out of scope by design.
- vulture still runs over the **full tree** (5.5 s) so cross-file usage is correct (no false
  positives from changed-files-only).

## What is still NOT done (honest scope)

- **Arming remains owner-only** and is deliberately not performed here: it requires (a) uncommenting
  `HARD_GATE_ENFORCE: "1"`, and (b) adding the check to branch protection. This PR makes the gate
  *safe to arm* — it does not arm it. `merged != enabled != safe to enable` (§7.2).
- **Duplicate detection is line-fingerprint, not AST/token** — it will not catch a copy that renames
  identifiers throughout. Documented trade-off; an AST/token index is a future upgrade if needed.
- These fixes make the gate *requireable*; whether to require it is a branch-protection decision.
