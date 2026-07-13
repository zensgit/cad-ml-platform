# Phase 0 · A4 — Hard-gate hardening (Dev & Verification)

**Date**: 2026-07-12 · **PR**: #505 (branch `phase0-a4-hardgate-mechanics-20260708`)
**Rigor**: L2 (CI mechanics — not L3 auth/model) · **Grounded on**: `origin/main@8ff94175`
**Model tier**: mostly spec-complete mechanical CI work (**Sonnet-5 tier**); the round-3/5
adversarial reviews and design calls are **Fable-5 tier**. Executed on Opus 4.8 / Fable 5.

## ⭐ CURRENT STATE (round 5 — authoritative; the round-by-round history below is provenance)

The gate as it stands on this branch:
- **Duplicate detection = a stdlib normalized-**token** fingerprint index over ALL tracked non-test
  `.py`** (via `tokenize`; global, O(lines), ~10s). **pylint is gone** (it timed out on the full
  tree; its output was fragile). The rounds below that mention pylint / a subsystem-bounded corpus /
  `paths:` filters describe **superseded** intermediate states.
- **Dead code = vulture** (the only external producer, pinned `==2.16`).
- **Fully fail-closed:** unresolvable base / `git ls-files` / incomplete analysis in enforce / any
  unexpected vulture exit code / rc=3-with-no-parsed-finding / any unrecognised vulture line → **exit
  2** (malfunction, both modes). A finding is exit 1 (enforce only).
- **Diff-scoped and pre-existing-debt-safe:** `git diff --unified=0 -w` — a *reindent* of a
  pre-existing duplicate is not treated as a change.
- **git-config-robust:** `-c core.quotePath=false -c diff.noprefix=false` + `ls-files -z` — non-ASCII
  paths, spaced paths, and `diff.noprefix=true` no longer bypass or wedge the gate.
- **Scoped exclusion:** only real test *locations* (a `tests`/`test` dir, `conftest.py`,
  `scripts/ci/test_*.py`) are out of scope — a bare `src/test_runtime.py` is **gated**.
- **The required CI context is `Hard Gate (diff-scoped)`** (mode-agnostic; arming won't orphan it).
- **Dry-run; owner-armed.** `HARD_GATE_ENFORCE ∈ {1,true,yes,on}` + add the context to branch
  protection.
- **Known limitations (documented, not bugs):** duplicate detection is token-fingerprint (a
  full-identifier-rename copy, or a *reflowed*/line-interleaved copy, or a *relocated* pre-existing
  duplicate is not caught); a base-vs-head duplicate index and an AST/token-shingle detector are the
  future upgrades.

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
| 5 | **Producer execution failure fails open.** `run_vulture`/`run_duplicate` ignored the subprocess return code + stderr, always returning `ok=True`. A tool that crashed / hit bad args / emitted unparseable output → `findings=[]` → **green**. | vulture exit codes are an **allow-list**: `0/3`=ok, `1`=partial (incomplete); **ANY other code (2, a traceback, a signal-kill like 137, an unknown value) → malfunction → raise** — a required gate must not guess completeness. `git ls-files` failure and an untokenizable corpus file are handled too (raise / partial). | unit: vulture rc∈{2,4,137,-9}/traceback → raise; rc 0/3→ok, 1→partial; git-ls-files fail→raise; untokenizable file→partial. |

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
| 6 | **Duplicate gap was only closed within a subsystem.** The corpus was bounded (pylint timed out on the full tree), so `src/core/a` copied into `src/api/b` could pass. | **Global** duplicate detection via a **tokenize-based fingerprint index** over the *whole* production tree (no subsystem bound): a window of `DUP_MIN_LINES=10` consecutive significant (tokenized) lines whose fingerprint also appears in a **different** file is a cross-file duplicate. Normalization uses the real tokenizer, so a `#` **inside a string** is not mistaken for a comment (the naive `split('#')` got that wrong). O(total-lines), full gate **~10 s** end-to-end over 1167 files (pylint was >150 s → timeout). Deterministic, tool-version-independent. | **golden: cross-SUBSYSTEM copy (`src/api` copies `src/core`) → exit 1** ("duplicate block also in src/core/orig.py"). Runs with no external tool. |
| 7 | **Corpus/producer failures degraded silently.** `git ls-files` return code ignored → a failure shrank the corpus to changed-files-only. vulture `rc=1` (a file couldn't parse) only warned, even in enforce. | `candidate_corpus` **raises** on `git ls-files` failure. Producers return a 3-state status (`ok`/`absent`/`partial`); **any non-`ok` in ENFORCE → `HardGateError` → exit 2** (dry-run warns). A required gate never passes what it could not fully analyse. | unit: git-ls-files failure → raise; vulture rc=1 → `partial`; enforce+`partial`/`absent` → raise; dry-run+same → warn+continue; unreadable corpus file → `partial`. |

### Supporting changes
- **pylint removed entirely** from duplicate detection — with it go the JSON parsing, order-dependent
  anchor, module→path mapping, exit-code bit-mask, and version pinning. Only **vulture** remains an
  external producer (still pinned `==2.16`); the fingerprint index is stdlib (`hashlib`).
- **Fingerprint semantics documented:** normalization is by `tokenize` (comments/whitespace dropped
  correctly, including a `#` inside a string; identifier names kept — pylint's R0801 doesn't rename
  either). Catches verbatim / whitespace- or comment-different copies; a rename-heavy paraphrase is
  out of scope by design.
- vulture still runs over the **full tree** (5.5 s) so cross-file usage is correct (no false
  positives from changed-files-only).

## Round 4/5 — bypasses, fake-green, false-positives (human review + a 6-dimension adversarial self-review)

A human review and an independent multi-agent adversarial review (6 dimensions → each finding
reproduced-or-refuted; 23 confirmed) found further defects. All reproduced first, then fixed, then
locked with observed-RED (56 self-tests now).

| Defect | Fix |
|---|---|
| **`src/test_runtime.py` bypassed the whole gate** — `is_test_path` excluded any file *named* `test_*.py`. | Exclude only real test *locations* (a `tests`/`test` dir, `conftest.py`, `scripts/ci/test_*.py`); a production `test_*.py` is gated. |
| **vulture `rc=3` + unrecognised output → fake-green** — `rc=3` ("dead code found") with zero parsed findings, or any unrecognised line, silently passed. | `rc=3` must parse ≥1 finding, and any non-empty unrecognised line → **malfunction (exit 2)**. |
| **Non-ASCII filename bypass** — under default `core.quotePath` git octal-escapes a Chinese path in the diff/ls-files, so it was missed. | `-c core.quotePath=false` on every git call (+ `ls-files -z` for spaces). |
| **`diff.noprefix=true` → armed gate passes EVERY PR** — the `b/` prefix the parser needs disappears → `changed_lines` empty. | `-c diff.noprefix=false`. |
| **`tokenize` fixed a naive `#`-in-string bug** (round 4). | Normalization is by the real tokenizer; comments/whitespace dropped correctly, names kept. |
| **Unexpected vulture exit codes fell through to 'ok'** (round 4). | Allow-list: `0/3`=ok, `1`=partial, **anything else → exit 2**. |
| **Reindenting a PRE-EXISTING duplicate red an innocent PR** (false positive — broke the core promise). | `git diff -w`: a whitespace-only reindent is not a change; genuinely-new copies still caught. (Residual: a *relocated* pre-existing dup — documented; base-index is the future fix.) |
| **`HARD_GATE_ENFORCE` only honoured exact `"1"`** — arming with `true`/`yes` silently stayed dry-run. | Accept `{1,true,yes,on}` (case-insensitive). |
| **Corpus roots gap** — 6 production `.py` under `clients/config/demo/examples/sdk` were invisible. | Enumerate **all** tracked non-test `.py`, not just `src/scripts`. |
| **No per-file size cap** (DoS). | Skip a file over 2 MB (→ partial, fail-closed in enforce). |
| **Docs described the old (pylint/subsystem/paths) implementation** — would mis-configure branch protection. | Script docstring, MD (this "current state" header), and PR body synced to the fingerprint design and the `Hard Gate (diff-scoped)` context name. |

## What is still NOT done (honest scope)

- **Arming remains owner-only** and is deliberately not performed here: it requires (a) uncommenting
  `HARD_GATE_ENFORCE: "1"`, and (b) adding the check to branch protection. This PR makes the gate
  *safe to arm* — it does not arm it. `merged != enabled != safe to enable` (§7.2).
- **Duplicate detection is token-fingerprint, keeping names** — it will not catch a copy that
  renames identifiers throughout. Documented trade-off; a rename-insensitive AST index is a future
  upgrade if a real case appears.
- These fixes make the gate *requireable*; whether to require it is a branch-protection decision.
