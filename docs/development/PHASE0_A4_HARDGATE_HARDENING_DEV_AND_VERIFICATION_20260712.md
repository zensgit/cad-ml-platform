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

## What is still NOT done (honest scope)

- **Arming remains owner-only** and is deliberately not performed here: it requires (a) uncommenting
  `HARD_GATE_ENFORCE: "1"`, and (b) adding the check to branch protection. This PR makes the gate
  *safe to arm* — it does not arm it. `merged != enabled != safe to enable` (§7.2).
- These fixes make the gate *requireable*; whether to require it is a branch-protection decision.
