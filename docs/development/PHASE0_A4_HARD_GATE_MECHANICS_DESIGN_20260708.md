# Phase 0 · A4 — diff-scoped hard-gate mechanics (my half of the split)

- **Status**: for-review groundwork. Branch only, **no PR** (backlog cap). Ships DRY-RUN; arming is owner-only.
- **Grounded on** `origin/main @ 8337ea6e`, executed locally (not just compiled).
- **The owner/agent split** (owner's framing): *I build the path-filtered, dry-run-first hard-gate mechanics + evidence; the owner adds the check to the branch-protection required list.* This is my half.

---

## 1. Why the existing gates can't just be flipped hard

`code-quality.yml` runs the dead-code and duplicate-code checks over the **whole, already-bloated tree** and ends every step in `|| true`:

```
:165  vulture src/ --min-confidence 80 || true
:123  pylint src/ --disable=all --enable=duplicate-code --min-similarity-lines=10 || true
```

Stripping `|| true` would red **every** PR on pre-existing debt, and there's no vulture baseline to diff against. That's the trap the #499 audit flagged (and why the roadmap said "path-filtered / dry-run-first").

## 2. The mechanic: diff-scoped, not whole-tree

`scripts/ci/hard_gate_diff.py` fails only on violations located on lines **this PR added or changed**:

1. `changed_lines(base)` parses `git diff --unified=0 {base}...HEAD` hunk headers → `{file: {added line numbers}}` (deletion-only hunks contribute nothing).
2. Finding-producers (`vulture`, `pylint --enable=duplicate-code`) run on the changed files and emit `(file, line, message)`.
3. `new_violations()` keeps only findings whose `(file, line)` is in the changed set.

So a PR **cannot be failed by dead code in a file it didn't touch** — the property that makes the gate safe to require. Pre-existing debt is left for a separate, deliberate cleanup, not dumped on unrelated authors.

## 3. Dry-run-first, and arming is owner-only

- **Ships dry-run**: `HARD_GATE_ENFORCE` unset → the gate prints `::warning::` for what it *would* block and exits 0. It can be observed against real PRs before it bites.
- **Arming = two owner actions**, neither performed here:
  1. set `HARD_GATE_ENFORCE=1` in `hard-gate.yml` (dry-run → blocking);
  2. add `Hard Gate (diff-scoped, dry-run)` to the branch-protection **required** list.
- The workflow **never touches branch protection**. `.github/workflows/hard-gate.yml` is path-filtered to `src/**` + the script + itself, so it doesn't even run on unrelated PRs.

## 4. Fail-safe producer handling
A missing linter must not fabricate a verdict: if `vulture`/`pylint` isn't installed, the gate emits `::warning:: tool unavailable` and does **not** fail. Only a real finding on a changed line fails (in enforce). The diff-filter self-test (`test_hard_gate_diff.py`) runs with **no** `|| true` — a broken filter fails the job, because the filter is the load-bearing part.

## 5. Verification (executed, see the verification MD)
- 8/8 filter + hunk-parser unit checks pass.
- **Observed-RED, two levels**: (a) synthetic — same violation caught on a changed line, ignored on an unchanged line; (b) real-git end-to-end — appended a dead function to `similarity.py`, the gate read the real diff and flagged it on the changed line (1143), and a control finding on an unchanged line was **not** flagged (anti-误伤). Reverted clean.

## 6. Composition with #500 / follow-ups
- This is **independent of the merge backlog** — it hardens the *existing* `code-quality.yml` dead-code/dup checks, which are on `main` today.
- Once **#500** lands, its `prune-safety` check joins the same "arm as required" batch (it's already hard/observed-RED); `Hard Gate` is the diff-scoped complement for dead-code/duplicate.
- Owner batch, when ready: strip `|| true` from `code-quality.yml`'s dead-code/dup steps (or delete them in favor of this diff-scoped gate), set `HARD_GATE_ENFORCE=1`, add `Hard Gate` + `Prune Safety` to required.
- **Model used**: built + executed in-session on Opus 4.8 (Fable 5 at daily cap). The mechanic is spec-complete enough that its future maintenance is Sonnet-5-class.
