# Phase 0 · A4 verification — diff-scoped hard-gate mechanics

Companion to `PHASE0_A4_HARD_GATE_MECHANICS_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`, **executed locally** (guardrail: run, don't just compile).

---

## 1. Filter + hunk-parser unit checks — 8/8 PASS

`python3 scripts/ci/test_hard_gate_diff.py` (no pytest, no deps — runs on local py3.9):

```
PASS  catches a violation on a CHANGED line (gate can fail)
PASS  catches a duplicate on a CHANGED line
PASS  IGNORES pre-existing debt on an UNCHANGED line (no 误伤)
PASS  IGNORES a violation in an UNTOUCHED file (no 误伤)
PASS  exactly the two changed-line findings survive
PASS  hunk parser: foo.py added lines == {11,12,13}
PASS  hunk parser: bar.py changed line == {5}
PASS  hunk parser: deletion-only hunk adds nothing (no line 50/51)
ALL PASS -- diff filter + hunk parser correct   (exit 0)
```

## 2. Observed-RED level 1 — synthetic (the filter genuinely discriminates)

```
same violation on changed line 11  -> caught=True
same violation on unchanged line 400 -> caught=False   (must be False)
```

The identical finding is caught or dropped purely by whether its line was changed — proving the gate can both fail (new debt) and stay quiet (pre-existing debt).

## 3. Observed-RED level 2 — real git, full pipeline

Appended a dead function to a real tracked file and drove the real `git diff`:

```
real git diff detected 4 changed line(s) in similarity.py: [1141, 1142, 1143]
gate would flag 1 NEW violation on changed line 1143: CAUGHT
✅ control: pre-existing (unchanged-line) finding NOT flagged
✅ reverted probe; observed-RED (real git + full filter pipeline) banked
git status --porcelain src/ -> clean
```

So the gate integrates with real git (not just a synthetic diff string), catches new dead code on a changed line, and does not punish the pre-existing line — end to end.

## 4. What the CI job proves that local cannot
Local py3.9 + no vulture/pylint means the *finding-producers* aren't exercised here — only the diff-filter (the risky part) is. `hard-gate.yml` installs `vulture`+`pylint` on py3.11 and runs them against changed files, so the full producer→filter→verdict path is exercised in CI. The self-test step (`test_hard_gate_diff.py`) runs with **no `|| true`**, so a regression in the filter fails the job. Stated plainly rather than claimed as fully-local.

## 5. Safety properties asserted
- **No 误伤**: covered by two unit checks + the real-git control (unchanged-line finding not flagged).
- **No fabricated verdict**: a missing linter emits `::warning:: tool unavailable`, never a pass or fail.
- **Cannot silently pass on a broken filter**: the self-test has no `|| true`.
- **Dry-run by default**: `HARD_GATE_ENFORCE` unset → exit 0; enforce is a one-line owner flip.
- **No branch-protection touch**: arming as *required* is documented as owner-only; the workflow performs no protection change.

## 6. Honesty note
This is the *mechanics* half, delivered as a branch (no PR — 4 already open). Arming it (enforce + required) is the owner's half and is not done. The gate is proven on the filter/real-git path; the producer path is proven only in CI. No code in `code-quality.yml` was changed — stripping its `|| true` is part of the owner's arming batch, not this groundwork.
