#!/usr/bin/env markdown
# Baseline Evaluation (Assembly)

This baseline evaluation is a fast, deterministic check for the assembly
understanding pipeline. It is intended to be run before and after model/rule
changes to verify no regressions.

## Run

```
make test-baseline
```

Or directly:

```
python3 scripts/run_baseline_evaluation.py
```

## Output

Results are written to:
- `evaluation_results/baseline_YYYYMMDD_HHMMSS.json`

The exit code is non-zero if the pass rate drops below the baseline threshold.

## What It Measures

The baseline evaluates:
- Edge F1 score
- Joint type accuracy
- Evidence coverage
- Overall score / pass rate

These are aggregated across the built-in golden cases embedded in the script.

## When To Use

- Before/after changes to assembly rules or inference logic
- As a quick CI gate when touching `src/assembly/` or related heuristics
