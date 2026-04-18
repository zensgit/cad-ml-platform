# PR430 Remote Failure Triage Development Plan

## Goal
- fix the first remote failure on `PR #430` without changing vector migration behavior
- harden `CI Enhanced / Coverage Report` against transient PyPI SSL download failures

## Scope
- update `.github/workflows/ci-enhanced.yml` coverage job install step with retry logic
- add a workflow regression test to lock the retry wiring

## Risk Controls
- limit the change to the coverage job only
- preserve the existing install order and optional `requirements-dev.txt` behavior
- validate with a focused workflow regression test

## Validation Plan
- `python3 -m py_compile tests/unit/test_ci_enhanced_coverage_install_retry.py`
- `.venv311/bin/flake8 tests/unit/test_ci_enhanced_coverage_install_retry.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_ci_enhanced_coverage_install_retry.py`
