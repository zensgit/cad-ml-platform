# Strict-Real And Superpass Wrapper Workflows Validation

Date: 2026-03-17

## Scope

This change completes wrapper workflow alignment for the remaining two GH dispatch chains:

- `hybrid-superpass-e2e.yml`
- `hybrid-blind-strict-real-e2e.yml`

Both workflows now follow the same structure already used by `evaluation-soft-mode-smoke.yml`:

1. dispatch the target workflow
2. persist a JSON summary
3. render a Markdown summary
4. upload both JSON and Markdown as artifacts
5. append the rendered Markdown to `GITHUB_STEP_SUMMARY`

## Files

- `.github/workflows/hybrid-superpass-e2e.yml`
- `.github/workflows/hybrid-blind-strict-real-e2e.yml`
- `tests/unit/test_hybrid_superpass_e2e_workflow.py`
- `tests/unit/test_hybrid_blind_strict_real_e2e_workflow.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `Makefile`

## Validation

### Direct tests

```bash
pytest -q \
  tests/unit/test_hybrid_superpass_e2e_workflow.py \
  tests/unit/test_hybrid_blind_strict_real_e2e_workflow.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

Result:

- `43 passed`

### Make target regression

```bash
make validate-hybrid-superpass-workflow
```

Result:

- `90 passed`

```bash
make validate-hybrid-blind-strict-real-e2e-gh
```

Result:

- `54 passed`

## Notes

- Validation in this round is repository-local and regression-based.
- No live GitHub workflow dispatch was executed for the new wrapper workflow files in this round.
- The implementation intentionally only touches low-conflict CI files and avoids unrelated model or API changes already present in the worktree.
