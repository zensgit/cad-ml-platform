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

### Live GitHub Actions validation

Triggered workflow:

```bash
gh workflow run hybrid-superpass-e2e.yml \
  --ref feat/hybrid-blind-drift-autotune-e2e \
  -f ref=feat/hybrid-blind-drift-autotune-e2e
```

Wrapper run:

- workflow: `Hybrid Superpass E2E`
- run_id: `23172013969`
- run_url: `https://github.com/zensgit/cad-ml-platform/actions/runs/23172013969`
- result: wrapper executed successfully end-to-end, produced JSON and Markdown artifacts, then failed because the nested `evaluation-report.yml` run returned failure

Nested dispatched run:

- workflow: `Evaluation Report`
- run_id: `23172020610`
- run_url: `https://github.com/zensgit/cad-ml-platform/actions/runs/23172020610`
- failure step: `Fail workflow when Hybrid superpass strict check requires blocking`

Artifact confirmation:

- downloaded artifact contained:
  - `hybrid_superpass_e2e_summary.json`
  - `hybrid_superpass_e2e_summary.md`
- rendered markdown correctly summarized the nested run id, run url, and failure diagnostics

## Notes

- `hybrid-superpass-e2e` is now validated both locally and on GitHub Actions.
- The observed remote failure was downstream business logic enforcement, not wrapper wiring failure.
- `hybrid-blind-strict-real-e2e` was validated locally in this round; no live remote dispatch was triggered because it needs an explicit DXF directory input meaningful to the runner environment.
- The implementation intentionally only touches low-conflict CI files and avoids unrelated model or API changes already present in the worktree.
