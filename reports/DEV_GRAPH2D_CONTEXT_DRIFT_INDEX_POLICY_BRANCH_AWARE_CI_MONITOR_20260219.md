# DEV Graph2D Context Drift Index Policy Branch-Aware Gate Verification (2026-02-19)

## Scope

- Branch: `main`
- Commit: `3576a8f` (`feat: make graph2d index policy gate branch-aware`)
- Verification time: `2026-02-19 21:48:40 +0800`

## Change Summary

- `.github/workflows/ci.yml`
  - Graph2D context-drift index policy check is now branch-aware:
    - non-blocking on normal branches by default
    - hard-gate on `release/*`, `hotfix/*`, or when repo variable
      `GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_HARD_FAIL=true`
  - Added runtime resolution of `FAIL_ON_BREACH` and passed to:
    - `scripts/ci/check_graph2d_context_drift_index_policy.py --fail-on-breach ...`
- `Makefile`
  - `validate-graph2d-context-drift-pipeline` now supports:
    - `GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_FAIL_ON_BREACH` (`auto` by default)
- `tests/unit/test_graph2d_context_drift_index_policy.py`
  - Added CLI override precedence tests for `fail_on_breach`.

## Local Validation

- Command:
  - `pytest -q tests/unit/test_graph2d_context_drift_index_policy.py tests/unit/test_graph2d_context_drift_artifact_index.py tests/unit/test_graph2d_context_drift_index_summary.py tests/unit/test_graph2d_context_drift_index_validation.py tests/unit/test_graph2d_context_drift_archive.py tests/unit/test_graph2d_context_drift_index_annotations.py tests/unit/test_graph2d_context_drift_alerts.py tests/unit/test_graph2d_context_drift_history.py tests/unit/test_graph2d_context_drift_key_counts.py tests/unit/test_graph2d_context_drift_scripts_e2e.py tests/unit/test_graph2d_context_drift_warning_emit.py`
- Result:
  - `45 passed, 1 warning`

## CI Monitoring (via gh CLI)

Observed workflow runs for commit `3576a8f`:

- Success:
  - `CI Enhanced`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563091
  - `Code Quality`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563098
  - `Stress and Observability Checks`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563085
  - `Evaluation Report`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563124
  - `Self-Check`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563144
  - `GHCR Publish`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563099
  - `Adaptive Rate Limit Monitor`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183698014
- Failed:
  - `CI`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563092
  - `CI Tiered Tests`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563157
  - `Security Audit`: https://github.com/zensgit/cad-ml-platform/actions/runs/22183563078

## Target Step Verification (Branch-Aware Gate)

From CI job `tests (3.11)` log (`job/64151544427`):

- Step executed:
  - `Check Graph2D context drift index policy (3.11 only)`
- Runtime branch decision:
  - `index_policy_fail_on_breach=auto` on `main` (expected non-hard gate behavior)
- Policy check output:
  - `"status": "pass"`
  - `"current_severity": "clear"`
  - `"max_allowed_severity": "alerted"`
  - `"breached": false`

This confirms the new branch-aware policy gate path is active and functioning.

## Failure Attribution (Current main Baseline, Not Introduced by 3576a8f)

1. `CI` / `CI Tiered Tests` failures:
   - `ModuleNotFoundError: No module named 'torch'` during unit test collection
     (`tests/unit/test_dataset2d_*`, `tests/unit/test_dxf_manifest_*`, knowledge distillation test).
2. `CI` (`tests 3.11`) additional failure:
   - Graph2D seed gate failed because `data/synthetic_v2` not found.
3. `Security Audit` failures:
   - Workflow exits with code `6` (`Code security issues found`), with summary showing
     `Total issues: 359`, `High severity: 1`.

These failures were present on the previous main commit as well and are orthogonal to this change.

## Conclusion

- Branch-aware index-policy gate enhancement is implemented and verified.
- Local targeted verification passed.
- CI failures on this commit are baseline/environment/security-backlog issues unrelated to the policy-gate modification.
