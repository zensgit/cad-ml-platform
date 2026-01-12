# METRICS_STRICT_CI_JOB_DESIGN

## Goal
- Add a dedicated CI job that runs metrics contract tests with STRICT_METRICS enabled.

## Approach
- Extend `.github/workflows/observability-checks.yml` with a `metrics-contract-strict` job.
- Reuse the existing dependency install steps and set `STRICT_METRICS=1` for the test run.

## Rationale
- Keeps strict-mode verification consistently enforced in CI without changing runtime behavior.
