# PR412 Remote Failure Triage Batch 2 Development Plan

Date: 2026-04-17
PR: #412
Branch: `phase3-vector-pipeline-20260417`

## Context

After the `unit-tier` rerun was stabilized, a new remote failure appeared in:

- `Stress and Observability Checks / stress-unit-tests`

The failure occurred during the Faiss health smoke step, where the workflow tried to parse `faiss_health.json` and hit `JSONDecodeError`.

## Findings

Local verification showed `/api/v1/health/faiss/health` still returns valid JSON when the API is actually ready. The failure is therefore best explained by workflow timing:

- `Start API server` only does `sleep 3`
- the subsequent `curl -s ... | tee faiss_health.json` assumes the service is ready immediately
- if the runner is slow, the file can be empty or incomplete and `json.load()` fails

This is a workflow robustness issue, not a regression in the vector pipeline slice.

## Planned changes

1. Add an explicit `Wait for API readiness` step using `curl -fsS ${API_BASE_URL}/health`
2. Change both Faiss health fetches to retry loops:
   - use `curl -fsS`
   - retry up to 10 attempts
   - only proceed once JSON parsing and key validation succeed
3. Add workflow regression assertions so this retry/readiness logic cannot silently disappear

## Scope

Files expected to change:

- `.github/workflows/stress-tests.yml`
- `tests/unit/test_stress_workflow_workflow_file_health.py`

Supporting verification/documentation:

- `docs/development/PR412_REMOTE_FAILURE_TRIAGE_BATCH2_VERIFICATION_20260417.md`

## Non-goals

- no change to Faiss health API response schema
- no change to vector registration logic
- no production code change in `src/`

