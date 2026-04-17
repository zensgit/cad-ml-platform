# PR412 Remote Failure Triage Batch 2 Verification

Date: 2026-04-17
PR: #412
Branch: `phase3-vector-pipeline-20260417`

## Remote failure

Workflow:

- `Stress and Observability Checks / stress-unit-tests`

Observed failure:

```text
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Location:

- Faiss health smoke check in `.github/workflows/stress-tests.yml`

## Root cause

The workflow started `uvicorn`, slept for 3 seconds, and then immediately executed:

```bash
curl -s ${API_BASE_URL}/api/v1/health/faiss/health | tee faiss_health.json
```

This is not robust on slower GitHub runners. The API endpoint is valid, but the workflow lacked readiness and retry guards before parsing JSON.

## Fix

Implemented:

1. `Wait for API readiness` step using `curl -fsS ${API_BASE_URL}/health`
2. Retry loop for initial Faiss health fetch
3. Retry loop for post-recover Faiss health fetch
4. Workflow regression test assertions for the new readiness/retry logic

## Local verification

Executed:

```bash
python3 - <<'PY'
import yaml, pathlib
with open('.github/workflows/stress-tests.yml') as f:
    yaml.safe_load(f)
print('yaml ok')
PY

.venv311/bin/python -m pytest -q \
  tests/unit/test_stress_workflow_workflow_file_health.py \
  tests/unit/test_additional_workflow_comment_helper_adoption.py

python3 scripts/ci/check_workflow_identity_invariants.py
```

Results:

- YAML parse: passed
- Pytest: `10 passed, 7 warnings`
- workflow identity invariants: passed

Additional local API sanity check:

- started local `uvicorn`
- requested `/api/v1/health/faiss/health`
- confirmed valid JSON body with required Faiss health fields

## Outcome

Batch 2 hardens the stress workflow against startup timing noise without changing application behavior.

