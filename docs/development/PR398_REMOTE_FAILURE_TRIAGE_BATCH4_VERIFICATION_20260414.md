# PR398 Remote Failure Triage Batch 4 Verification

Date: 2026-04-14
Branch: `submit/local-main-20260414`
Scope: Graph2D DXF prediction contract regression in remote `CI`

## Remote Failure Confirmed

Failed workflow:

- `CI` run `24401743395`

Failed jobs:

- `tests (3.10)` job `71275155556`
- `tests (3.11)` job `71275155564`

Common failure across both jobs:

- `graph2d_prediction.min_confidence` returned `0.35`
- tests still asserted `0.5`

Remote timestamps from the failing job:

- `Run unit tests` started at `2026-04-14T13:39:18Z`
- failed at `2026-04-14T13:52:06Z`

## Local Verification

Targeted regression file:

```bash
.venv311/bin/python -m pytest -q tests/integration/test_analyze_dxf_graph2d_prediction_contract.py
```

Result:

- `6 passed, 7 warnings`

## Sidecar Review

Command:

```bash
claude -p "Review the current unstaged diff in tests/integration/test_analyze_dxf_graph2d_prediction_contract.py for correctness regressions. Respond with either 'No findings' or short findings."
```

Result:

- `No findings`

## Outcome

The regression was a stale integration test baseline, not an API implementation bug.

The patched contract tests now:

- follow the config-derived Graph2D default threshold
- keep below-threshold test cases explicitly below that default
- preserve the original behavioral guarantees without hardcoding the superseded `0.5` baseline
