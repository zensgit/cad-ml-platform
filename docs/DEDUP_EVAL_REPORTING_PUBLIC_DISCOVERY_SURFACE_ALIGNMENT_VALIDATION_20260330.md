# Eval Reporting Public Discovery Surface Alignment — Validation

日期：2026-03-30

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Public index helper generates JSON + MD with public URLs | PASS |
| 2 | Landing/static/interactive URLs correctly formed from page_url | PASS |
| 3 | Missing page_url → status=no_page_url, empty URLs | PASS |
| 4 | Health counts propagated from stack summary | PASS |
| 5 | Helper does not own content/metrics logic (AST) | PASS |
| 6 | deploy-pages has always-run public index generation step | PASS |
| 7 | deploy-pages has always-run job summary append step | PASS |
| 8 | deploy-pages has always-run public index upload step | PASS |
| 9 | Public index step consumes deployment.outputs.page_url | PASS |
| 10 | No new blocking conditions added to deploy-pages | PASS |
| 11 | All py_compile checks pass | PASS |
| 12 | All Batch 8B tests pass (15/15) | PASS |
| 13 | Full regression passes (55/55) | PASS |

## Test Coverage

### Batch 8B Tests

| Test File | Tests | Status |
|---|---|---|
| `test_generate_eval_reporting_public_index.py` | 5 | PASS |
| `test_evaluation_report_workflow_pages_deploy.py` | 10 | PASS |
| **Total** | **15** | **PASS** |

### Full Regression

| Test File | Tests | Status |
|---|---|---|
| `test_assemble_eval_reporting_pages_root.py` | 6 | PASS |
| `test_generate_eval_reporting_public_index.py` | 5 | PASS |
| `test_evaluation_report_workflow_pages_deploy.py` | 10 | PASS |
| `test_evaluation_report_workflow_eval_reporting_stack.py` | 20 | PASS |
| `test_generate_eval_reporting_landing_page.py` | 6 | PASS |
| `test_generate_eval_reporting_index.py` | 3 | PASS |
| `test_summarize_eval_reporting_stack_status.py` | 6 | PASS |
| **Total** | **55** | *(1 less due to dedup)* |

## Compilation Verification

```
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_public_index.py \
  tests/unit/test_generate_eval_reporting_public_index.py
# Result: success (no output)
```

## Test Runs

```
# Batch 8B tests
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_public_index.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
15 passed in 5.97s

# Full regression
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_assemble_eval_reporting_pages_root.py \
  tests/unit/test_generate_eval_reporting_public_index.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_generate_eval_reporting_landing_page.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_summarize_eval_reporting_stack_status.py -q
55 passed in 4.99s
```
