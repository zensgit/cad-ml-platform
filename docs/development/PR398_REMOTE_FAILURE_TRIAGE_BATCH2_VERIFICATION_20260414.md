# PR398 Remote Failure Triage Batch 2 Verification

Date: 2026-04-14
PR: `#398`
Head: `submit/local-main-20260414`

## Fixed Issues

### 1. CI Tiered Tests env drift

Changed:

- `tests/unit/test_comment_evaluation_report_pr_js.py`

Validation:

```bash
env \
  GITHUB_EVENT_INPUTS_JSON='{"min_combined":"0.99"}' \
  GITHUB_STEPS_JSON='{"graph2d_review_gate":{"outputs":{"status":"passed","exit_code":"0","headline":"ok"}}}' \
  CI_WATCH_SUMMARY_JSON_FOR_COMMENT='/tmp/leak-ci-watch.json' \
  WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT='/tmp/leak-workflow-health.json' \
  WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT='/tmp/leak-workflow-inventory.json' \
  EVALUATION_COMMENT_SUPPORT_MANIFEST_JSON_FOR_COMMENT='/tmp/leak-comment-support.json' \
  EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT='/tmp/leak-stack.json' \
  .venv311/bin/python -m pytest -q \
    tests/unit/test_comment_evaluation_report_pr_js.py::test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output \
    -vv
```

Result:

- `1 passed`

Additional regression:

```bash
.venv311/bin/python -m pytest -q tests/unit/test_comment_evaluation_report_pr_js.py
```

Result:

- `26 passed`

### 2. Security Audit B324 in low_conf_queue

Changed:

- `src/ml/low_conf_queue.py`

Validation:

```bash
rg -n "md5|B324" src/ml/low_conf_queue.py
```

Result:

- no matches

Regression:

```bash
.venv311/bin/python -m pytest -q tests/unit/test_low_conf_queue.py
```

Result:

- `23 passed`

Targeted governance coverage:

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_low_conf_queue.py \
  tests/unit/test_training_data_governance.py \
  -k 'low_conf or human_verified_entries or enqueue_sets_sample_source or check_retrain_uses_eligible or export_default_only_eligible'
```

Result:

- `27 passed, 14 deselected`

Note:

- `bandit` is not installed in local `.venv311`, so local verification used direct source inspection plus affected-unit coverage.

### 3. Adaptive Rate Limit Monitor missing checkout

Changed:

- `.github/workflows/adaptive-rate-limit-monitor.yml`
- `tests/unit/test_additional_workflow_comment_helper_adoption.py`

Validation:

```bash
.venv311/bin/python -m pytest -q tests/unit/test_additional_workflow_comment_helper_adoption.py
```

Result:

- `6 passed`

Workflow consistency:

```bash
python3 scripts/ci/check_workflow_identity_invariants.py
```

Result:

- `ok: workflow identity check passed for 12 check(s)`

## Remaining Separate Issues

Not fixed in this batch:

- repo-wide `Action Pin Guard` tag-pin debt
- unrelated `PR Auto Label and Comment` failure stream
- broader security baseline items outside `src/ml/low_conf_queue.py`

## Conclusion

This batch closes three actionable remote failure sources with local verification:

- deterministic Node test harness for evaluation-report comment checks
- removal of MD5 usage from `low_conf_queue`
- checkout restoration for adaptive-rate-limit PR comment job
