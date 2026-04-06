# Eval Reporting Stack Notification Consumer Alignment — Design

日期：2026-03-30

## Scope

Batch 7B of the Claude Collaboration Eval Reporting plan:

- Add `--stack-summary-json` and `--index-json` CLI args to `notify_eval_results.py`
- Inject eval reporting stack status into Slack and email payloads
- Update workflow notify step to pass the two paths
- Preserve standalone `--report-url` backward compatibility

## Design Decisions

### 1. New CLI Args

```
--stack-summary-json  Path to eval_reporting_stack_summary.json
--index-json          Path to eval_reporting_index.json
```

Both are optional. When empty or file missing, stack status is `available: false` and no stack fields are injected.

### 2. Helper Function — `_load_stack_status`

Internal function in `notify_eval_results.py`:

```python
_load_stack_status(stack_summary_json, index_json) -> dict
```

Returns: `available`, `status`, `missing_count`, `stale_count`, `mismatch_count`, `landing_page`, `static_report`, `interactive_report`.

### 3. Slack Payload Injection

When `available` is True, a new field is appended to the first attachment:

```json
{
  "title": "Eval Reporting Stack",
  "value": "status=ok, missing=0, stale=0, mismatch=0",
  "short": false
}
```

### 4. Email Payload Injection

When `available` is True, a one-line summary is appended to both `text_body` and `html_body`.

### 5. Workflow Integration

Both notify invocations in `evaluation-report.yml` now pass:

```bash
--stack-summary-json reports/ci/eval_reporting_stack_summary.json \
--index-json reports/eval_history/eval_reporting_index.json
```

### 6. Backward Compatibility

- `--report-url` standalone calls still work
- Threshold-breach logic unchanged
- No new blocking conditions introduced
