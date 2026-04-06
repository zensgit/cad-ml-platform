# Claude Collaboration Batch 5 Validation Ledger

日期：2026-03-30

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 5A Status

- `status`: `complete`
- `implementation_scope`: `evaluation-report workflow wiring to canonical refresh + diagnostics retention`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `new_files`:
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
  - `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_DESIGN_20260330.md`
  - `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_VALIDATION_20260330.md`
- `tests_run`: `36`
- `test_results`: `36 passed in 7.77s`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 5A Evidence

- `workflow_file`: `.github/workflows/evaluation-report.yml`
- `refresh_step_proof`: `"Refresh eval reporting stack" step calls refresh_eval_reporting_stack.py with continue-on-error: true and captures exit_code`
- `static_report_path_proof`: `REPORT_PATH=reports/eval_history/report_static verified in test_workflow_env_includes_eval_reporting_stack_vars`
- `interactive_report_path_proof`: `INTERACTIVE_REPORT_PATH=reports/eval_history/report_interactive verified in test_static_and_interactive_report_paths_are_distinct`
- `bundle_upload_proof`: `"Upload eval reporting stack artifacts" step includes eval_reporting_bundle.json`
- `health_upload_proof`: `"Upload eval reporting stack artifacts" step includes eval_reporting_bundle_health_report`
- `index_upload_proof`: `"Upload eval reporting stack artifacts" step includes eval_reporting_index`
- `fail_closed_after_upload_proof`: `test_fail_step_exists_after_uploads verifies fail step index > all upload step indices; test_fail_step_checks_refresh_exit_code verifies exit code check`
- `design_md`: `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_DESIGN_20260330.md`
- `validation_md`: `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_VALIDATION_20260330.md`

### Batch 5A Command Log

```text
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_check_eval_reporting_bundle_health.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_refresh_eval_reporting_stack.py -q
```

### Batch 5A Result Log

```text
py_compile: success (no output)

pytest: 36 passed in 7.77s
```

---

## Batch 5B Status

- `status`: `complete`
- `implementation_scope`: `workflow stack summary helper + GITHUB_STEP_SUMMARY append + summary artifact upload + stale weekly_summary reference fix`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `new_files`:
  - `scripts/ci/summarize_eval_reporting_stack_status.py`
  - `tests/unit/test_summarize_eval_reporting_stack_status.py`
  - `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_DESIGN_20260330.md`
  - `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_VALIDATION_20260330.md`
- `tests_run`: `41 (batch 5B) + 55 (full regression)`
- `test_results`: `41 passed in 5.36s (batch 5B), 55 passed in 5.84s (full regression)`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 5B Evidence

- `workflow_file`: `.github/workflows/evaluation-report.yml`
- `summary_helper`: `scripts/ci/summarize_eval_reporting_stack_status.py`
- `summary_json`: `reports/ci/eval_reporting_stack_summary.json`
- `summary_md`: `reports/ci/eval_reporting_stack_summary.md`
- `job_summary_append_proof`: `test_append_to_job_summary_step_exists — verifies always() step that cats summary MD to GITHUB_STEP_SUMMARY`
- `annotation_step_proof`: `summary step itself serves as the annotation surface via GITHUB_STEP_SUMMARY`
- `artifact_upload_proof`: `test_stack_summary_upload_step_exists — verifies always() upload step for eval_reporting_stack_summary`
- `final_fail_order_proof`: `test_summary_and_upload_before_fail_step — verifies summary/append/upload indices < fail step index`
- `stale_reference_fix_proof`: `test_stale_weekly_summary_reference_removed — verifies weekly_summary.outputs.output_md no longer in Create job summary`
- `design_md`: `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `validation_md`: `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

### Batch 5B Command Log

```text
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/summarize_eval_reporting_stack_status.py \
  tests/unit/test_summarize_eval_reporting_stack_status.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_summarize_eval_reporting_stack_status.py \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_check_eval_reporting_bundle_health.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_refresh_eval_reporting_stack.py -q

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_summarize_eval_reporting_stack_status.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_check_eval_reporting_bundle_health.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_refresh_eval_reporting_stack.py \
  tests/unit/test_generate_eval_report.py \
  tests/unit/test_generate_eval_report_v2.py -q
```

### Batch 5B Result Log

```text
py_compile: success (no output, 2 files compile cleanly)

pytest batch 5B: 41 passed in 5.36s

pytest full regression: 55 passed in 5.84s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 5A Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。实际复跑 py_compile 通过，pytest 为 36 passed in 9.86s。Batch 5A 已把 evaluation-report workflow 默认接到 refresh_eval_reporting_stack.py，并新增 static / interactive / top-level stack 三类 artifact upload，最终 fail step 也位于 upload 之后。非阻塞备注：workflow 的 job summary 里仍残留对已删除 steps.weekly_summary.outputs.output_md 的引用，这不影响 Batch 5A 主契约验收，但应在 Batch 5B 做 top-level stack summary surface 时一并替换掉。`

### Batch 5B Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。实际复跑 py_compile 通过；Batch 5B 定向回归为 41 passed in 22.88s，合并回归为 55 passed in 22.97s。新增的 summarize_eval_reporting_stack_status.py 保持在 thin summary helper 边界内，只消费 bundle / health / index，并产出 workflow-friendly JSON / Markdown summary。evaluation-report.yml 已新增 always-run summary、GITHUB_STEP_SUMMARY append、summary artifact upload，且最终 fail step 仍位于这些步骤之后。Batch 5A 中残留的 steps.weekly_summary.outputs.output_md 引用也已移除。`
