# Claude Collaboration Batch 13 Validation Ledger

日期：2026-04-01

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 13A Status

- `status`: `complete`
- `implementation_scope`: `publish payload / policy from release-draft payload + deploy-pages workflow integration`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `new_files`:
  - `scripts/ci/generate_eval_reporting_release_draft_publish_payload.py`
  - `tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py`
  - `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_PAYLOAD_ALIGNMENT_DESIGN_20260401.md`
  - `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_PAYLOAD_ALIGNMENT_VALIDATION_20260401.md`
- `tests_run`: `49`
- `test_results`: `49 passed in 22.45s`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 13A Evidence

- `release_draft_publish_payload_artifact_proof`: `test_publish_payload_with_ready_release — verifies surface_kind, publish_policy=disabled_by_default, publish_allowed=true, publish_requires_explicit_enable=true, github_release_tag`
- `job_summary_proof`: `test_deploy_pages_has_publish_payload_summary_step — verifies always-run step with GITHUB_STEP_SUMMARY and eval_reporting_release_draft_publish_payload.md`
- `artifact_upload_proof`: `test_deploy_pages_has_publish_payload_upload_step — verifies always-run upload with eval_reporting_release_draft_publish_payload in path`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_PAYLOAD_ALIGNMENT_DESIGN_20260401.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_PAYLOAD_ALIGNMENT_VALIDATION_20260401.md`

### Batch 13A Command Log

```text
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_release_draft_publish_payload.py \
  tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 13A Result Log

```text
py_compile: success (no output)

pytest: 49 passed in 22.45s
```

---

## Batch 13B Status

- `status`: `complete`
- `implementation_scope`: `optional GitHub release draft publish automation from publish payload`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `new_files`:
  - `scripts/ci/post_eval_reporting_release_draft_publish.js`
  - `tests/unit/test_post_eval_reporting_release_draft_publish_js.py`
  - `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_AUTOMATION_ALIGNMENT_DESIGN_20260401.md`
  - `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_AUTOMATION_ALIGNMENT_VALIDATION_20260401.md`
- `tests_run`: `55 (batch 13B) + 70 (full regression)`
- `test_results`: `55 passed in 9.79s (batch 13B), 70 passed in 7.23s (full regression)`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 13B Evidence

- `release_draft_publish_artifact_proof`: `test_publish_result_has_required_fields — verifies surface_kind, publish_enabled, publish_attempted, publish_succeeded, publish_mode, github_release_tag, github_release_id and MD with Publish Attempted / Succeeded / Mode / readiness / Tag`
- `job_summary_proof`: `test_deploy_pages_has_publish_result_summary_step — verifies always-run step with GITHUB_STEP_SUMMARY and eval_reporting_release_draft_publish_result.md`
- `artifact_upload_proof`: `test_deploy_pages_has_publish_result_upload_step — verifies always-run upload with eval_reporting_release_draft_publish_result in path`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_AUTOMATION_ALIGNMENT_DESIGN_20260401.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_AUTOMATION_ALIGNMENT_VALIDATION_20260401.md`

### Batch 13B Command Log

```text
node --check scripts/ci/post_eval_reporting_release_draft_publish.js

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py \
  tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 13B Result Log

```text
node --check: success (no output)

pytest batch 13B: 55 passed in 9.79s

pytest full regression: 70 passed in 7.23s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 13A Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。py_compile 通过；定向回归实测 49 passed in 9.37s。已确认 publish payload 只消费 release-draft payload，workflow 已接入 helper / summary / upload，且 repository_url 由上游 release-draft payload 透传进入 publish payload。`

### Batch 13B Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。node --check 通过；Batch 13B 定向回归实测 55 passed in 10.27s；联动回归实测 70 passed in 7.00s。已确认 publish automation 只消费 publish payload artifact，workflow 中默认固定 publishEnabled=false、continue-on-error=true，仅在显式 enable 且 payload 允许时才尝试创建 draft release，且 publish result artifact / summary / upload 已完整接线。`
