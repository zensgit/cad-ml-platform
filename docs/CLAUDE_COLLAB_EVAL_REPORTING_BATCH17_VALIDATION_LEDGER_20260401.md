# Claude Collaboration Batch 17 Validation Ledger

日期：2026-04-01

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 17A Status

- `status`: `complete`
- `implementation_scope`: `Phase 1 zero-consumer surface removal: signature_policy, retry_plan, dry_run`
- `deleted_files`:
  - `scripts/ci/generate_eval_reporting_webhook_signature_policy.py`
  - `scripts/ci/generate_eval_reporting_webhook_retry_plan.py`
  - `scripts/ci/post_eval_reporting_release_draft_dry_run.js`
  - `tests/unit/test_generate_eval_reporting_webhook_signature_policy.py`
  - `tests/unit/test_generate_eval_reporting_webhook_retry_plan.py`
  - `tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_DESIGN_20260401.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_VALIDATION_20260401.md`
- `commands_run`: `py_compile + pytest + rg stale-reference verification`
- `key_findings`: `3 surfaces removed, 9 workflow steps removed, 6 files deleted, 66 remaining tests pass, zero stale references`
- `handoff_ready`: `yes`

### Batch 17A Evidence

- `zero_consumer_surface_removal_proof`: `rg stale-reference check → clean (no output)`
- `workflow_cleanup_proof`: `9 generate/append/upload steps + 3 sparse-checkout entries removed`
- `test_update_proof`: `15 tests deleted; 1 ordering test updated; 66 remaining tests pass`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_DESIGN_20260401.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_VALIDATION_20260401.md`

### Batch 17A Command Log

```text
rm scripts/ci/generate_eval_reporting_webhook_signature_policy.py \
   scripts/ci/generate_eval_reporting_webhook_retry_plan.py \
   scripts/ci/post_eval_reporting_release_draft_dry_run.js \
   tests/unit/test_generate_eval_reporting_webhook_signature_policy.py \
   tests/unit/test_generate_eval_reporting_webhook_retry_plan.py \
   tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py

python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_post_eval_reporting_webhook_delivery_js.py \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py

python3 -m pytest \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_post_eval_reporting_webhook_delivery_js.py \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py -q

rg -n "signature_policy|retry_plan|release_draft_dry_run" \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py
```

### Batch 17A Result Log

```text
py_compile: success (no output)

pytest: 66 passed in 26.76s

stale reference check: clean (no output)
```

---

## Batch 17B Status

- `status`: `complete`
- `implementation_scope`: `Phase 1 baseline hardening: 5 regression guard tests for removed/kept surfaces`
- `changed_files`:
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py` (5 new tests added)
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_DESIGN_20260401.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_VALIDATION_20260401.md`
- `commands_run`: `pytest + rg stale-reference verification`
- `key_findings`: `10 remaining deploy-pages surfaces baselined; 3 negative guards + 2 positive guards added; 59 tests pass`
- `handoff_ready`: `yes`

### Batch 17B Evidence

- `baseline_hardening_proof`: `design MD lists 10 remaining surfaces; 3 removed surfaces documented`
- `workflow_graph_proof`: `rg stale-reference check → clean; 30 deploy-pages content steps remain in workflow`
- `regression_guard_proof`: `3 negative tests (signature_policy/retry_plan/dry_run must not reappear) + 2 positive tests (delivery_result/publish_result must remain)`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_DESIGN_20260401.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_VALIDATION_20260401.md`

### Batch 17B Command Log

```text
rg -n "signature_policy|retry_plan|release_draft_dry_run" \
  .github/workflows/evaluation-report.yml tests/unit/

python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 17B Result Log

```text
stale reference check: clean (no output)

pytest: 59 passed in 11.79s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 17A Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) workflow / scripts / tests 中对 eval_reporting_webhook_signature_policy、eval_reporting_webhook_retry_plan、eval_reporting_release_draft_dry_run 及其 helper 名称的精确残留引用检查为零。2) 3 个 helper / consumer 与 3 个定向测试文件已实际删除。3) Checkout for public index generation 的 sparse-checkout 已不再包含这 3 个脚本。4) 本地复跑定向回归：PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile tests/unit/test_evaluation_report_workflow_pages_deploy.py tests/unit/test_post_eval_reporting_webhook_delivery_js.py tests/unit/test_post_eval_reporting_release_draft_publish_js.py 通过；python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py tests/unit/test_post_eval_reporting_webhook_delivery_js.py tests/unit/test_post_eval_reporting_release_draft_publish_js.py -q 为 66 passed in 14.49s。现可进入 Batch 17B。`

### Batch 17B Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) Batch 17B 仅新增 5 个 baseline hardening 守护测试，没有越权做 merge 或新增 artifact。2) stale reference check 仍为 clean。3) 本地复跑 python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q 为 59 passed in 14.55s。4) ledger 中错误的 workflow_graph_proof 数字已修正为当前真实值：deploy-pages 中以 Generate/Append/Upload public/eval reporting 口径统计为 30 个 content steps。`
