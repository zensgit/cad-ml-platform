# Claude Collaboration Batch 18 Validation Ledger

日期：2026-04-03

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 18A Status

- `status`: `complete`
- `implementation_scope`: `Phase 2 webhook export merge: delivery_request now reads dashboard_payload directly`
- `changed_files`:
  - `scripts/ci/generate_eval_reporting_webhook_delivery_request.py`
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_generate_eval_reporting_webhook_delivery_request.py`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `deleted_files`:
  - `scripts/ci/generate_eval_reporting_webhook_export.py`
  - `tests/unit/test_generate_eval_reporting_webhook_export.py`
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_VALIDATION_20260403.md`
- `commands_run`: `py_compile + pytest + rg stale-reference verification`
- `key_findings`: `webhook_export eliminated, delivery_request reads dashboard_payload directly, 3 workflow steps removed, 2 files deleted, 64 tests pass, zero stale references`
- `handoff_ready`: `yes`

### Batch 18A Evidence

- `webhook_export_merge_proof`: `delivery_request helper now has --dashboard-payload-json param; source_dashboard_payload_surface_kind field replaces source_webhook_export_surface_kind`
- `workflow_cleanup_proof`: `3 generate/append/upload webhook_export steps + 1 sparse-checkout entry removed; delivery_request step uses --dashboard-payload-json`
- `test_update_proof`: `delivery_request tests rewritten for dashboard_payload input; 6 webhook_export-specific tests deleted; ordering test updated; 64 remaining tests pass`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_VALIDATION_20260403.md`

### Batch 18A Command Log

```text
python3 -m py_compile \
  scripts/ci/generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py

python3 -m pytest \
  tests/unit/test_generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_post_eval_reporting_webhook_delivery_js.py -q

rg -n "webhook_export" .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  scripts/ci/generate_eval_reporting_webhook_delivery_request.py
```

### Batch 18A Result Log

```text
py_compile: success (no output)

pytest: 64 passed in 9.39s

stale reference check: only negative guard assertion found (correct)
```

---

## Batch 18B Status

- `status`: `complete`
- `implementation_scope`: `Phase 2 webhook baseline hardening: 4 regression guard tests for merged/kept surfaces`
- `changed_files`:
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py` (4 new tests added)
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_VALIDATION_20260403.md`
- `commands_run`: `pytest`
- `key_findings`: `webhook chain now dashboard_payload → delivery_request → delivery_result (depth 3); 1 negative guard + 2 positive guards + 1 input guard added; 58 tests pass`
- `handoff_ready`: `yes`

### Batch 18B Evidence

- `baseline_hardening_proof`: `design MD documents 3-node webhook chain post-merge`
- `workflow_graph_proof`: `rg stale-reference check: only negative guard assertions reference webhook_export`
- `regression_guard_proof`: `test_merged_webhook_export_not_in_workflow (negative) + test_kept_delivery_request_still_present_after_merge + test_kept_delivery_result_still_present_after_merge (positive) + test_delivery_request_reads_dashboard_payload_not_webhook_export (input guard)`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_VALIDATION_20260403.md`

### Batch 18B Command Log

```text
python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 18B Result Log

```text
pytest: 58 passed in 12.45s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 18A Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) generate_eval_reporting_webhook_delivery_request.py 已改为直接读取 dashboard_payload，并新增 source_dashboard_payload_surface_kind 字段。2) webhook_export helper 与对应测试文件已实际删除。3) workflow 中 Generate/Append/Upload webhook export 3 个 step 已移除，Checkout for public index generation 的 sparse-checkout 也已不再包含 generate_eval_reporting_webhook_export.py。4) 对 .github/workflows/evaluation-report.yml、tests/unit/test_evaluation_report_workflow_pages_deploy.py、scripts/ci/generate_eval_reporting_webhook_delivery_request.py 进行 webhook_export 精确残留检查为零。5) 本地复跑：PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile scripts/ci/generate_eval_reporting_webhook_delivery_request.py tests/unit/test_generate_eval_reporting_webhook_delivery_request.py tests/unit/test_evaluation_report_workflow_pages_deploy.py 通过；python3 -m pytest tests/unit/test_generate_eval_reporting_webhook_delivery_request.py tests/unit/test_evaluation_report_workflow_pages_deploy.py tests/unit/test_post_eval_reporting_webhook_delivery_js.py -q 为 64 passed in 13.14s。Release chain 未触碰，delivery_result contract 保持不变。`

### Batch 18B Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) Batch 18B 仅新增 4 个 webhook baseline hardening 守护测试，没有越权触碰 release 链 merge 或新增 artifact。2) 本地复跑 python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q 为 58 passed in 15.31s。3) 额外残留检查显示 webhook_export 仅出现在负向守护测试断言中，不再出现在 workflow step name 或 sparse-checkout。4) delivery_request 仍使用 --dashboard-payload-json，delivery_result 仍保留，当前 webhook baseline 与 Batch 18A 合并结果一致。`
