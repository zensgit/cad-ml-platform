# Claude Collaboration Batch 21 Validation Ledger

日期：2026-04-03

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 21A Status

- `status`: `complete`
- `implementation_scope`: `Phase 5 final release merge: publish_result reads dashboard_payload directly`
- `changed_files`:
  - `scripts/ci/post_eval_reporting_release_draft_publish.js`
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_post_eval_reporting_release_draft_publish_js.py`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `deleted_files`:
  - `scripts/ci/generate_eval_reporting_release_draft_payload.py`
  - `tests/unit/test_generate_eval_reporting_release_draft_payload.py`
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_VALIDATION_20260403.md`
- `commands_run`: `node --check + pytest + rg stale-reference verification`
- `key_findings`: `draft_payload eliminated, publish_result reads dashboard_payload directly, release chain now dashboard_payload → publish_result (depth 2), 3 workflow steps removed, 2 files deleted, 55 tests pass, zero stale references`
- `handoff_ready`: `yes`

### Batch 21A Evidence

- `draft_payload_merge_proof`: `publish JS now has dashboardPayloadPath param; loadDashboardPayload replaces loadDraftPayload; draft_title/body derived internally from dashboard_payload fields`
- `workflow_cleanup_proof`: `3 generate/append/upload draft_payload steps + 1 sparse-checkout entry removed; publish_result step uses dashboardPayloadPath`
- `test_update_proof`: `JS tests updated for loadDashboardPayload; 14 draft_payload-specific tests removed; ordering tests updated; 55 remaining tests pass`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_VALIDATION_20260403.md`

### Batch 21A Command Log

```text
node --check scripts/ci/post_eval_reporting_release_draft_publish.js

python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_generate_eval_reporting_dashboard_payload.py -q

rg -n "release_draft_payload|draftPayloadPath" \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  scripts/ci/post_eval_reporting_release_draft_publish.js
```

### Batch 21A Result Log

```text
node --check: success (no output)

pytest: 55 passed in 10.20s

stale reference check: clean (no output)
```

---

## Batch 21B Status

- `status`: `complete`
- `implementation_scope`: `Phase 5 final release baseline hardening: 3 regression guard tests for merged/kept surfaces`
- `changed_files`:
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py` (3 new tests added)
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`
- `commands_run`: `pytest`
- `key_findings`: `final release chain: dashboard_payload → publish_result (depth 2); 1 negative guard + 1 positive guard + 1 input guard added; 47 tests pass`
- `handoff_ready`: `yes`

### Batch 21B Evidence

- `baseline_hardening_proof`: `design MD documents final 2-node release chain`
- `workflow_graph_proof`: `draft_payload absent from workflow steps and sparse-checkout`
- `regression_guard_proof`: `test_merged_draft_payload_not_in_workflow (negative) + test_kept_publish_result_still_present_after_final_merge (positive) + test_publish_result_reads_dashboard_not_draft_payload (input guard)`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`

### Batch 21B Command Log

```text
python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 21B Result Log

```text
pytest: 47 passed in 10.32s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 21A Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) post_eval_reporting_release_draft_publish.js 已改为直接读取 dashboard_payload，参数为 dashboardPayloadPath，draft title / body / URL 组装逻辑已内联，不再依赖 draft_payload helper。2) generate_eval_reporting_release_draft_payload.py 与对应测试文件已实际删除。3) workflow 中 Generate/Append/Upload draft_payload 共 3 个 step 已移除，Checkout for public index generation 的 sparse-checkout 也已不再包含该脚本。4) 对 .github/workflows/evaluation-report.yml、tests/unit/test_evaluation_report_workflow_pages_deploy.py、scripts/ci/post_eval_reporting_release_draft_publish.js 进行 release_draft_payload / draftPayloadPath 精确残留检查为零。5) 本地复跑：node --check scripts/ci/post_eval_reporting_release_draft_publish.js 通过；python3 -m pytest tests/unit/test_post_eval_reporting_release_draft_publish_js.py tests/unit/test_evaluation_report_workflow_pages_deploy.py tests/unit/test_generate_eval_reporting_dashboard_payload.py -q 为 55 passed in 10.56s。publish_result 输出 contract 保持不变。`

### Batch 21B Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) Batch 21B 仅新增 3 个 final release baseline hardening 守护测试，没有引入新 artifact，也没有触碰 workflow consolidate。2) 本地复跑 python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q 为 47 passed in 10.84s。3) 对 .github/workflows/evaluation-report.yml 与 scripts/ci/post_eval_reporting_release_draft_publish.js 进行 release_draft_payload / draftPayloadPath 精确残留检查为零。4) draft_payload 现在只应以负向守护断言的形式存在于 pages deploy baseline tests；publish_result 仍保留，当前 final release baseline 与 Batch 21A 合并结果一致。`
