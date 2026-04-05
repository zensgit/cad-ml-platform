# Claude Collaboration Batch 20 Validation Ledger

日期：2026-04-03

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 20A Status

- `status`: `complete`
- `implementation_scope`: `Phase 4 release publish-payload merge: publish_result reads draft_payload directly`
- `changed_files`:
  - `scripts/ci/post_eval_reporting_release_draft_publish.js`
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_post_eval_reporting_release_draft_publish_js.py`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `deleted_files`:
  - `scripts/ci/generate_eval_reporting_release_draft_publish_payload.py`
  - `tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py`
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_PAYLOAD_MERGE_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_PAYLOAD_MERGE_VALIDATION_20260403.md`
- `commands_run`: `node --check + pytest + rg stale-reference verification`
- `key_findings`: `publish_payload eliminated, publish_result reads draft_payload directly, 3 workflow steps removed, 2 files deleted, 59 tests pass, zero stale references`
- `handoff_ready`: `yes`

### Batch 20A Evidence

- `publish_payload_merge_proof`: `publish JS now has draftPayloadPath param; loadDraftPayload replaces loadPublishPayload; publish_allowed derived from readiness===ready; github_release_tag generated internally`
- `workflow_cleanup_proof`: `3 generate/append/upload publish_payload steps + 1 sparse-checkout entry removed; publish_result step uses draftPayloadPath`
- `test_update_proof`: `JS tests updated for loadDraftPayload; 7 publish_payload-specific workflow tests removed; ordering test updated; 59 remaining tests pass`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_PAYLOAD_MERGE_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_PAYLOAD_MERGE_VALIDATION_20260403.md`

### Batch 20A Command Log

```text
node --check scripts/ci/post_eval_reporting_release_draft_publish.js

python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py -q

rg -n "release_draft_publish_payload" \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  scripts/ci/post_eval_reporting_release_draft_publish.js
```

### Batch 20A Result Log

```text
node --check: success (no output)

pytest: 59 passed in 12.18s

stale reference check: clean (no output)
```

---

## Batch 20B Status

- `status`: `complete`
- `implementation_scope`: `Phase 4 release publish baseline hardening: 4 regression guard tests for merged/kept surfaces`
- `changed_files`:
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py` (4 new tests added)
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_BASELINE_HARDENING_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_BASELINE_HARDENING_VALIDATION_20260403.md`
- `commands_run`: `pytest`
- `key_findings`: `release chain now dashboard_payload → draft_payload → publish_result (depth 3); 1 negative guard + 2 positive guards + 1 input guard added; 53 tests pass`
- `handoff_ready`: `yes`

### Batch 20B Evidence

- `baseline_hardening_proof`: `design MD documents 3-node release chain post-merge (dashboard_payload → draft_payload → publish_result)`
- `workflow_graph_proof`: `publish_payload absent from workflow steps and sparse-checkout`
- `regression_guard_proof`: `test_merged_publish_payload_not_in_workflow (negative) + test_kept_draft_payload/publish_result_still_present_after_publish_merge (positive) + test_publish_result_reads_draft_payload_not_publish_payload (input guard)`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_BASELINE_HARDENING_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_BASELINE_HARDENING_VALIDATION_20260403.md`

### Batch 20B Command Log

```text
python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 20B Result Log

```text
pytest: 53 passed in 21.75s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 20A Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) post_eval_reporting_release_draft_publish.js 已改为直接读取 draft_payload，参数为 draftPayloadPath，publish_allowed / github_release_tag 逻辑已内联，不再依赖 publish_payload helper。2) generate_eval_reporting_release_draft_publish_payload.py 与对应测试文件已实际删除。3) workflow 中 Generate/Append/Upload publish_payload 共 3 个 step 已移除，Checkout for public index generation 的 sparse-checkout 也已不再包含该脚本。4) 对 .github/workflows/evaluation-report.yml、tests/unit/test_evaluation_report_workflow_pages_deploy.py、scripts/ci/post_eval_reporting_release_draft_publish.js 进行 release_draft_publish_payload 精确残留检查为零。5) 本地复跑：node --check scripts/ci/post_eval_reporting_release_draft_publish.js 通过；python3 -m pytest tests/unit/test_post_eval_reporting_release_draft_publish_js.py tests/unit/test_evaluation_report_workflow_pages_deploy.py tests/unit/test_generate_eval_reporting_release_draft_payload.py -q 为 59 passed in 12.86s。draft_payload / publish_result contract 保持不变。`

### Batch 20B Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) Batch 20B 仅新增 4 个 release publish baseline hardening 守护测试，没有引入新 artifact，也没有触碰 draft_payload -> publish_result deeper merge。2) 本地复跑 python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q 为 53 passed in 12.50s。3) 对 .github/workflows/evaluation-report.yml 与 scripts/ci/post_eval_reporting_release_draft_publish.js 进行 release_draft_publish_payload / publishPayloadPath 精确残留检查为零。4) publish_payload 现在只应以负向守护断言的形式存在于 pages deploy baseline tests；draft_payload 与 publish_result 仍保留，当前 release baseline 与 Batch 20A 合并结果一致。`
