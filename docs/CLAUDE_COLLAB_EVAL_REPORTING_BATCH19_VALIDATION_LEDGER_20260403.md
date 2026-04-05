# Claude Collaboration Batch 19 Validation Ledger

日期：2026-04-03

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 19A Status

- `status`: `complete`
- `implementation_scope`: `Phase 3 release snippet/prefill merge: draft_payload now reads dashboard_payload directly`
- `changed_files`:
  - `scripts/ci/generate_eval_reporting_release_draft_payload.py`
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_generate_eval_reporting_release_draft_payload.py`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `deleted_files`:
  - `scripts/ci/generate_eval_reporting_release_note_snippet.py`
  - `scripts/ci/generate_eval_reporting_release_draft_prefill.py`
  - `tests/unit/test_generate_eval_reporting_release_note_snippet.py`
  - `tests/unit/test_generate_eval_reporting_release_draft_prefill.py`
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_SNIPPET_PREFILL_MERGE_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_SNIPPET_PREFILL_MERGE_VALIDATION_20260403.md`
- `commands_run`: `py_compile + pytest + rg stale-reference verification`
- `key_findings`: `snippet + prefill eliminated, draft_payload reads dashboard_payload directly, 6 workflow steps removed, 4 files deleted, 64 tests pass, zero stale references`
- `handoff_ready`: `yes`

### Batch 19A Evidence

- `release_merge_proof`: `draft_payload helper now has --dashboard-payload-json param; generates draft_title/body internally; source_dashboard_payload_surface_kind replaces source_prefill_surface_kind`
- `workflow_cleanup_proof`: `6 generate/append/upload steps + 2 sparse-checkout entries removed; draft_payload step uses --dashboard-payload-json`
- `test_update_proof`: `draft_payload tests rewritten for dashboard_payload input; 12 snippet/prefill-specific tests deleted; ordering tests updated; 64 remaining tests pass`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_SNIPPET_PREFILL_MERGE_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_SNIPPET_PREFILL_MERGE_VALIDATION_20260403.md`

### Batch 19A Command Log

```text
python3 -m py_compile \
  scripts/ci/generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py

python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py -q

rg -n "release_note_snippet|release_draft_prefill" \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  scripts/ci/generate_eval_reporting_release_draft_payload.py
```

### Batch 19A Result Log

```text
py_compile: success (no output)

pytest: 64 passed in 13.95s

stale reference check: clean (no output)
```

---

## Batch 19B Status

- `status`: `complete`
- `implementation_scope`: `Phase 3 release baseline hardening: 6 regression guard tests for merged/kept surfaces`
- `changed_files`:
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py` (6 new tests added)
- `new_docs`:
  - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
  - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`
- `commands_run`: `pytest`
- `key_findings`: `release chain now dashboard_payload → draft_payload → publish_payload → publish_result (depth 4); 2 negative guards + 3 positive guards + 1 input guard added; 55 tests pass`
- `handoff_ready`: `yes`

### Batch 19B Evidence

- `baseline_hardening_proof`: `design MD documents 4-node release chain post-merge`
- `workflow_graph_proof`: `snippet/prefill absent from workflow steps and sparse-checkout`
- `regression_guard_proof`: `test_merged_release_note_snippet_not_in_workflow + test_merged_release_draft_prefill_not_in_workflow (negative) + test_kept_release_draft_payload/publish_payload/publish_result_still_present_after_merge (positive) + test_release_draft_payload_reads_dashboard_not_prefill (input guard)`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`

### Batch 19B Command Log

```text
python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 19B Result Log

```text
pytest: 55 passed in 11.43s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 19A Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) generate_eval_reporting_release_draft_payload.py 已改为直接读取 dashboard_payload，并新增 source_dashboard_payload_surface_kind 字段；draft_title / draft_body_markdown 在该 helper 内部生成。2) release_note_snippet / release_draft_prefill 两个 helper 与对应测试文件已实际删除。3) workflow 中 Generate/Append/Upload snippet + prefill 共 6 个 step 已移除，Checkout for public index generation 的 sparse-checkout 也已不再包含这两个脚本。4) 对 .github/workflows/evaluation-report.yml、tests/unit/test_evaluation_report_workflow_pages_deploy.py、scripts/ci/generate_eval_reporting_release_draft_payload.py 进行 release_note_snippet / release_draft_prefill 精确残留检查为零。5) 本地复跑：PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile scripts/ci/generate_eval_reporting_release_draft_payload.py tests/unit/test_generate_eval_reporting_release_draft_payload.py tests/unit/test_evaluation_report_workflow_pages_deploy.py 通过；python3 -m pytest tests/unit/test_generate_eval_reporting_release_draft_payload.py tests/unit/test_evaluation_report_workflow_pages_deploy.py tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py tests/unit/test_post_eval_reporting_release_draft_publish_js.py -q 为 64 passed in 13.77s。publish_payload / publish_result contract 保持不变。`

### Batch 19B Verifier Decision

- `decision`: `accepted`
- `notes`: `已人工复验通过。1) Batch 19B 仅新增 6 个 release baseline hardening 守护测试，没有引入新 artifact，也没有触碰 publish_payload / publish_result deeper merge。2) 对 .github/workflows/evaluation-report.yml、tests/unit/test_evaluation_report_workflow_pages_deploy.py、scripts/ci/generate_eval_reporting_release_draft_payload.py 进行 release_note_snippet / release_draft_prefill 精确残留检查为零。3) Checkout for public index generation 的 sparse-checkout 也已不再包含已删除的 snippet / prefill 脚本。4) 本地复跑 python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q 为 55 passed in 27.49s。当前 release chain baseline 为 dashboard_payload -> draft_payload -> publish_payload -> publish_result。`
