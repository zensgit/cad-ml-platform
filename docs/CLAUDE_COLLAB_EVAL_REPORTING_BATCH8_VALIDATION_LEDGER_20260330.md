# Claude Collaboration Batch 8 Validation Ledger

日期：2026-03-30

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 8A Status

- `status`: `complete`
- `implementation_scope`: `Pages-ready root thin assembler + workflow artifact + deploy-pages alignment`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
- `new_files`:
  - `scripts/ci/assemble_eval_reporting_pages_root.py`
  - `tests/unit/test_assemble_eval_reporting_pages_root.py`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
  - `docs/DEDUP_EVAL_REPORTING_PAGES_ROOT_ARTIFACT_ALIGNMENT_DESIGN_20260330.md`
  - `docs/DEDUP_EVAL_REPORTING_PAGES_ROOT_ARTIFACT_ALIGNMENT_VALIDATION_20260330.md`
- `tests_run`: `31`
- `test_results`: `31 passed in 18.61s`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 8A Evidence

- `pages_root_artifact_proof`: `test_pages_ready_artifact_upload_exists — verifies eval-reporting-pages- in artifact name and eval_pages in path`
- `landing_as_public_root_proof`: `test_assemble_copies_landing_page_as_root_index — verifies pages_root/index.html contains "landing"`
- `deploy_pages_download_proof`: `test_deploy_pages_downloads_pages_artifact — verifies eval-reporting-pages- artifact with path ./public; test_deploy_pages_no_longer_downloads_old_report_artifact — verifies old step removed`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PAGES_ROOT_ARTIFACT_ALIGNMENT_DESIGN_20260330.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PAGES_ROOT_ARTIFACT_ALIGNMENT_VALIDATION_20260330.md`

### Batch 8A Command Log

```text
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/assemble_eval_reporting_pages_root.py \
  tests/unit/test_assemble_eval_reporting_pages_root.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_assemble_eval_reporting_pages_root.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py -q
```

### Batch 8A Result Log

```text
py_compile: success (no output)

pytest: 31 passed in 18.61s
```

---

## Batch 8B Status

- `status`: `complete`
- `implementation_scope`: `public discovery surface: public index helper + deploy-pages job summary/upload/generation steps`
- `changed_files`:
  - `.github/workflows/evaluation-report.yml`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- `new_files`:
  - `scripts/ci/generate_eval_reporting_public_index.py`
  - `tests/unit/test_generate_eval_reporting_public_index.py`
  - `docs/DEDUP_EVAL_REPORTING_PUBLIC_DISCOVERY_SURFACE_ALIGNMENT_DESIGN_20260330.md`
  - `docs/DEDUP_EVAL_REPORTING_PUBLIC_DISCOVERY_SURFACE_ALIGNMENT_VALIDATION_20260330.md`
- `tests_run`: `15 (batch 8B) + 55 (full regression)`
- `test_results`: `15 passed in 5.97s (batch 8B), 55 passed in 4.99s (full regression)`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 8B Evidence

- `public_index_artifact_proof`: `test_deploy_pages_has_public_index_upload_step — verifies always-run upload step with eval_reporting_public_index in path`
- `public_url_summary_proof`: `test_build_public_index_with_page_url — verifies landing/static/interactive URLs formed from page_url`
- `deploy_pages_summary_proof`: `test_deploy_pages_has_job_summary_append_step — verifies always-run GITHUB_STEP_SUMMARY append with eval_reporting_public_index.md`
- `design_md`: `docs/DEDUP_EVAL_REPORTING_PUBLIC_DISCOVERY_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `validation_md`: `docs/DEDUP_EVAL_REPORTING_PUBLIC_DISCOVERY_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

### Batch 8B Command Log

```text
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_public_index.py \
  tests/unit/test_generate_eval_reporting_public_index.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_public_index.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_assemble_eval_reporting_pages_root.py \
  tests/unit/test_generate_eval_reporting_public_index.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_generate_eval_reporting_landing_page.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_summarize_eval_reporting_stack_status.py -q
```

### Batch 8B Result Log

```text
py_compile: success (no output, 2 files compile cleanly)

pytest batch 8B: 15 passed in 5.97s

pytest full regression: 55 passed in 4.99s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 8A Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。实际复跑 py_compile 通过；Batch 8A 定向回归为 31 passed in 6.26s。assemble_eval_reporting_pages_root.py 仅做已有 landing/static/interative/canonical JSON 的复制组装，没有越权成新的 owner；workflow evaluate job 已新增 Assemble Pages-ready root 与 Upload Pages-ready artifact，两者都位于 Fail workflow on refresh failure 之前；deploy-pages job 已改为下载 eval-reporting-pages artifact 而不是旧的 static report artifact。当前 Pages 发布根已明确由 landing page 驱动，Batch 8A 合同已满足。`

### Batch 8B Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。实际复跑 py_compile 通过；Batch 8B 定向回归为 15 passed in 5.41s；合并回归为 55 passed in 6.99s。generate_eval_reporting_public_index.py 只消费现有 index / stack summary / Pages deployment URL，生成 public discovery JSON+MD，没有越权成新的 owner；deploy-pages job 已新增 always-run 的 checkout、artifact 下载、public index 生成、job summary append、public index artifact upload 五个步骤，且均围绕现有 canonical artifacts 展开。当前 landing/static/interactive 的公开 URL 与 stack summary 状态都已形成独立 public discovery surface，Batch 8B 合同已满足。`
