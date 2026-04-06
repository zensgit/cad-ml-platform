# Claude Collaboration Batch 12 Execution Runbook

日期：2026-04-01

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH12_DEVELOPMENT_PLAN_20260401.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH12_EXECUTION_RUNBOOK_20260401.md 执行。
现在只允许做 Batch 12A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH12_VALIDATION_LEDGER_20260401.md，
并明确停止，等待人工验证。
不允许提前做 Batch 12B。
```

---

## 总执行规则

1. 必须先做 Batch 12A
2. Batch 12A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 12B
4. 不允许修改 Development Plan 中写死的接口和范围
5. 不允许新建新的 summary owner / metrics owner
6. 不允许回退或覆盖其他批次已经完成的工作
7. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 12A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `scripts/ci/generate_eval_reporting_release_draft_prefill.py`
   - `scripts/ci/generate_eval_reporting_release_note_snippet.py`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH11_VALIDATION_LEDGER_20260331.md`
2. 实现 `scripts/ci/generate_eval_reporting_release_draft_payload.py`
3. 修改 workflow `deploy-pages` job
4. 补充/更新测试
5. 跑验证命令
6. 补齐该批 design/validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 12A 设计约束

`generate_eval_reporting_release_draft_payload.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_draft_prefill.json`
- 归一化为 GitHub release draft 友好的最小 payload

它不允许负责：

- 再次读取 release-note snippet / dashboard payload / release summary / public index / stack summary
- 重新生成 release draft prefill
- 直接创建或编辑 GitHub release draft
- 修改 notify / PR comment / status check owner

### Batch 12A artifact 合同

建议默认输出：

- `reports/ci/eval_reporting_release_draft_payload.json`
- `reports/ci/eval_reporting_release_draft_payload.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_payload"`
- `generated_at`
- `release_readiness`
- `draft_title`
- `draft_body_markdown`
- `repository_url`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `source_prefill_surface_kind`

其中：

- `draft_title` 只能由现有 prefill 的 `draft_title` / `release_readiness` 薄推导得到
- `draft_body_markdown` 必须可以直接用于 GitHub release draft body 或人工 handoff

Markdown 至少要包含：

- `Eval Reporting`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

### Batch 12A workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft payload`
- always-run `Append eval reporting release draft payload to job summary`
- always-run `Upload eval reporting release draft payload`

这些步骤必须位于 release draft prefill 之后。

### Batch 12A 验收条件

必须同时满足：

- release-draft payload 只消费 release-draft prefill artifact
- payload artifact 能稳定 materialize
- job summary 中出现 release draft 友好的最小摘要
- artifact 已作为独立 surface 上传

### Batch 12A 必跑命令

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 12A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH12_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PAYLOAD_SURFACE_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PAYLOAD_SURFACE_ALIGNMENT_VALIDATION_20260401.md`

---

## Batch 12B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 12A ledger 和 validation MD
2. 实现 `scripts/ci/post_eval_reporting_release_draft_dry_run.js` 或等价 thin consumer
3. 修改 workflow `deploy-pages` job
4. 新增/更新测试
5. 跑验证命令
6. 补齐该批 design/validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 12B 设计约束

`post_eval_reporting_release_draft_dry_run.js` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_draft_payload.json`
- 产出 gated dry-run / optional publish 的最小 GitHub release draft 计划

它不允许负责：

- 再次读取 release draft prefill / release-note snippet / dashboard payload / release summary / public index / stack summary
- 重新生成 release draft payload
- 在默认模式下真正创建 GitHub release
- 修改 notify / PR comment / status check owner

### Batch 12B contract

建议默认输出：

- `reports/ci/eval_reporting_release_draft_dry_run.json`
- `reports/ci/eval_reporting_release_draft_dry_run.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_dry_run"`
- `generated_at`
- `release_readiness`
- `draft_title`
- `draft_body_markdown`
- `publish_enabled`
- `publish_attempted`
- `publish_allowed`
- `github_release_tag`

其中：

- 默认必须是 dry-run，不得在未授权上下文中自动发布
- 只有在显式 gated 条件满足时才允许 optional publish
- 无权限场景必须 fail-soft，不得破坏主评估链

Markdown 至少要包含：

- `Dry Run`
- `Publish Enabled`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

### Batch 12B workflow 约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft dry run`
- always-run `Append eval reporting release draft dry run to job summary`
- always-run `Upload eval reporting release draft dry run`

这些步骤必须位于 release draft payload 之后，且默认不做实际发布。

### Batch 12B 验收条件

必须同时满足：

- dry-run surface 只消费 release draft payload artifact
- workflow 中存在独立的 dry-run / optional publish steps
- default 路径不真正发布 GitHub release
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

### Batch 12B 必跑命令

```bash
node --check scripts/ci/post_eval_reporting_release_draft_dry_run.js

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 12B 合并回归

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_draft_prefill.py \
  tests/unit/test_generate_eval_reporting_release_draft_payload.py \
  tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py \
  tests/unit/test_generate_eval_reporting_webhook_export.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 12B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH12_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_DRY_RUN_SURFACE_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_DRY_RUN_SURFACE_ALIGNMENT_VALIDATION_20260401.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 本批变更摘要
2. 修改文件列表
3. 新增文件列表
4. design MD 路径
5. validation MD 路径
6. 实际命令
7. 实际结果
8. 未解决风险
9. 一句明确结论：
   - `Batch 12A complete, stopped for validation`
   - 或 `Batch 12B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 12A accepted`

Claude 才能继续 Batch 12B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
