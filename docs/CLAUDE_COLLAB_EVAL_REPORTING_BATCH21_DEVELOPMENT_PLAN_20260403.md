# Claude Collaboration Batch 21 Development Plan

日期：2026-04-03

## 目标概览

本轮执行 release 链 merge 的最后一步，只收缩 `draft_payload` 这一层，让 release 链最终落到 `dashboard_payload -> publish_result`。

执行顺序固定：

1. 先把 `draft_payload` 吸收进 `publish_result`
2. 再建立 merge 后的最终 release baseline，防止 `draft_payload` 重新出现

执行原则：

- 本轮允许修改 `.github/workflows/evaluation-report.yml`
- 本轮允许修改 `scripts/ci/post_eval_reporting_release_draft_publish.js`
- 本轮允许删除 `draft_payload` 对应 helper / test / workflow step
- 本轮允许更新相关 workflow / pages deploy / publish-result tests
- 本轮必须补齐对应的 design / validation / ledger 文档
- 本轮禁止提前实现 workflow consolidate
- 本轮禁止改变 `publish_result` 的输出 schema / field contract

---

## 当前真实基线

截至 Batch 20 完成后，release 链当前形态为：

- `eval_reporting_dashboard_payload`
- `eval_reporting_release_draft_payload`
- `eval_reporting_release_draft_publish_result`

当前链路为：

```text
dashboard_payload -> draft_payload -> publish_result
```

其中 `draft_payload` 仍然只是朝 `publish_result` 的薄包装层，主要承载：

- `draft_title`
- `draft_body_markdown`
- `release_readiness`
- public report / landing URLs
- 从 `dashboard_payload` 派生但未改变对外 owner 的内容

因此当前最低风险的下一步是：

- 让 `publish_result` 直接读取 `eval_reporting_dashboard_payload.json`
- 把 `draft_payload` 内部负责的 draft title / body / URL 组装逻辑内联到 `post_eval_reporting_release_draft_publish.js`
- 删除 `draft_payload` 本身
- 保持 `publish_result` 输出字段保持不变

本轮不做：

- `dashboard_payload` 删除
- `publish_result` schema 调整
- webhook 链任何改动
- owner / public surface 变更
- workflow consolidate / upload-summary 合并

---

## Batch 21A：Draft Payload Merge Into Publish Result

### 目标

落地 release 链 merge 的最后一步：删除 `draft_payload` 中间层，让 `publish_result` 直接消费 `dashboard_payload`。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md`

### 必做代码范围

必须完成：

1. 修改 `post_eval_reporting_release_draft_publish.js`
   - 改为直接读取 `eval_reporting_dashboard_payload.json`
   - 在该 consumer 内部完成现有 `draft_payload` 承担的 draft title / body / URL 组装逻辑
2. 删除：
   - `generate_eval_reporting_release_draft_payload.py`
   - `tests/unit/test_generate_eval_reporting_release_draft_payload.py`
3. 更新 workflow：
   - 删除 `Generate/Append/Upload eval reporting release draft payload`
   - 删除对应 sparse-checkout 项
   - 更新 `Generate eval reporting release draft publish result` 的输入参数
4. 更新相关测试：
   - `tests/unit/test_post_eval_reporting_release_draft_publish_js.py`
   - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
   - 任何直接断言 `draft_payload` 存在或顺序的测试

### 设计约束

- 不允许修改 `eval_reporting_dashboard_payload.json` 的输入 contract
- 不允许修改 `eval_reporting_release_draft_publish_result.json` 的输出 schema / field
- 不允许删除 `dashboard_payload` 或 `publish_result`
- 不允许触碰 webhook 链 merge
- 不允许提前做 consolidate

### Batch 21A 验收条件

必须同时满足：

- `publish_result` 已直接读取 `dashboard_payload`
- workflow 不再 materialize `draft_payload`
- sparse-checkout 不再包含 `generate_eval_reporting_release_draft_payload.py`
- `draft_payload` helper / tests 已删除
- 相关测试已更新并通过
- design / validation / ledger 已回填

---

## Batch 21B：Final Release Baseline Hardening

### 目标

在 Batch 21A merge 完成后，建立最终 release baseline，防止 `draft_payload` 重新出现，同时保护 `publish_result` 保持存在。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md`

### 必做内容

至少应覆盖：

1. negative guard：
   - `draft_payload` 不得重新出现在 workflow step name / sparse-checkout / helper wiring
2. positive guard：
   - `publish_result` 仍存在
3. input guard：
   - `publish_result` 使用 `dashboardPayloadPath` / `eval_reporting_dashboard_payload.json`
   - 不再使用 `draftPayloadPath` / `release_draft_payload.json`
4. final release merge 后的 workflow / artifact baseline
5. 回归测试命令和结果

### 设计约束

- 不允许新增新 artifact
- 不允许提前实现 workflow consolidate
- 只能围绕 Batch 21A merge 后的 final release baseline 做 hardening

### Batch 21B 验收条件

必须同时满足：

- merge 后的最终 release baseline 有明确测试保护
- 文档写清 `draft_payload` 已被吸收
- 无越权 consolidate / refactor

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH20_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_PAYLOAD_MERGE_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE4_RELEASE_PUBLISH_BASELINE_HARDENING_DESIGN_20260403.md`

如有需要，可继续读取：

- `scripts/ci/generate_eval_reporting_dashboard_payload.py`
- `scripts/ci/generate_eval_reporting_release_draft_payload.py`
- `scripts/ci/post_eval_reporting_release_draft_publish.js`
- `tests/unit/` 下与 workflow pages deploy / dashboard payload / publish result 相关的测试

---

## 不做的事

本轮明确不做：

- 不做 workflow consolidate
- 不做 `publish_result` contract 调整
- 不新增任何 artifact
- 不改变 dashboard payload / publish result 的对外 contract
- 不做大范围 workflow 重排

---

## 额外说明

- 本轮是 release 链 merge 的最终收口。
- 如果 Batch 21A/21B 结论明确，下一轮才进入 workflow consolidate。
