# Claude Collaboration Batch 19 Development Plan

日期：2026-04-03

## 目标概览

本轮开始执行 release 链 merge 的第一步，只收缩 `release_note_snippet` 和 `release_draft_prefill` 两层，不直接碰 `publish_result`。

执行顺序固定：

1. 先把 `release_note_snippet` / `release_draft_prefill` 吸收进 `release_draft_payload`
2. 再建立 merge 后的新 release baseline，防止这两层重新出现

执行原则：

- 本轮允许修改 `.github/workflows/evaluation-report.yml`
- 本轮允许修改 `scripts/ci/generate_eval_reporting_release_draft_payload.py`
- 本轮允许删除 `release_note_snippet` / `release_draft_prefill` 对应 helper / test / workflow step
- 本轮允许更新相关 workflow / pages deploy / helper tests
- 本轮必须补齐对应的 design / validation / ledger 文档
- 本轮禁止提前实现 `publish_payload -> publish_result` merge
- 本轮禁止改变 `publish_payload` / `publish_result` 的输入输出 contract

---

## 当前真实基线

截至 Batch 18 完成后：

- webhook 链第一步 merge 已落地
- `delivery_request` 已直接读取 `dashboard_payload`
- `delivery_result` contract 未变

当前 release 链最明显的中间层仍有：

- `eval_reporting_dashboard_payload`
- `eval_reporting_release_note_snippet`
- `eval_reporting_release_draft_prefill`
- `eval_reporting_release_draft_payload`
- `eval_reporting_release_draft_publish_payload`
- `eval_reporting_release_draft_publish_result`

其中 `release_note_snippet` 和 `release_draft_prefill` 都只是朝 `release_draft_payload` 方向的薄包装层。Batch 16 已明确它们应向 `publish_result` 方向收缩。

因此当前最低风险的下一步是：

- 让 `release_draft_payload` 直接读取 `eval_reporting_dashboard_payload.json`
- 删除 `release_note_snippet` 和 `release_draft_prefill`
- 保持 `publish_payload` 仍然只消费 `release_draft_payload`
- 保持 `publish_result` 仍然只消费 `publish_payload`

本轮不做：

- `release_draft_payload -> publish_result` merge
- `publish_payload -> publish_result` merge
- `publish_result` contract 调整
- 任何新的 public surface 变更

---

## Batch 19A：Snippet / Prefill Merge Into Draft Payload

### 目标

落地 release 链 merge 的第一步：删除 `release_note_snippet` 和 `release_draft_prefill` 两层，让 `release_draft_payload` 直接消费 `dashboard_payload`。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_SNIPPET_PREFILL_MERGE_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_SNIPPET_PREFILL_MERGE_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH19_VALIDATION_LEDGER_20260403.md`

### 必做代码范围

必须完成：

1. 修改 `generate_eval_reporting_release_draft_payload.py`
   - 改为直接读取 `eval_reporting_dashboard_payload.json`
   - 内部直接生成 draft title / body / URLs 所需字段
2. 删除：
   - `generate_eval_reporting_release_note_snippet.py`
   - `generate_eval_reporting_release_draft_prefill.py`
   - 对应 unit tests
3. 更新 workflow：
   - 删除 `Generate/Append/Upload eval reporting release note snippet`
   - 删除 `Generate/Append/Upload eval reporting release draft prefill`
   - 删除对应 sparse-checkout 项
   - 更新 `Generate eval reporting release draft payload` 的输入参数
4. 更新相关测试：
   - `test_generate_eval_reporting_release_draft_payload.py`
   - `test_evaluation_report_workflow_pages_deploy.py`
   - 任何直接断言 `release_note_snippet` / `release_draft_prefill` 存在或顺序的测试

### 设计约束

- 不允许修改 `generate_eval_reporting_release_draft_publish_payload.py` 的输入 contract
- 不允许修改 `post_eval_reporting_release_draft_publish.js` 的输入 contract
- 不允许修改 `eval_reporting_release_draft_publish_result.json` 的 schema / field
- 不允许触碰 webhook 链 merge
- 不允许一次性把 release 链直接并到 `publish_result`

### Batch 19A 验收条件

必须同时满足：

- `release_draft_payload` 已直接读取 `dashboard_payload`
- workflow 不再 materialize `release_note_snippet` / `release_draft_prefill`
- sparse-checkout 不再包含这两个脚本
- `release_note_snippet` / `release_draft_prefill` helper / tests 已删除
- 相关测试已更新并通过
- design / validation / ledger 已回填

---

## Batch 19B：Release Merge Baseline Hardening

### 目标

在 Batch 19A merge 完成后，建立新的 release baseline，防止 `release_note_snippet` / `release_draft_prefill` 重新出现，同时保护 `release_draft_payload` / `publish_payload` / `publish_result` 保持存在。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE3_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH19_VALIDATION_LEDGER_20260403.md`

### 必做内容

至少应覆盖：

1. negative guard：
   - `release_note_snippet` 不得重新出现在 workflow step name / sparse-checkout / helper wiring
   - `release_draft_prefill` 不得重新出现在 workflow step name / sparse-checkout / helper wiring
2. positive guard：
   - `release_draft_payload` 仍存在
   - `publish_payload` 仍存在
   - `publish_result` 仍存在
3. input guard：
   - `release_draft_payload` 使用 `--dashboard-payload-json`
   - 不再使用 `--prefill-json`
4. release merge 后的 workflow / artifact baseline
5. 回归测试命令和结果

### 设计约束

- 不允许新增新 artifact
- 不允许提前实现 `publish_payload -> publish_result` merge
- 只能围绕 Batch 19A merge 后的 release baseline 做 hardening

### Batch 19B 验收条件

必须同时满足：

- merge 后的新 release baseline 有明确测试保护
- 文档写清 `release_note_snippet` / `release_draft_prefill` 已被吸收
- 无越权 publish-result merge / refactor

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_DESIGN_20260403.md`

如有需要，可继续读取：

- `scripts/ci/generate_eval_reporting_release_draft_payload.py`
- `scripts/ci/generate_eval_reporting_release_draft_publish_payload.py`
- `scripts/ci/post_eval_reporting_release_draft_publish.js`
- `tests/unit/` 下与 workflow pages deploy / draft payload / publish payload / publish result 相关的测试

---

## 不做的事

本轮明确不做：

- 不做 `publish_payload -> publish_result` merge
- 不做 `publish_result` contract 调整
- 不新增任何 artifact
- 不改变 dashboard payload / publish payload / publish result 的对外 contract
- 不做大范围 workflow 重排

---

## 额外说明

- 本轮是 release 链 merge 的第一步，不是最终收口。
- 如果 Batch 19A/19B 结论明确，下一轮才进入更深的 release payload / publish result merge。
