# Claude Collaboration Batch 7 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `eval reporting` 的上层消费面收口，顺序固定：

1. 先让 PR comment 消费 top-level `eval reporting stack summary`
2. 再让通知脚本消费同一份 stack summary

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/summarize_eval_reporting_stack_status.py`
  - `scripts/ci/generate_eval_reporting_index.py`
  - `scripts/generate_eval_reporting_landing_page.py`
- 只能新增 thin comment / notification helper
- 不允许新建新的 summary owner / metrics owner
- 不允许重算 health / bundle / index / trend
- `PR comment` 和 `notification` 只能消费现有 stack summary / index / landing artifacts
- 改动必须是 additive，不允许重写现有 PR comment 和 notification 主体合同

---

## 当前真实基线

截至当前仓库状态：

- top-level `eval reporting` stack 已稳定产出：
  - `reports/eval_history/eval_reporting_bundle.json`
  - `reports/eval_history/eval_reporting_bundle_health_report.json`
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/eval_history/index.html`
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/ci/eval_reporting_stack_summary.md`
- `.github/workflows/evaluation-report.yml` 已：
  - 默认调用 `scripts/ci/refresh_eval_reporting_stack.py`
  - 生成 landing page
  - 生成 stack summary
  - 上传 stack artifact 和 landing page artifact
- `scripts/ci/comment_evaluation_report_pr.js` 当前仍主要消费：
  - combined / vision / ocr 分数
  - graph2d / hybrid blind / workflow guardrail 等摘要
  - 尚未消费 `eval_reporting_stack_summary.json`
- `scripts/notify_eval_results.py` 当前仍主要消费：
  - 最新 `*_combined.json`
  - recent trend
  - `--report-url`
  - 尚未消费 `eval_reporting_stack_summary.json`
- 当前缺口：
  - PR comment 还没有统一展示 top-level eval reporting stack 状态
  - Slack / email / GitHub channel 通知还没有消费 top-level stack summary
  - 人类第一时间看到的协作面还没有对齐 landing page / stack summary

---

## Batch 7A：PR Comment Consumes Eval Reporting Stack Summary

### 目标

让 `scripts/ci/comment_evaluation_report_pr.js` 增量展示一个最小的 `Eval Reporting Stack` 区块。

### 必做改动

1. 修改 `scripts/ci/comment_evaluation_report_pr.js`
2. 视需要新增极薄 JS helper
3. 修改 `.github/workflows/evaluation-report.yml` 的 PR comment step env
4. 更新/新增测试

### 设计约束

#### 允许新增的 thin helper

如有必要，推荐新增：

- `scripts/ci/eval_reporting_stack_comment_helpers.js`

它只能负责：

- 读取：
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/eval_history/eval_reporting_index.json`
- 归一化 comment-friendly view model

它不允许负责：

- materialize bundle / health / index / summary
- 渲染完整 comment body
- 重算 metrics / trend / health
- 新 owner schema

#### comment 输入

comment 必须优先消费：

- `EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT`
- `EVAL_REPORTING_INDEX_JSON_FOR_COMMENT`

如果这两个 env 未提供或文件缺失，comment 仍必须成功生成，只是不显示 `Eval Reporting Stack` 区块，或显示明确的 unavailable/missing 状态。

#### comment 区块内容

只允许新增一个最小 additive 区块，例如：

- section title：`Eval Reporting Stack`
- overall status / light
- missing / stale / mismatch counts
- landing page path
- static report path
- interactive report path

允许：

- inline code path
- 1 个极小 markdown table
- 1 行 summary

不允许：

- 复制 landing page 内容
- 重渲染 health 明细表
- 再加新的 leaderboard
- 改写已有 evaluation overview / graph2d / hybrid blind 区块语义

#### workflow env 约束

`.github/workflows/evaluation-report.yml` 的 PR comment step 必须显式传入：

- `EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT=reports/ci/eval_reporting_stack_summary.json`
- `EVAL_REPORTING_INDEX_JSON_FOR_COMMENT=reports/eval_history/eval_reporting_index.json`

允许直接传默认路径，不要求再引入新的 workflow output。

### Batch 7A 验收条件

必须同时满足：

- PR comment body 出现 `Eval Reporting Stack` 区块
- comment 只消费 stack summary / index，不重算 health
- workflow 已向 comment step 传入 stack summary / index 路径
- summary/index 缺失时 comment 仍能成功生成

---

## Batch 7B：Notifications Consume The Same Stack Summary

### 目标

让 `scripts/notify_eval_results.py` 复用同一份 top-level stack summary，在 Slack / email / GitHub channel 中展示最小 stack 状态。

### 必做改动

1. 修改 `scripts/notify_eval_results.py`
2. 如有必要，新增一个极薄 Python helper
3. 修改 `.github/workflows/evaluation-report.yml` 的 notify step env 或 CLI args
4. 新增/更新测试

### 设计约束

#### 允许新增的 thin helper

如有必要，推荐新增：

- `scripts/eval_reporting_stack_notification_helpers.py`

它只能负责：

- 读取：
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/eval_history/eval_reporting_index.json`
- 归一化 notification-friendly summary line / fields

它不允许负责：

- 重算 combined/trend
- 重算 health
- 发送通知
- 新建 notification 专用 schema

#### notify 输入

通知脚本可以新增 CLI 参数，推荐：

- `--stack-summary-json`
- `--index-json`

也可以新增 env fallback，但主合同必须清晰稳定。

notify 仍必须支持当前只传 `--report-url` 的 standalone 调用；stack summary 只是 additive surface。

#### notify 输出

Slack / email / GitHub issue channel 只允许最小增量：

- overall stack status
- missing / stale / mismatch counts
- primary link：
  - 优先 `--report-url`
  - 如适用可补 landing page / static / interactive 的 path 信息

不允许：

- 新建第二套 health 计算
- 复制 landing page 明细表
- 修改 threshold-breach 判定逻辑
- 把 top-level stack status 变成新的阻断条件

#### workflow 传参

`.github/workflows/evaluation-report.yml` 的 notify step 必须传入：

- `--stack-summary-json reports/ci/eval_reporting_stack_summary.json`
- `--index-json reports/eval_history/eval_reporting_index.json`

或提供等价 env，但必须是对现有 stack summary/index 的消费，不允许直接读 raw bundle/health/index 分散文件。

### Batch 7B 验收条件

必须同时满足：

- `notify_eval_results.py` 能消费 stack summary / index
- Slack / email payload 中能看到最小 stack status surface
- 现有 `--report-url` 调用和 threshold-breach 行为保持兼容
- workflow notify step 已接入 stack summary / index 传参

---

## 必须新增或更新的测试

### Batch 7A

- 更新：
  - `tests/unit/test_comment_evaluation_report_pr_js.py`
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
- 如新增 helper，再新增：
  - `tests/unit/test_eval_reporting_stack_comment_helpers_js.py`

### Batch 7B

- 新增：
  - `tests/unit/test_notify_eval_results.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
- 如新增 helper，再新增：
  - `tests/unit/test_eval_reporting_stack_notification_helpers.py`

---

## 建议的设计 / 验证 MD

### Batch 7A

- `docs/DEDUP_EVAL_REPORTING_STACK_PR_COMMENT_CONSUMER_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_STACK_PR_COMMENT_CONSUMER_ALIGNMENT_VALIDATION_20260330.md`

### Batch 7B

- `docs/DEDUP_EVAL_REPORTING_STACK_NOTIFICATION_CONSUMER_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_STACK_NOTIFICATION_CONSUMER_ALIGNMENT_VALIDATION_20260330.md`

