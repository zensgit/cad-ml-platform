# Claude Collaboration Batch 4 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `eval reporting` 的运营化收口，顺序固定：

1. 先补顶层 `eval_reporting_bundle` 的 health / freshness / pointer guard
2. 再补一个 CI/cron 友好的 `refresh + discovery root`

执行原则：

- 必须沿用现有 canonical owner
- 只能新增 thin checker / thin orchestrator / thin discovery artifact
- 不允许把顶层 bundle、health report 或 discovery index 变成新的 metrics owner
- `static report` / `interactive report` / sub-bundle 仍必须可独立运行

---

## 当前真实基线

截至当前仓库状态：

- `scripts/ci/generate_eval_signal_reporting_bundle.py` 已存在
- `scripts/ci/generate_history_sequence_reporting_bundle.py` 已存在
- `scripts/ci/generate_eval_reporting_bundle.py` 已存在
- `scripts/eval_reporting_bundle_helpers.py` 已存在
- `scripts/eval_with_history.sh` 已默认 materialize：
  1. `history_sequence_reporting_bundle`
  2. `eval_reporting_bundle`
  3. explainability guard
- persisted history row 已包含：
  - `artifacts.reporting_bundle_json`
  - `artifacts.eval_reporting_bundle_json`
- 顶层 bundle 已稳定输出：
  - `eval_reporting_bundle.json`
  - `eval_reporting_bundle.md`
  - `report_static/index.html`
  - `report_interactive/index.html`
- 当前剩余缺口：
  - 没有独立的 `eval_reporting_bundle` health/freshness report
  - 没有顶层 pointer mismatch / missing artifact 守护
  - 没有单一 `refresh` 入口供 CI/cron 直接调用
  - 没有顶层 discovery/index artifact

---

## Batch 4A：Top-Level Bundle Health Guard

### 目标

让顶层 `eval_reporting_bundle` 拥有独立、可持久化、可被 CI/cron 消费的 health report。

### 必做改动

1. 新增 `scripts/ci/check_eval_reporting_bundle_health.py`
2. 如有必要，最小扩充 `scripts/eval_reporting_bundle_helpers.py`
3. 更新 `Makefile`

### 设计约束

#### health checker owner / wrapper 边界

`scripts/ci/check_eval_reporting_bundle_health.py` 可以拥有：

- top-level bundle health 判断
- freshness 计算
- pointer mismatch 检查
- 输出 JSON / Markdown health report

但它不允许拥有：

- sub-bundle summary 聚合
- report HTML 渲染
- top-level bundle materialization
- weekly/trend owner 逻辑

#### checker 输入

health checker 必须以 `eval_reporting_bundle.json` 为 canonical root，再读取：

- `eval_signal_bundle_json`
- `history_sequence_bundle_json`
- `static_report_html`
- `interactive_report_html`
- `plots_dir`

如 top-level bundle 缺失，可以 fallback 直接看默认路径，但最终状态必须明确区分：

- `missing_root_bundle`
- `missing_sub_bundle`
- `missing_report`
- `stale_bundle`
- `pointer_mismatch`

#### health report 合同

建议默认输出：

- `reports/eval_history/eval_reporting_bundle_health_report.json`
- `reports/eval_history/eval_reporting_bundle_health_report.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_bundle_health_report"`
- `generated_at`
- `eval_history_dir`
- `bundle_json`
- `summary`
- `checks`
- `missing_artifacts`
- `stale_artifacts`
- `mismatch_artifacts`

其中：

- `summary` 至少包含：
  - `ok`
  - `missing_count`
  - `stale_count`
  - `mismatch_count`
- `checks` 是平铺的 named checks

#### freshness 约束

允许使用 CLI threshold，建议至少支持：

- `--max-root-age-hours`
- `--max-sub-bundle-age-hours`
- `--max-report-age-hours`

默认阈值要保守，不得把轻微延迟直接判死。

#### Make target

新增：

- `eval-reporting-bundle-health`

必须是 thin wrapper。

### Batch 4A 验收条件

必须同时满足：

- health checker 能在 root bundle 存在时正常产出 JSON / MD
- 能正确区分 missing / stale / mismatch 三类问题
- 顶层 helper 若扩充，也没有越权成为新的 owner
- `make eval-reporting-bundle-health` 仍是 thin wrapper

---

## Batch 4B：Refresh Entry + Discovery Root

### 目标

补一个 CI/cron 友好的单一 refresh 入口，并 materialize 顶层 discovery/index artifact。

### 必做改动

1. 新增 `scripts/ci/refresh_eval_reporting_stack.py`
2. 新增 `scripts/ci/generate_eval_reporting_index.py`
3. 更新 `Makefile`

### 设计约束

#### refresh orchestrator 角色

`scripts/ci/refresh_eval_reporting_stack.py` 只允许顺序调用：

1. `scripts/ci/generate_eval_reporting_bundle.py`
2. `scripts/ci/check_eval_reporting_bundle_health.py`
3. `scripts/ci/generate_eval_reporting_index.py`

它不允许：

- 自己重算 summary
- 自己生成 HTML report
- 自己拥有新的 metrics schema

#### discovery/index artifact

默认输出：

- `reports/eval_history/eval_reporting_index.json`
- `reports/eval_history/eval_reporting_index.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_index"`
- `generated_at`
- `eval_history_dir`
- `eval_reporting_bundle_json`
- `eval_reporting_bundle_health_json`
- `eval_signal_bundle_json`
- `history_sequence_bundle_json`
- `static_report_html`
- `interactive_report_html`
- `plots_dir`

它只做 discovery / navigation，不复制 aggregate metrics。

#### refresh 入口

新增：

- `make eval-reporting-refresh`

它必须是 CI/cron 友好的单命令入口，并保持 thin wrapper。

#### 不做的事

本批不做：

- GitHub workflow 文件改造
- 顶层 compare leaderboard
- 新图表
- 新 summary owner

### Batch 4B 验收条件

必须同时满足：

- `refresh_eval_reporting_stack.py` 能 materialize bundle + health + index
- `eval_reporting_index.json/md` 可作为顶层 discovery root
- `make eval-reporting-refresh` 是稳定的单入口
- 任何一步失败时 refresh 脚本 fail-closed

---

## 必须新增或更新的测试

### Batch 4A

- `tests/unit/test_check_eval_reporting_bundle_health.py`
- `tests/unit/test_eval_reporting_bundle_helpers.py`
- `tests/unit/test_eval_history_make_targets.py`

### Batch 4B

- `tests/unit/test_refresh_eval_reporting_stack.py`
- `tests/unit/test_generate_eval_reporting_index.py`
- `tests/unit/test_eval_history_make_targets.py`
- `tests/unit/test_generate_eval_reporting_bundle.py`

---

## Claude 输出要求

Claude 每批次完成后必须同时提交：

- 代码变更
- 该批 design MD
- 该批 validation MD
- 实际执行命令
- 实际测试结果
- 未解决风险
- 更新后的 validation ledger

并且必须明确声明：

- `Batch 4A complete, stopped for validation`
或
- `Batch 4B complete, stopped for validation`
