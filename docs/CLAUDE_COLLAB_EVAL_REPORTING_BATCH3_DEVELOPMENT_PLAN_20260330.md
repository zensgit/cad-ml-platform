# Claude Collaboration Batch 3 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `top-level eval reporting` 收口工程，顺序固定：

1. 先补 `eval_reporting_bundle` 的 shared helper 与显式失败语义
2. 再把 `eval_reporting_bundle` 接进默认 materialization 链

执行原则：

- 必须沿用现有 canonical owner
- 只能新增 thin helper / thin orchestration
- 不允许把顶层 bundle 变成新的 metrics owner
- `static report` / `interactive report` 必须继续可独立运行

---

## 当前真实基线

截至当前仓库状态：

- `scripts/summarize_eval_signal_runs.py` 是 `eval_signal` 的唯一 summary owner
- `scripts/summarize_history_sequence_runs.py` 是 `history_sequence` 的唯一 summary owner
- `scripts/ci/generate_eval_signal_reporting_bundle.py` 已存在
- `scripts/ci/generate_history_sequence_reporting_bundle.py` 已存在
- `scripts/ci/generate_eval_reporting_bundle.py` 已存在
- 顶层 bundle 已可 materialize：
  - `eval_signal` sub-bundle
  - `history_sequence` sub-bundle
  - `static report`
  - `interactive report`
- `static report` 与 `interactive report` 已分离到不同输出目录：
  - `report_static/index.html`
  - `report_interactive/index.html`
- 当前剩余缺口：
  - 顶层 bundle 还没有共享 loader helper
  - 顶层 bundle 对 interactive report 还没有显式返回码失败语义
  - 顶层 bundle 还没有接进默认 materialization 链
  - 顶层 bundle 还没有作为 persisted reporting root 出现在默认 artifact 指针里

---

## Batch 3A：Top-Level Helper + Failure Semantics

### 目标

让 `eval_reporting_bundle` 拥有最小但稳定的 shared helper，并补齐顶层 orchestrator 的失败语义。

### 必做改动

1. 新增 `scripts/eval_reporting_bundle_helpers.py`
2. 更新 `scripts/ci/generate_eval_reporting_bundle.py`
3. 如有必要，最小更新：
   - `scripts/generate_eval_report.py`
   - `scripts/generate_eval_report_v2.py`
4. 更新 `Makefile`

### 设计约束

#### helper owner / wrapper 边界

`scripts/eval_reporting_bundle_helpers.py` 只允许负责：

- 加载 `eval_reporting_bundle.json`
- 加载两个 sub-bundle
- 构建顶层 discovery context

它不允许负责：

- summary 聚合
- metrics 计算
- HTML render
- trend plotting
- weekly markdown 生成

#### 顶层 helper 合同

建议至少提供：

- `load_eval_reporting_bundle(history_dir, bundle_json_path=None) -> dict | None`
- `load_eval_reporting_assets(history_dir, bundle_json_path=None) -> (top_level_bundle, eval_signal_bundle, history_sequence_bundle)`
- `build_eval_reporting_discovery_context(top_level_bundle, eval_signal_bundle, history_sequence_bundle) -> dict`

现有 report script 若接入 helper，只允许把它当 discovery loader 使用，不允许反向把 helper 变成 render owner。

#### 顶层 bundle 失败语义

更新 `scripts/ci/generate_eval_reporting_bundle.py`：

- `static report` 失败时继续显式 `return rc`
- `interactive report` 也必须显式检查失败
- 不允许依赖“异常没抛出就算成功”
- 若 interactive report 返回非零，顶层 bundle 必须 fail-closed

#### 顶层 manifest 合同

`eval_reporting_bundle.json` 继续至少包含：

- `status`
- `surface_kind = "eval_reporting_bundle"`
- `generated_at`
- `eval_history_dir`
- `eval_signal_bundle_json`
- `history_sequence_bundle_json`
- `static_report_html`
- `interactive_report_html`
- `plots_dir`

本批允许只做增量补字段，不允许改名或改语义。

### Batch 3A 验收条件

必须同时满足：

- 顶层 helper 能稳定加载 top-level bundle + 两个 sub-bundle
- `generate_eval_reporting_bundle.py` 对 interactive report 具备显式失败判定
- `make eval-reporting-bundle` 仍是 thin wrapper
- static/v2 report 仍可 standalone 运行
- 不引入新的 metrics owner

---

## Batch 3B：Default Materialization + Persisted Artifact Pointer

### 目标

把 `eval_reporting_bundle` 接进默认 reporting 执行链，使其成为默认 materialized root，而不是仅靠手动脚本触发。

### 必做改动

1. 更新 `scripts/eval_with_history.sh`
2. 更新 `scripts/validate_eval_history.py`
3. 更新 `docs/eval_history.schema.json`
4. 更新 `Makefile`

### 设计约束

#### 默认 materialization 顺序

`scripts/eval_with_history.sh` 在持久化 `history_sequence` row 后，必须按这个顺序执行：

1. materialize `history_sequence_reporting_bundle`
2. materialize `eval_reporting_bundle`
3. 再进入后续 gate / validate / summary 路径

要求：

- `eval_reporting_bundle` 必须在 default path 下生成
- 若 `eval_reporting_bundle` 生成失败，脚本必须 fail-closed
- 必须使用独立 exit code，不与已有 bundle/gate 代码冲突

#### persisted artifact pointer

持久化 row 或其 artifact map 必须新增对顶层 bundle 的最小指针，建议至少包含：

- `artifacts.eval_reporting_bundle_json`
- `artifacts.eval_reporting_bundle_md`

如当前 persisted row 模型不适合放 MD 路径，至少必须写入 JSON 路径。

#### schema / validator 对齐

`scripts/validate_eval_history.py` 与 `docs/eval_history.schema.json` 必须同步接受新的 artifact pointer。

#### 不做的事

本批不做：

- 顶层 bundle 新的 metrics 字段
- 顶层 compare report
- 顶层专用 trend
- 改写 static/v2 report 的 standalone 模式

### Batch 3B 验收条件

必须同时满足：

- `eval_with_history.sh` 默认会 materialize 顶层 `eval_reporting_bundle`
- persisted row 或 artifact map 能发现顶层 bundle 路径
- validator / schema 已对齐
- 默认 materialization 失败时脚本 fail-closed
- `make eval-reporting-bundle` 仍保持 thin wrapper

---

## 必须新增或更新的测试

### Batch 3A

- `tests/unit/test_eval_reporting_bundle_helpers.py`
- `tests/unit/test_generate_eval_reporting_bundle.py`
- `tests/unit/test_generate_eval_report.py`
- `tests/unit/test_generate_eval_report_v2.py`
- `tests/unit/test_eval_history_make_targets.py`

### Batch 3B

- `tests/unit/test_eval_with_history_script_history_sequence.py`
- `tests/unit/test_validate_eval_history_history_sequence.py`
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

- `Batch 3A complete, stopped for validation`
或
- `Batch 3B complete, stopped for validation`
