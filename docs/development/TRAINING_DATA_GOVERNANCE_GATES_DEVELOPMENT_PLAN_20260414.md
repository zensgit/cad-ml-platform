# Training Data Governance Gates Development Plan

日期：2026-04-14

## 背景

`Training Data Governance Phase 1 + 2` 已经完成了主训练链的 provenance、fail-closed、golden validation 和 retrain ingestion 收口，但还缺最后一层工程化约束：

- 本地默认环境仍可能误用不兼容的 `.venv`
- 治理验证主要依附于 `core-fast`，缺独立可见信号
- active learning 在 direct-call 路径上存在默认 provenance 缺口
- 治理相关测试在整组运行时仍受 singleton / env / file-store 污染风险影响

本计划针对这些“最后一公里”问题做收口。

---

## 总目标

把训练数据治理从“主路径逻辑正确”推进到“本地、CI、文档三处都稳定可复现”。

具体目标：

1. 给治理链单独的 GitHub Actions 信号
2. 固化一组一键可跑的治理回归测试入口
3. 默认使用兼容仓库依赖的 `.venv311`
4. 修复 active learning 默认 provenance 与 `eligible_for_training` 语义的断层
5. 消除治理测试的运行顺序污染

---

## 非目标

本批不做以下事项：

- 不改 watcher required workflow 列表
- 不重构 `analyze.py` 的决策契约
- 不把 Claude Code CLI 变成主执行依赖
- 不处理第三方包 `ezdxf/pyparsing` 的弃用 warning

---

## 实施范围

### Phase A - Environment Hardening

目标：

- 修正本地默认 Python 解释器选择
- 修掉开发依赖解析冲突

计划：

- `Makefile` 优先使用 `.venv311/bin/python`
- 调整 `requirements-dev.txt` 中与 `urllib3==1.26.20` 冲突的 pin

验收：

- `make` 默认落到 `.venv311`
- `pip install -r requirements.txt -r requirements-dev.txt` 可完成
- `pip check` 无 broken requirements

### Phase B - Governance Entry Points

目标：

- 给治理检查和治理回归单独入口

计划：

- 保留 `make validate-training-governance`
- 新增 `make test-training-governance`
- 统一由 `.venv311` 执行

验收：

- 一条命令可跑 invariants
- 一条命令可跑治理相关 pytest 回归

### Phase C - CI Visibility

目标：

- 把治理门禁从 `core-fast` 子步骤升级为独立 workflow 信号

计划：

- 新增 `.github/workflows/governance-gates.yml`
- 运行：
  - `make validate-training-governance`
  - `make test-training-governance`
- 上传日志并写入 step summary

验收：

- GitHub Actions 中出现独立 `Governance Gates`
- 不破坏现有 `CI` / `CI Tiered Tests`

### Phase D - Active Learning Semantics Alignment

目标：

- 修复 direct-call feedback 路径和治理语义不一致

计划：

- `src/core/active_learning.py::submit_feedback()` 默认 `label_source="human_feedback"`
- 内部统一做 `normalized_label_source`

验收：

- 直调 `submit_feedback(sample.id, "bolt")` 后：
  - `human_verified=true`
  - `eligible_for_training=true`
  - retrain threshold / export / stats 与 API 路径一致

### Phase E - Test Isolation

目标：

- 消除治理测试对运行顺序和 file-store 状态的敏感性

计划：

- `tests/unit/test_training_data_governance.py` 增加 `autouse` fixture
- reset singleton
- 隔离 `ACTIVE_LEARNING_*` 环境变量
- 更新旧断言文案到 `eligible_samples` 语义

验收：

- 单测单独跑通过
- 整组治理回归跑通过

---

## 代码触点

### 新增文件

- `.github/workflows/governance-gates.yml`
- `scripts/backfill_manifest_cache_paths.py`
- `scripts/ci/check_training_data_governance.py`
- `tests/unit/test_auto_retrain_governance.py`
- `tests/unit/test_check_training_data_governance.py`

### 修改文件

- `Makefile`
- `requirements-dev.txt`
- `scripts/auto_retrain.sh`
- `scripts/ci/summarize_core_fast_gate.py`
- `src/core/active_learning.py`
- `tests/unit/test_active_learning_loop.py`
- `tests/unit/test_training_data_governance.py`

---

## 风险与控制

### 风险 1

独立 workflow 增加 CI 面积。

控制：

- 只增加一条轻量 Python 3.11 workflow
- 不改现有 CI 依赖关系

### 风险 2

改动 `submit_feedback()` 默认值可能影响非 API 调用方。

控制：

- 该默认值与 API 层 `FeedbackRequest(default="human_feedback")` 保持一致
- `claude_suggestion` / `model_auto` 仍需显式传入，语义更清晰

### 风险 3

 watcher required workflow 扩张会连带触发一串 guardrail 变更。

控制：

- 本批明确不做 watcher 集合扩张
- 先观察 `Governance Gates` 单独运行稳定性

---

## 完成标准

满足以下条件即可视为本批完成：

1. `make validate-training-governance` 通过
2. `make test-training-governance` 通过
3. 新 workflow YAML 结构有效
4. active learning 相关失败用例全部恢复
5. 开发与验证文档落地

---

## 与 Claude Code CLI 的关系

Claude Code CLI 可以调用，但仅限 sidecar 场景：

- 审阅摘要
- 文档辅助
- 变更复核

主线原则不变：

- 治理门禁必须脱离 Claude CLI 独立可复现
