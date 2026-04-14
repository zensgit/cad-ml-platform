# Training Data Governance Gates Rollout MD

日期：2026-04-14

## 目标

在已有 `Training Data Governance Phase 1 + 2` 基础上，完成最后一层工程化收口：

- 把训练数据治理从 `core-fast` 子步骤提升为独立可见的 CI 信号
- 把本地默认执行环境收口到兼容仓库依赖的 `.venv311`
- 修复 active learning 直调路径与治理语义之间的默认值断层
- 固化一组可重复运行的治理回归测试入口

本 MD 是对以下文档的增量落地：

- `docs/development/TRAINING_DATA_GOVERNANCE_DEVELOPMENT_PLAN_20260414.md`
- `docs/development/TRAINING_DATA_GOVERNANCE_ROLLOUT_MD_20260414.md`
- `docs/development/TRAINING_DATA_GOVERNANCE_VERIFICATION_20260414.md`

---

## 本批范围

### 1. 独立治理工作流

新增独立 GitHub Actions workflow：

- `.github/workflows/governance-gates.yml`

职责：

- 运行 `make validate-training-governance`
- 运行 `make test-training-governance`
- 上传治理日志 artifact
- 在 `GITHUB_STEP_SUMMARY` 中写入单独摘要

设计意图：

- 不替换现有 `CI` / `CI Tiered Tests`
- 单独暴露训练治理信号，便于快速判断是“模型能力回归”还是“数据治理回归”

### 2. Make 目标收口

新增目标：

- `make test-training-governance`

用途：

- 一键运行治理相关快速回归
- 作为新 workflow 的主执行入口
- 作为本地开发时的最小验证命令

纳入测试集：

- `tests/unit/test_active_learning_loop.py`
- `tests/test_active_learning_api.py`
- `tests/unit/test_low_conf_queue.py`
- `tests/unit/test_finetune_from_feedback.py`
- `tests/unit/test_training_scripts.py`
- `tests/unit/test_training_data_governance.py`
- `tests/unit/test_auto_retrain_governance.py`
- `tests/unit/test_check_training_data_governance.py`

### 3. 本地 Python 环境优先级修正

修改 `Makefile` 的 `PYTHON` 解析顺序：

- 优先 `.venv311/bin/python`
- 次选 `.venv/bin/python`
- 再回退到系统 `python3.11` / `python3.10` / `python3`

原因：

- 旧 `.venv` 是 Python 3.13，不兼容当前 pinned 依赖
- `.venv311` 已验证可完整安装 `requirements.txt + requirements-dev.txt`
- 如果不改优先级，本地 `make` 仍可能误用旧环境

### 4. 依赖文件收口

修正 `requirements-dev.txt` 中的冲突 pin：

- `types-requests==2.32.4.20250913` -> `types-requests==2.31.0.0`

原因：

- 仓库同时 pin 了 `urllib3==1.26.20`
- 新版本 `types-requests` 解析依赖到 `urllib3>=2`
- 该冲突会导致 `pip install -r requirements-dev.txt` 失败

### 5. Active Learning 默认 provenance 收口

修改：

- `src/core/active_learning.py`

变更：

- `submit_feedback()` 默认 `label_source="human_feedback"`
- 内部统一做 `normalized_label_source`

目的：

- 修复 direct-call 路径下，人工反馈未显式传 `label_source` 时：
  - 样本会变成 `LABELED`
  - 但不会进入 `eligible_for_training`
  - 从而让 threshold / export / API stats 一起失真

### 6. 测试隔离修复

修改：

- `tests/unit/test_active_learning_loop.py`
- `tests/unit/test_training_data_governance.py`

收口点：

- 旧文案断言从 `labeled_samples` 迁移到 `eligible_samples`
- `training_data_governance` 测试增加 `autouse` fixture
- 显式 reset singleton + 隔离 `ACTIVE_LEARNING_*` 环境变量

目的：

- 避免 file-store 持久化状态污染行为测试
- 让治理测试对运行顺序不敏感

---

## 代码触点

### 新增

- `.github/workflows/governance-gates.yml`
- `scripts/backfill_manifest_cache_paths.py`
- `scripts/ci/check_training_data_governance.py`
- `tests/unit/test_auto_retrain_governance.py`
- `tests/unit/test_check_training_data_governance.py`

### 修改

- `Makefile`
- `requirements-dev.txt`
- `scripts/auto_retrain.sh`
- `scripts/ci/summarize_core_fast_gate.py`
- `src/core/active_learning.py`
- `tests/unit/test_active_learning_loop.py`
- `tests/unit/test_training_data_governance.py`

---

## 设计决策

### 为什么增加独立 workflow，而不是只留在 core-fast

因为 `core-fast` 更像聚合门禁：

- 失败时不够细
- 很难一眼判断是 tolerance / openapi / provider / training governance 哪条线出问题

独立 `Governance Gates` 的价值是：

- 信号更清晰
- 更适合后续单独加 artifact / report / owner
- 不破坏现有 `CI` 结构

### 为什么没有把 Governance Gates 立即加入 `CI_WATCH_REQUIRED_WORKFLOWS`

本批故意不扩大 watcher 的 required workflow 集合。

原因：

- watcher 相关默认名单在 `Makefile` 和若干脚本里有自己的稳定性约束
- 当前目标是先把治理门禁独立可见化
- 等 workflow 在主干稳定一轮后，再决定是否提升为 watcher-required

这是一项后续可选增强，不属于本批阻塞项

### 为什么 Claude Code CLI 只作为 sidecar

可以调用，但不应成为主执行路径依赖。

原则：

- 治理门禁、训练导出、验证脚本必须能在纯本地环境复现
- Claude CLI 适合做附加审阅、摘要、研发辅助
- 不适合成为 CI 必需组件

---

## 推荐执行顺序

### 本地

```bash
make validate-training-governance
make test-training-governance
```

### GitHub Actions

观察以下信号：

- `CI`
- `CI Tiered Tests`
- `Governance Gates`

### 环境

统一使用：

```bash
source .venv311/bin/activate
```

---

## 后续建议

### P0

- 观察 `Governance Gates` 在真实 `push` / `pull_request` 上至少一轮稳定运行
- 若稳定，再考虑是否加入 watcher-required workflow 清单

### P1

- 给 `Governance Gates` 增加更细的 step summary
- 单独输出治理 JSON artifact 供后续报告消费

### P2

- 进入 Phase 3：收口 analyze 决策契约
- 把 `part_type / fine_part_type / coarse_part_type / confidence_source / needs_review` 定成唯一决策接口

---

## 一句话总结

这一批不是“再加功能”，而是把训练数据治理从“主链路已收口”推进到“本地、CI、文档三处都可复现、可见、可维护”。
