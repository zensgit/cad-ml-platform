# Training Data Governance Gates 验证报告

**日期**: 2026-04-14  
**验证范围**: Governance Gates 独立 workflow + `.venv311` 环境收口 + active learning/default provenance 修复 + 治理回归测试收口

---

## 1. 环境验证

### 1.1 Python 环境

本地兼容环境：

- `.venv311`
- `Python 3.11.15`

结论：

- 可完整安装 `requirements.txt`
- 可完整安装修正后的 `requirements-dev.txt`
- `pip check` 无 broken requirements

### 1.2 依赖冲突修复

问题：

- `requirements.txt` 固定 `urllib3==1.26.20`
- `requirements-dev.txt` 原先固定 `types-requests==2.32.4.20250913`
- 两者解析冲突，导致开发依赖无法完整安装

修复：

- 将 `types-requests` pin 到 `2.31.0.0`

验证结果：

- `pip install -r requirements-dev.txt` 通过
- `pip check` 通过

---

## 2. 治理门禁验证

### 2.1 Invariants 检查

执行命令：

```bash
make validate-training-governance
```

结果：

```json
{"status": "ok", "checks_count": 10, "violations_count": 0}
```

验证点包括：

- `golden_train_set.csv` / `golden_val_set.csv` 存在
- train / val `file_path` 零重叠
- train / val `cache_path` 零重叠
- `auto_retrain.sh` 含 provenance / fail-closed / backfill 关键 token
- `append_reviewed_to_manifest.py` 含 human-verified gate
- 两个 Graph2D finetune 脚本含 leakage prevention
- `active_learning` API / core 含治理默认值
- `backfill_manifest_cache_paths.py` 存在且含关键逻辑

### 2.2 Governance 回归测试

执行命令：

```bash
make test-training-governance
```

结果：

```text
80 passed, 7 warnings in 2.32s
```

覆盖测试集：

- `tests/unit/test_active_learning_loop.py`
- `tests/test_active_learning_api.py`
- `tests/unit/test_low_conf_queue.py`
- `tests/unit/test_finetune_from_feedback.py`
- `tests/unit/test_training_scripts.py`
- `tests/unit/test_training_data_governance.py`
- `tests/unit/test_auto_retrain_governance.py`
- `tests/unit/test_check_training_data_governance.py`

说明：

- `7 warnings` 均为 `ezdxf/queryparser.py` 的 `pyparsing` 弃用告警
- 本批不处理该类第三方 warning

---

## 3. Active Learning 修复验证

### 3.1 submit_feedback 默认 provenance

修复点：

- `src/core/active_learning.py`

变更：

- `submit_feedback(..., label_source="human_feedback")`
- 内部使用 `normalized_label_source`

修复前表现：

- 测试直接调用 `learner.submit_feedback(sample.id, "bolt")`
- 样本虽然变成 `LABELED`
- 但不进入 `eligible_for_training`
- 导致：
  - `check_retrain_threshold()` 不 ready
  - `/active-learning/stats` 不 ready
  - `export_training_data()` 返回 `No samples to export`

修复后验证：

以下 5 个定向失败用例已全部通过：

- `TestActiveLearningLoop.test_retrain_threshold_trigger`
- `TestActiveLearningLoop.test_export_training_data`
- `TestActiveLearningLoadSamples.test_retrain_threshold_recommendation_when_not_ready`
- `test_active_learning_stats_retrain_ready`
- `test_active_learning_export_labeled`

### 3.2 治理测试隔离

问题：

- `tests/unit/test_training_data_governance.py` 在整组运行时会受前序 `ACTIVE_LEARNING_STORE=file` 和持久化 singleton 状态污染
- 导致 `export_training_data()` 计数偶发从 `3` 变成 `4`

修复：

- 新增 `autouse` fixture
- 测试前后 `reset_active_learner()`
- 强制设置：
  - `ACTIVE_LEARNING_STORE=memory`
  - `ACTIVE_LEARNING_DATA_DIR=<tmp>`
  - `ACTIVE_LEARNING_RETRAIN_THRESHOLD=10`

验证：

- `test_export_default_only_eligible` 在完整治理回归中稳定通过
- 断言已收口为 `count == 3`

---

## 4. Workflow 验证

### 4.1 新增工作流

文件：

- `.github/workflows/governance-gates.yml`

本地结构校验：

```python
{'name': 'Governance Gates', 'jobs': ['training-governance']}
```

工作流行为：

- checkout
- setup-python 3.11
- install `requirements.txt + requirements-dev.txt`
- run `make validate-training-governance`
- run `make test-training-governance`
- upload governance logs
- append step summary

### 4.2 与现有 CI 的关系

结论：

- 现有 `CI` / `CI Tiered Tests` 仍保留 `validate-core-fast`
- `Governance Gates` 是额外、独立、可见信号
- 本批**没有**把它加入 `CI_WATCH_REQUIRED_WORKFLOWS`

理由：

- 先观察真实 workflow 稳定性
- 避免一次性扩大 watcher required set

---

## 5. Claude Code CLI 验证

本机可执行：

```bash
claude --version
```

结果：

```text
2.1.105 (Claude Code)
```

附加验证：

- 使用 `claude -p --output-format text --tools "" ...` 做了一次无工具 sidecar 审阅
- CLI 成功返回了 4 点变更摘要与风险建议

结论：

- Claude Code CLI 可调用
- 适合作为 sidecar 审阅或摘要工具
- 本批实现与验证**未依赖** Claude CLI 成功执行，主路径仍然是本地可复现脚本与 pytest

---

## 6. 变更文件

### 新增

- `.github/workflows/governance-gates.yml`
- `docs/development/TRAINING_DATA_GOVERNANCE_GATES_DEVELOPMENT_PLAN_20260414.md`
- `docs/development/TRAINING_DATA_GOVERNANCE_GATES_ROLLOUT_MD_20260414.md`
- `docs/development/TRAINING_DATA_GOVERNANCE_GATES_VERIFICATION_20260414.md`
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

## 7. 最终结论

| 项目 | 状态 |
|------|------|
| `.venv311` 本地兼容环境 | ✓ |
| `requirements-dev.txt` 冲突修复 | ✓ |
| active learning 默认 provenance 修复 | ✓ |
| 治理测试隔离修复 | ✓ |
| `validate-training-governance` 门禁 | ✓ |
| `test-training-governance` 回归入口 | ✓ |
| `Governance Gates` 独立 workflow | ✓ |
| 本地验证 | ✓ |

**结论**：

训练数据治理现在已经具备三层闭环：

1. 代码路径 fail-closed  
2. 本地 `make` / pytest 可复现  
3. GitHub Actions 中独立可见

这意味着该治理链已经从“实现完成”推进到“工程上可持续维护”。
