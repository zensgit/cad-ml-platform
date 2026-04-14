# Training Data Governance Rollout MD

日期：2026-04-14

## 目标

把训练数据治理从“约定”变成“代码路径上的强约束”，并确保后续 Claude 辅助不会污染生产训练真值。

本 MD 面向直接落地执行，默认配合：

- `docs/development/TRAINING_DATA_GOVERNANCE_DEVELOPMENT_PLAN_20260414.md`

一起使用。

---

## Batch 1 - Provenance Schema

### 目标

先扩样本和反馈 schema，让系统能表达：

- 样本从哪里来
- 标签由谁给出
- 是否人工确认
- 是否允许训练

### 必做改动

1. 修改 `src/core/active_learning.py`
2. 修改 `src/api/v1/active_learning.py`
3. 修改 `src/api/v1/feedback.py`
4. 修改 `src/ml/low_conf_queue.py`
5. 更新相关测试

### 建议字段

在 `ActiveLearningSample` 增加：

- `sample_source: str`
- `label_source: str`
- `review_source: Optional[str]`
- `human_verified: bool`
- `verified_by: Optional[str]`
- `verified_at: Optional[datetime]`
- `eligible_for_training: bool`
- `training_block_reason: Optional[str]`

在 `feedback` 输入增加可选字段：

- `label_source`
- `review_source`
- `human_verified`
- `verified_by`

在 `low_conf_queue.csv` 增加列：

- `sample_source`
- `label_source`
- `human_verified`
- `eligible_for_training`

### 执行约束

- 历史未带 provenance 的数据不得默认视为可训练
- 新字段缺失时默认走保守值：
  - `human_verified=false`
  - `eligible_for_training=false`
  - `training_block_reason=missing_provenance`

### 验收

- API / review queue / export 返回体中都能看到 provenance 字段
- pending review 样本默认不会直接被训练导出

---

## Batch 2 - Training Export Gate

### 目标

把“谁可以进入训练”从脚本口头约束改成导出硬门槛。

### 必做改动

1. 修改 `scripts/append_reviewed_to_manifest.py`
2. 修改 `scripts/finetune_from_feedback.py`
3. 修改 `scripts/auto_retrain.sh`
4. 必要时修改 `src/core/active_learning.py` 的 export 逻辑

### 强约束

训练导出必须满足：

- `human_verified=true`
- `eligible_for_training=true`
- `label_source` 不属于：
  - `synthetic_demo`
  - `model_auto`
  - `rule_auto`
- `claude_suggestion` 必须已经被人工确认

### Fail-Closed 要求

`scripts/finetune_from_feedback.py` 必须取消默认 mock fallback：

- 当前行为：
  - 向量缺失时使用随机 mock 数据
- 目标行为：
  - 向量缺失直接失败退出
  - 日志明确写出缺失样本数量与 `doc_id`

`scripts/train_knowledge_distillation.py` 的 synthetic / random teacher 路径必须变成显式：

- 仅当传入 `--demo` 时允许
- 默认生产路径禁止自动 synthetic fallback

### 验收

- 无真实向量时真实训练脚本退出非零
- auto retrain 不会在缺失 provenance 或缺失向量时悄悄继续

---

## Batch 3 - Golden Validation Set

### 目标

建立固定黄金验证集，替换当前广泛存在的 `random_split(seed=42)` 评估方式。

### 必做改动

1. 新建 `data/manifests/golden_val_set.csv`
2. 修改：
   - `scripts/finetune_graph2d_v2_augmented.py`
   - `scripts/finetune_graph2d_from_pretrained.py`
   - `scripts/evaluate_graph2d_v2.py`
   - `scripts/auto_retrain.sh`
3. 增加训练集/黄金集重叠检查

### 实施规则

- 训练脚本应支持固定验证 manifest
- 自动重训 gate 只认黄金集结果
- 黄金集构建后不允许随每次训练自动改写

### 推荐 CLI 变化

- `--golden-val-manifest data/manifests/golden_val_set.csv`
- 或：
  - `--train-manifest`
  - `--val-manifest`

### 验收

- 同一模型重复评估使用完全相同的验证样本
- 自动重训的 gate 结果不再受随机切分波动影响

---

## Batch 4 - Observability And Audit

### 目标

给治理链补最少但必要的可观测性。

### 必做改动

补充统计或日志项：

- `training_export_total`
- `training_export_blocked_total`
- `training_export_blocked_by_reason`
- `training_sample_by_label_source`
- `training_sample_human_verified_ratio`
- `golden_validation_overlap_count`

### 验收

- 可以从日志或 API 统计中回答：
  - 本次训练用了多少人工确认样本
  - 有多少 Claude 辅助样本被拦截
  - 是否存在训练/黄金集重叠

---

## 代码触点清单

### P0

- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `src/api/v1/feedback.py`
- `scripts/finetune_from_feedback.py`
- `scripts/append_reviewed_to_manifest.py`
- `scripts/auto_retrain.sh`

### P1

- `src/ml/low_conf_queue.py`
- `scripts/train_knowledge_distillation.py`
- `scripts/finetune_graph2d_v2_augmented.py`
- `scripts/finetune_graph2d_from_pretrained.py`
- `scripts/evaluate_graph2d_v2.py`

### P2

- `src/api/v1/analyze.py`
- `src/ml/hybrid_classifier.py`
- `src/core/providers/classifier.py`

---

## 验证建议

### 静态检查

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  src/api/v1/feedback.py \
  scripts/finetune_from_feedback.py \
  scripts/append_reviewed_to_manifest.py
```

### 单测建议

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/test_active_learning_api.py \
  tests/test_feedback.py \
  tests/unit/test_low_conf_queue.py -q
```

### 回归建议

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/test_active_learning_api.py \
  tests/test_feedback.py \
  tests/unit/test_low_conf_queue.py \
  tests/test_health_and_metrics.py -q
```

---

## 回滚策略

若 Batch 1 或 Batch 2 引发兼容性问题：

1. 保留新字段，但默认 `eligible_for_training=false`
2. 暂停 `auto_retrain.sh`
3. 只允许人工导出、人工确认后训练

不建议回滚到“训练脚本自动使用 mock / synthetic fallback”的旧行为。

---

## 最终建议

落地顺序必须是：

1. 先做 provenance
2. 再做 fail-closed
3. 再固化黄金验证集
4. 最后才谈大规模自动重训和更深的架构收口

如果只能做一件事，先做 Batch 1 + Batch 2。
