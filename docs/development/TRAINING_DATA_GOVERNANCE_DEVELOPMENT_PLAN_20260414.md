# CAD-ML Platform - Training Data Governance Development Plan

> **版本**: 1.0.0
> **创建日期**: 2026-04-14
> **状态**: Draft
> **优先级**: P0

---

## 1. 背景

当前仓库已经具备较完整的模型能力和训练飞轮能力：

- `src/ml/low_conf_queue.py` 负责低置信样本入队
- `scripts/append_reviewed_to_manifest.py` 负责把审核样本追加进训练 manifest
- `scripts/auto_retrain.sh` 负责自动重训
- `src/core/active_learning.py` / `src/api/v1/active_learning.py` 负责主动学习样本管理
- `src/api/v1/feedback.py` 负责收集人工反馈

但训练数据治理仍存在三个核心缺口：

1. 缺少稳定的数据来源字段，无法区分人工纠正、Claude 建议、模型自动、规则自动。
2. 训练路径默认不够 fail-closed，存在 mock data / synthetic fallback 混入真实训练路径的风险。
3. 训练和评估广泛使用 `random_split(seed=42)`，尚未形成独立且固定的黄金验证集，存在泄露和不可比对风险。

---

## 2. 总目标

建立一条“可追溯、可审计、可拒绝污染数据”的训练数据闭环。

具体目标：

- 所有训练样本都能回答“谁给的标签、是否人工确认、是否允许训练”。
- 所有生产训练默认 fail-closed，不再在缺失真实数据时自动退回 mock / synthetic。
- 所有模型评估统一收口到固定黄金验证集。
- 把 Claude 从“可能隐式进入真值链”收敛到“显式的辅助来源”。

---

## 3. 非目标

本轮不做以下工作：

- 不重写 `analyze.py` 全链路架构
- 不改动 Hybrid / Graph2D / V16 / V6 的模型结构
- 不扩展新的 Claude provider 能力
- 不实现新的自动标注器或新的训练算法

---

## 4. Phase 划分

### Phase 1 - Data Provenance And Fail-Closed Training

目标：先把训练样本来源和训练资格做硬约束。

#### 4.1 核心决策

新增以下字段作为统一治理契约：

- `sample_source`
- `label_source`
- `review_source`
- `human_verified`
- `verified_by`
- `verified_at`
- `eligible_for_training`
- `training_block_reason`

建议枚举值：

- `sample_source`
  - `analysis_review_queue`
  - `feedback_api`
  - `legacy_low_conf_queue`
  - `imported_manifest`
- `label_source`
  - `human_feedback`
  - `human_review`
  - `claude_suggestion`
  - `model_auto`
  - `rule_auto`
  - `synthetic_demo`
- `review_source`
  - `human`
  - `claude_assisted`
  - `mixed`

训练资格规则：

- 只有 `human_verified=true` 且 `eligible_for_training=true` 的样本可进入真实训练导出
- `claude_suggestion` 不能直接视为训练真值
- `claude_suggestion` 只有在人工确认后才允许训练
- `synthetic_demo` 永不进入生产训练导出

#### 4.2 影响文件

- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `src/api/v1/feedback.py`
- `src/ml/low_conf_queue.py`
- `scripts/append_reviewed_to_manifest.py`
- `scripts/finetune_from_feedback.py`
- `scripts/auto_retrain.sh`
- `scripts/train_knowledge_distillation.py`

#### 4.3 关键改动

1. 扩展 `ActiveLearningSample` 和主动学习 API schema
2. 为 feedback API 增加来源归因字段
3. 为低置信审核队列增加 provenance 列
4. 为 manifest 追加脚本增加训练资格筛选
5. 将 `finetune_from_feedback.py` 改为 fail-closed
6. 将 distillation / demo synthetic 路径改为显式 `--demo`

#### 4.4 验收标准

- 没有 provenance 字段的样本默认 `eligible_for_training=false`
- 真实训练脚本拿不到真实向量时直接失败
- `claude_suggestion` 样本未经人工确认不得进入训练导出
- review queue / export / feedback stats 能统计来源分布

---

### Phase 2 - Golden Validation Set And Evaluation Gate

目标：把评估基线固定下来，避免 train/val 污染和横向对比失真。

#### 4.5 核心决策

建立固定黄金验证集，例如：

- `data/manifests/golden_val_set.csv`

原则：

- 黄金验证集不从训练脚本里动态 `random_split`
- 黄金验证集不随训练批次自动变化
- 所有模型评估、自动重训 gate、回归对比统一使用这份集合

#### 4.6 影响文件

- `scripts/finetune_graph2d_v2_augmented.py`
- `scripts/finetune_graph2d_from_pretrained.py`
- `scripts/evaluate_graph2d_v2.py`
- `scripts/auto_retrain.sh`
- 其他显式使用 `random_split(..., seed=42)` 的训练/评估脚本

#### 4.7 关键改动

1. 新增黄金验证 manifest
2. 训练脚本增加：
   - `--train-manifest`
   - `--val-manifest`
   - 或 `--golden-val-manifest`
3. 自动重训评估步骤改为只评估黄金验证集
4. 增加“训练集与黄金集重叠检查”

#### 4.8 验收标准

- 自动重训不再使用隐式随机切分做 gate
- 相同 checkpoint 在重复评估时，验证集样本集合完全一致
- CI 可以报告黄金集规模、来源和重叠检查结果

---

### Phase 3 - Decision Contract Consolidation

目标：为后续从 `analyze.py` 抽离决策逻辑做铺垫，但不在本轮做大重构。

#### 4.9 核心决策

统一以下字段为最终决策契约：

- `part_type`
- `fine_part_type`
- `coarse_part_type`
- `confidence`
- `confidence_source`
- `final_decision_source`
- `needs_review`
- `review_priority`

#### 4.10 影响文件

- `src/api/v1/analyze.py`
- `src/ml/hybrid_classifier.py`
- `src/core/providers/classifier.py`
- `src/core/classification/review_governance.py`

#### 4.11 验收标准

- 训练样本导出只依赖统一决策字段
- review / feedback / active learning 不再重复发明近义字段

---

## 5. 实施顺序

建议严格按以下顺序执行：

1. Phase 1A: schema 与 provenance 字段
2. Phase 1B: fail-closed 训练与导出
3. Phase 2A: 固定黄金验证集
4. Phase 2B: 自动重训 gate 接入黄金集
5. Phase 3: 决策契约收口

不建议跳过 Phase 1 直接做 Phase 2。

---

## 6. 风险与缓解

### 风险 1：历史样本没有 provenance 字段

缓解：

- 历史样本统一标记 `label_source=legacy_unknown`
- 默认 `eligible_for_training=false`
- 允许通过人工批量补录后再放开

### 风险 2：自动重训被 fail-closed 阻断

缓解：

- 短期允许 `--demo` / `--allow-legacy` 显式开关
- 默认生产路径仍保持严格关闭

### 风险 3：黄金验证集过小或类别不平衡

缓解：

- 首版先固化现有验证集
- 第二版再做类分布校正和人工补样

---

## 7. 推荐验收指标

- `training_sample_provenance_coverage = 100%`
- `human_verified_training_ratio = 100%`
- `claude_unverified_training_ratio = 0%`
- `golden_validation_overlap_count = 0`
- `auto_retrain_fail_closed_trigger_count` 可观测

---

## 8. 一句话结论

本计划的核心不是继续扩模型，而是先把“训练数据的可信度和训练资格”做成系统级约束；只有这样，后续 Hybrid / Graph2D / Claude 辅助链路的收益才可持续。
