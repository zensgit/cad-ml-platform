# Phase 3 Decision Contract Rollout

日期: 2026-04-15

## 目标

把 `src/api/v1/analyze.py` 中与最终分类结果相关的核心职责逐步抽离，形成稳定、可复用的决策收口结构。

本轮 Phase 3 分三刀完成：

1. `decision contract`
2. `finalization policy`
3. `override policy`

## Slice 1

### 目标

统一最终分类字段 contract：

- `part_type`
- `fine_part_type`
- `coarse_part_type`
- `decision_source`
- `final_decision_source`
- `is_coarse_label`

### 落地

- `src/core/classification/decision_contract.py`
- `src/core/similarity.py` 改为复用共享 helper
- `src/api/v1/analyze.py` 写侧与 vector metadata 导出统一走共享 helper

### 结果

- `analyze` 响应和向量 metadata 不再各自维护一套 contract 逻辑
- `similarity/compare/vectors/qdrant` 读侧统一到一套解析语义

## Slice 2

### 目标

抽离 `analyze` 最终收口逻辑：

- coarse label 派生
- branch conflicts
- knowledge summary
- review governance

### 落地

- `src/core/classification/finalization.py`
- `src/api/v1/analyze.py` 改为调用 `finalize_classification_payload(...)`

### 结果

- `analyze.py` 不再直接展开大段最终收口细节
- finalization policy 从 API 层抽成共享 helper

## Slice 3

### 目标

抽离最终覆盖逻辑：

- `FusionAnalyzer override`
- `Hybrid override`

### 落地

- `src/core/classification/override_policy.py`
- `src/api/v1/analyze.py` 改为调用：
  - `apply_fusion_override(...)`
  - `apply_hybrid_override(...)`

### 结果

- 最终覆盖顺序仍保持：
  - `Fusion override`
  - `Hybrid override`
  - `finalization`
  - `active learning`
- override policy 从 API 层抽成共享 helper

## 当前结构

Phase 3 完成后，最终分类主链已经分成 3 层：

1. `override_policy`
   - 负责 Fusion/Hybrid 是否改写最终标签
2. `finalization`
   - 负责 coarse/knowledge/review 的最终收口
3. `decision_contract`
   - 负责稳定字段的最终暴露与读写一致性

## 本轮新增文件

- `src/core/classification/decision_contract.py`
- `src/core/classification/finalization.py`
- `src/core/classification/override_policy.py`
- `tests/unit/test_classification_decision_contract.py`
- `tests/unit/test_classification_finalization.py`
- `tests/unit/test_classification_override_policy.py`

## 关键收益

- `analyze.py` 明显瘦身，热点复杂度开始分层
- 最终分类 contract、finalization、override 各自有独立测试面
- 后续继续抽 decision service 时，不必再从 API 路由里硬拆

## 后续建议

如果继续做 Phase 3，下一步最合理的是：

- 把 env 读取和 policy 参数解析进一步收口
- 再往前抽 `graph2d_fusable / l4_prediction / fusion_inputs` 这类输入构造
- 最终形成独立的 classification orchestration / decision service
