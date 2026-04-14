# Phase 3 Decision Contract Slice 1 Development Plan

日期: 2026-04-15

## 背景

`src/api/v1/analyze.py` 的最终分类字段长期以隐式 `cls_payload` 形式存在。  
读侧已经有 `src/core/similarity.py::extract_vector_label_contract()`，但写侧仍在手工拼装：

- `part_type`
- `fine_part_type`
- `coarse_part_type`
- `final_decision_source`
- `is_coarse_label`

这会导致两个问题：

1. `analyze` 响应与向量 metadata 的字段归一化逻辑分叉。
2. 后续 Phase 3 想继续抽 decision service 时，没有一个稳定的最小 contract 可复用。

## 本次目标

本次只做 Phase 3 的第一刀，不改主决策顺序，不抽大服务，只先收口最终字段契约。

目标：

- 新增共享 decision contract helper
- 让 `analyze` 的最终写侧走共享 helper
- 让 `similarity/qdrant` 的读侧复用同一 helper
- 保持现有融合 / Hybrid override / review governance 顺序不变

## 设计

新增模块：

- `src/core/classification/decision_contract.py`

提供两个 helper：

1. `extract_label_decision_contract(payload)`
   - 从任意 payload 中提取稳定 label contract
   - 统一处理：
     - `part_type`
     - `fine_part_type`
     - `coarse_part_type`
     - `decision_source`
     - `final_decision_source`
     - `is_coarse_label`

2. `build_classification_decision_contract(payload)`
   - 面向 `analyze` 最终分类结果
   - 在 label contract 基础上补齐：
     - `confidence_source`
     - `rule_version`

## 接线范围

### 1. analyze 写侧

位置：

- `src/api/v1/analyze.py`

动作：

- 在 review governance 之后、写入 `results["classification"]` 之前，统一补齐最终 decision contract
- 这样不会改变上游判定顺序，只改变最终字段的稳定暴露方式

### 2. 向量 metadata 导出

位置：

- `src/api/v1/analyze.py`

动作：

- 用共享 helper 替代原本手写的 `part_type/fine/coarse/final_decision_source/is_coarse_label` 回填逻辑

### 3. similarity / qdrant 读侧

位置：

- `src/core/similarity.py`

动作：

- 让 `extract_vector_label_contract()` 复用共享 helper
- 保持对外函数名不变，降低兼容风险

## 预期结果

本次完成后：

- `analyze` 响应会稳定暴露 `decision_source` 和 `is_coarse_label`
- `fine_part_type` 在无 Hybrid 细分类时也能稳定回落到最终 `part_type`
- 向量 metadata 与 `similarity` 读侧不再各自维护一套 contract 逻辑

## 非目标

本次不做：

- 不改 Fusion / Hybrid / Graph2D 的覆盖顺序
- 不抽独立 decision service
- 不修改 active learning / review governance 策略
- 不改 compare / similarity 的外部 API 结构
