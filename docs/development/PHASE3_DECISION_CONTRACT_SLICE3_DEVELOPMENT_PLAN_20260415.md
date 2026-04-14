# Phase 3 Decision Contract Slice 3 Development Plan

日期: 2026-04-15

## 背景

Slice 1 已收口最终 decision contract。  
Slice 2 已抽离最终 finalization policy。  
但 `src/api/v1/analyze.py` 里仍保留一段关键的 override 决策内联逻辑：

- `FusionAnalyzer` override
- `HybridClassifier` override

这段逻辑的特征是：

- 与 env flag 紧耦合
- 会直接改写 `part_type / confidence / rule_version / confidence_source`
- 是最终分类覆盖顺序的关键部分

如果这段逻辑继续留在 `analyze.py`，后续要抽 decision service 时，主复杂度仍在 API 层。

## 本次目标

本次只抽 override policy，不改覆盖顺序。

目标：

- 新增共享 override policy helper
- `analyze.py` 改为只负责读取 env 和调用 helper
- 保持现有行为：
  - Fusion override 先于 Hybrid override
  - `default_rule_only` 仍跳过 Fusion override
  - Hybrid `env / auto / auto_low_conf / auto_drawing_type` 语义不变

## 设计

新增模块：

- `src/core/classification/override_policy.py`

导出：

1. `apply_fusion_override(...)`
2. `apply_hybrid_override(...)`

### Fusion helper 职责

- 在 `override_enabled=false` 时不改 payload
- 阈值不足时写 `fusion_override_skipped`
- `source=rule_based && rule_hits=[RULE_DEFAULT]` 时写 `reason=default_rule_only`
- 满足条件时写：
  - `part_type`
  - `confidence`
  - `rule_version = FusionAnalyzer-*`
  - `confidence_source = fusion`

### Hybrid helper 职责

- 保持现有四种 mode：
  - `env`
  - `auto`
  - `auto_low_conf`
  - `auto_drawing_type`
- 满足条件时写：
  - `hybrid_override_applied`
  - `part_type`
  - `confidence`
  - `rule_version = HybridClassifier-v1`
  - `confidence_source = hybrid`
- `env` 强制模式但不满足阈值时写 `hybrid_override_skipped`

## analyze 接线

`analyze.py` 仍负责：

- 构建 `fusion_decision`
- 构建 `hybrid_result`
- 读取 env 和阈值

但不再自己展开 override 条件树，而是：

- `cls_payload = apply_fusion_override(...)`
- `cls_payload = apply_hybrid_override(...)`

## 测试策略

新增：

- `tests/unit/test_classification_override_policy.py`

覆盖：

- Fusion override 正常覆盖
- Fusion `default_rule_only` 跳过
- Hybrid `auto`
- Hybrid `auto_low_conf`
- Hybrid `env skip`

回归：

- `test_analyze_dxf_fusion.py`
- `test_analyze_dxf_hybrid_override.py`
- `test_analyze_dxf_coarse_knowledge_outputs.py`
- `similarity / compare / vectors / qdrant` contract 测试

## 非目标

本次不做：

- 不改 `fusion_decision` 的输入构造
- 不改 `graph2d_fusable` 的 gate 逻辑
- 不改 finalization policy
- 不改 active learning 逻辑
