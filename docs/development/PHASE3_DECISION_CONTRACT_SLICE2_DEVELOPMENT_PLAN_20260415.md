# Phase 3 Decision Contract Slice 2 Development Plan

日期: 2026-04-15

## 背景

Slice 1 已经把最终分类字段 contract 收口成共享 helper，但 `src/api/v1/analyze.py` 里仍保留了大段“最终收口”逻辑，包括：

- coarse label 派生
- branch conflict 聚合
- knowledge summary 构建
- review governance 构建

这段逻辑仍然直接堆在 API 路由里，导致：

1. `analyze.py` 仍承担过多策略收口职责
2. 下一步继续抽 decision service 时，很难只替换局部
3. contract 虽已统一，但 finalize policy 还没统一

## 本次目标

本次只抽“最终收口 policy”，不改判定顺序。

目标：

- 新增共享 finalization helper
- `analyze.py` 改为只传入上下文和阈值
- 保持 Fusion / Hybrid override / Graph2D gate / Active Learning 顺序不变

## 设计

新增模块：

- `src/core/classification/finalization.py`

导出：

- `finalize_classification_payload(...)`

职责：

1. 用 Slice 1 的 decision contract helper 先补齐稳定字段
2. 生成以下 coarse 派生字段：
   - `coarse_fine_part_type`
   - `coarse_hybrid_label`
   - `coarse_graph2d_label`
   - `coarse_filename_label`
   - `coarse_titleblock_label`
   - `coarse_history_label`
   - `coarse_process_label`
   - `coarse_part_family`
3. 聚合 `branch_conflicts`
4. 生成 `knowledge_checks / violations / standards_candidates / knowledge_hints`
5. 运行 `build_review_governance`
6. 最后再次补齐稳定 decision contract

## analyze 接线

`src/api/v1/analyze.py` 只保留：

- Hybrid override 之前和之中的原有逻辑
- review threshold 读取
- 调用 `finalize_classification_payload(...)`
- Active Learning 投递

这样做的效果是：

- `analyze.py` 从“自己展开最终收口细节”变成“调用统一 finalize policy”
- 后续 Slice 3 可以继续把 override 条件和 finalize policy 进一步切开

## 测试策略

新增：

- `tests/unit/test_classification_finalization.py`

覆盖：

- branch conflict 聚合
- knowledge payload 回填
- review governance 信号
- contract backfill

回归：

- `test_analyze_dxf_coarse_knowledge_outputs.py`
- `test_analyze_dxf_fusion.py`
- `test_analyze_dxf_hybrid_override.py`
- `similarity / compare / vectors / qdrant` 相关 contract 测试

## 非目标

本次不做：

- 不改 Hybrid override 触发条件
- 不改 Graph2D soft override/gate 规则
- 不重写 Active Learning 入队逻辑
- 不拆出独立 decision service class
