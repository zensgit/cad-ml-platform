# Phase 3 Decision Contract Slice 1 Verification

日期: 2026-04-15

## 变更范围

代码：

- `src/core/classification/decision_contract.py`
- `src/core/classification/__init__.py`
- `src/api/v1/analyze.py`
- `src/core/similarity.py`

测试：

- `tests/unit/test_classification_decision_contract.py`
- `tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py`

## 验证目标

确认以下几点：

1. `analyze` 写侧会稳定补齐最终 decision contract
2. 向量 metadata 导出与 `similarity` 读侧使用同一套 contract 逻辑
3. 现有 Fusion / Hybrid override 行为不回退
4. 现有 vector / similarity / compare 相关 contract 不回退

## 实际执行

### 批次 1

命令：

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_classification_decision_contract.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py
```

结果：

- `5 passed`

### 批次 2

命令：

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_similarity_topk.py \
  tests/unit/test_compare_endpoint.py \
  tests/unit/test_vectors_module_endpoints.py
```

结果：

- `25 passed`

### 批次 3

命令：

```bash
.venv311/bin/python -m pytest -q tests/unit/test_qdrant_vector_store.py
```

结果：

- `25 passed, 2 skipped`

### 批次 4

命令：

```bash
.venv311/bin/python -m pytest -q tests/integration/test_analyze_dxf_fusion.py
.venv311/bin/python -m pytest -q tests/integration/test_analyze_dxf_hybrid_override.py
```

结果：

- `8 passed`
- `3 passed`

## 结果汇总

总计：

- `41 passed`
- `2 skipped`

## 结论

本次 Phase 3 Slice 1 达成：

- `analyze` 最终分类结果新增稳定写入：
  - `decision_source`
  - `final_decision_source`
  - `is_coarse_label`
  - 回落后的 `fine_part_type`
- 向量 metadata 与 `similarity` 读侧已统一到共享 helper
- 现有 Fusion / Hybrid override 测试未回退
- 现有 compare / topk / vectors / qdrant contract 测试未回退

## 备注

- 本次尝试调用 `Claude Code CLI` 做只读 sidecar 审阅；CLI 本机可执行，但非交互 `-p` 审阅在当前登录态下未产出有效结果，因此未将其纳入主验证链。
- 主验证结论完全基于本地 `.venv311` 下的 pytest 结果。
