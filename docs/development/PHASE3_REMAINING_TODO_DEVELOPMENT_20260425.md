# Phase 3 Remaining TODO Development

日期：2026-04-25

## 开发内容

本次变更只新增计划文档，不修改运行时代码。

新增：

- `docs/development/PHASE3_REMAINING_TODO_20260425.md`

该 TODO 文档基于本地仓库状态整理 Phase 3 剩余开发量、优先级、交付物和验证标准。

## 评估依据

- 当前分支：`phase3-vectors-crud-router-20260422`
- 最新本地提交：
  - `1e0d0f60 fix: preserve vectors register operation id`
  - `d75d7e90 refactor: split vectors crud router`
- 关键文件规模：
  - `src/api/v1/analyze.py`：164 行
  - `src/api/v1/vectors.py`：654 行
- `vectors.py` 当前仍保留的主要 route：
  - `GET /api/v1/vectors/`
  - `POST /api/v1/vectors/similarity/batch`
  - `POST /api/v1/vectors/backend/reload`

## 结论

Phase 3 router 收口的剩余开发量主要不是模型能力问题，而是 API 边界治理：

- 继续拆分 `vectors.py` 的剩余 router。
- 保留 OpenAPI operationId 与 response schema。
- 保留现有 `src.api.v1.vectors.*` monkeypatch 兼容面，逐步迁移而不是一次性删除。

