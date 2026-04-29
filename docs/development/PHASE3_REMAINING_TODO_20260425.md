# Phase 3 Remaining TODO

日期：2026-04-25

## 范围假设

本 TODO 基于当前本地仓库状态评估：

- Training Data Governance Phase 1 + 2 已完成。
- Phase 3 的主要方向是 API router / pipeline / decision boundary 收口。
- `analyze.py` 已降到 164 行，当前不是最大风险点。
- `vectors.py` 仍有 654 行，是剩余 Phase 3 router 收口的主要工作区。
- 当前分支 `phase3-vectors-crud-router-20260422` 已包含 PR #468 的 CRUD router 拆分与 OpenAPI operationId 修复；远端合并状态当前不在本地可验证范围内。

## 剩余开发量估算

如果目标是完成 Phase 3 router 收口，不含更大规模 decision service 抽象：

- 预计剩余：3-4 个小 PR。
- 预计开发量：1.5-2.5 个工作日。
- 主要风险：OpenAPI operationId 漂移、老测试 monkeypatch 面变化、`vectors.py` 内共享 helper 被过早移动。

如果把 decision service 架构治理也纳入同一阶段：

- 预计追加：1-2 个 PR。
- 预计追加开发量：1-2 个工作日。
- 主要风险：需要重新定义 classify/analyze 的稳定输入输出边界，测试面会比 router 拆分更宽。

综合估算：

- Phase 3 router 收口：约 60%-70% 已完成。
- Phase 3 加 decision service 收口：约 45%-55% 已完成。

## TODO 列表

| 优先级 | 任务 | 当前状态 | 交付物 | 完成标准 | 建议验证 |
| --- | --- | --- | --- | --- | --- |
| P0 | PR #468 收口 | 本地修复已提交；远端状态需网络确认 | 合并 `phase3-vectors-crud-router-20260422` | checks 通过且 PR 合并 | `gh pr checks 468`、`gh pr view 468` |
| P1 | `vectors` list router 拆分 | 未开始 | `src/api/v1/vectors_list_router.py`，可选 `vector_list_models.py` | `/api/v1/vectors/` path、response model、operationId 不变 | OpenAPI contract + list delegation tests |
| P1 | `vectors` batch similarity router 拆分 | pipeline 已抽取，router 未拆 | `src/api/v1/vectors_similarity_router.py`，可选 `vector_similarity_models.py` | `/api/v1/vectors/similarity/batch` 行为不变 | batch similarity regression + route uniqueness |
| P1 | `vectors` backend reload admin router 拆分 | pipeline 已抽取，router 未拆 | `src/api/v1/vectors_admin_router.py` | `/api/v1/vectors/backend/reload` admin token、metrics、error contract 不变 | reload delegation + auth failure metrics tests |
| P2 | `vectors.py` helper ownership 清理 | 部分 helper 仍留在 router 文件 | `src/core/vector_filtering.py` 或保留 facade + shared core helper | `vectors.py` 只保留兼容 patch facade 与 router include | targeted unit + OpenAPI snapshot |
| P2 | route ownership guard | 未开始 | 测试或文档约束新增 route 不回流大文件 | 新增路由必须进入 split router | route decorator count / ownership test |
| P3 | decision service 抽象 | 方向明确，优先级低于 router 收口 | `src/core/decision_service.py` 或 equivalent | `HybridClassifier.classify()` 之外形成稳定 decision boundary | classify/analyze integration + contract tests |

## 推荐执行顺序

1. 先合并 PR #468，避免后续 router 拆分叠在未合并分支上。
2. 拆 `vectors` list router，因为它仍牵涉 memory/redis/qdrant 读取 helper，是剩余 router 中最容易产生 patch 面漂移的一块。
3. 拆 batch similarity router，风险较低，主要保护 request/response models 与 operationId。
4. 拆 backend reload router，重点保护 admin token dependency 与 reload metrics。
5. 最后做 helper ownership 清理，不要提前移动共享 helper，避免把小 PR 变成大范围测试修复。
6. decision service 放在 Phase 3 router 收口之后单独立项。

## 暂不建议做的事

- 不建议为了消除 OpenAPI snapshot mismatch 直接刷新 `config/openapi_schema_snapshot.json`，除非确认是有意 API 变更。
- 不建议一次性把 `vectors.py` 的 helper 全部移入 core，因为现有测试依赖 `src.api.v1.vectors.*` monkeypatch 面。
- 不建议在 PR #468 未合并前继续开多个重叠 router 拆分分支，后续冲突成本会明显增加。

