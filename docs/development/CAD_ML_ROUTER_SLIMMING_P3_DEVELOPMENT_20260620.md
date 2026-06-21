# P3 Router Slimming (materials / health / dedup) — Development

Date: 2026-06-20
Stage: P3 (background fill during P1/P4 human-data waits)
Companion: `CAD_ML_ROUTER_SLIMMING_P3_VERIFICATION_20260620.md`
Decision context: `CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` §Phase 1 line 138
(“continue remaining shared helper ownership cleanup only in small slices with
facade compatibility tests”)

## 0. 为什么

数据闭环（B-Rep provenance #485 + scorecard 消费 #486）代码侧已完成；高价值的
P1-data / P4 是人工瓶颈（真实 STEP/IGES、人工 review labels），我无法代劳。按计划，
P3（router 瘦身）正是“在人工等待期做的后台填空”。三个最厚的 router：

| 文件 | 原行数 |
| --- | --- |
| `src/api/v1/dedup.py` | 1518 |
| `src/api/v1/health.py` | 1189 |
| `src/api/v1/materials.py` | 1146 |

策略与 `vectors.py` 一致：**行为保持**地把纯逻辑 / 模型搬到单一职责模块，router 保留
为薄 facade 并 **re-export** 被搬走的符号（既有 import / monkeypatch / `response_model`
引用不变），每个 slice 配 facade-compat 测试。core 模块永不 import fastapi。

## 1. 已落地：materials.py — 模型抽取（slice 1）

materials.py 业务逻辑早已在 `src/core/materials/*`；其臃肿来自 34 个 Pydantic
response/request 模型（行 21–317）。本 slice 用 AST 驱动把这 34 个模型整体搬到
`src/api/v1/materials_models.py`，materials.py 顶部 `from ... import (…34…)` 全部
re-export，并删除随之失效的 `from pydantic import BaseModel, Field`。

- materials.py：1146 → **886** 行（−260）
- facade-compat 测试钉住：34 个模型仍可从 `src.api.v1.materials` import 且与
  `materials_models` 同一对象；`__all__ == ["router"]` 不变
- 路由顺序未动（handlers 未搬）——`/{grade}` 贪婪 catch-all 仍在最后，未触发误匹配

陷阱（来自并行分析，已规避）：不要统一三个 handler 各自不同的 not-found 契约
（200+found=False vs 404）；core 返回 None/dict，facade 抛 HTTPException。

## 2. 计划中（后续 slice，已并行分析出安全切面）

### health.py（~7 个纯逻辑切面，~200 行）
抽到 `src/core/health_*.py`：`compute_hit_ratio`、`compute_cache_tuning_recommendation`、
`build_provider_snapshot`、`extract_provider_plugin_diagnostics`、
`run_provider_health_check`、`compute_model_health_status`、`determine_faiss_status`。
**保持在 handler**（不抽）：`provider_health`（async 编排 + 实时 registry + metrics）、
`faiss_health`（模块全局态 quirk）、v16 classifier 变更类。facade 需 re-export
`hybrid_runtime_config` / `provider_registry_health`（测试直接 import）。

### dedup.py（最复杂，最后做）
安全切面：env helpers（`_check_forced_async` 等，是测试直接 import 的 facade 目标）、
gating 常量/纯函数、13 个 Pydantic 模型（需连带 `_VERSION_GATE_MODES` 常量，
`Dedup2DTenantConfig` 校验依赖它）。**绝不搬**：5 个 Redis 绑定（test monkeypatch
目标）、4 个 Depends provider（测试直接 import）、`dedup_2d_search`（268 行编排）、
precision-L4（234 行状态机，延后）。另发现孤儿 `_run_dedup_2d_pipeline`（无引用，
单独清理）。

## 3. 边界

- 仅做行为保持的 facade 抽取，不改路由路径 / 契约 / 状态码。
- 风险高的块（async/Redis/全局态/precision verifier）保留在 handler，明确不抽。
- 人工瓶颈项（P1-data / P4 真实数据与 label）不在本范围。
- Phase 7（parametric/generative）按计划门控（Phases 2–4 可信为前置），不在本范围。
