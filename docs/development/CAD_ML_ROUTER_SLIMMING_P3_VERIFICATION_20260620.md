# P3 Router Slimming — Verification

Date: 2026-06-20
Companion: `CAD_ML_ROUTER_SLIMMING_P3_DEVELOPMENT_20260620.md`

**Principle**: 行为保持的 facade 抽取 —— 真正的验证是“路由契约不变 + 既有测试全绿 +
facade re-export 钉死”。下列在 `.venv311`（pytest 7.4.3 / py 3.11）验证通过。

## 1. materials.py slice（已落地）

### 1.1 静态

```bash
python3 -c "import ast; [ast.parse(open(f).read()) for f in ['src/api/v1/materials.py','src/api/v1/materials_models.py']]"
# 旧 router 不再残留 pydantic / 模型定义
grep -n 'pydantic\|BaseModel\|Field' src/api/v1/materials.py   # 期望：无输出
wc -l src/api/v1/materials.py                                  # 期望：886（原 1146）
```

期望：compile OK；无残留；行数下降。（✅）

### 1.2 行为（既有 materials 测试全绿）

```bash
.venv311/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_materials_api.py tests/unit/test_materials_classify.py \
  tests/unit/test_materials_compatibility.py tests/unit/test_materials_cost.py \
  tests/unit/test_materials_equivalence.py tests/unit/test_materials_export.py
```

期望：**183 passed**（HTTP 契约、cost/classify/compatibility/equivalence/export 行为不变）。（✅）

### 1.3 facade-compat（新增钉死）

```bash
.venv311/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_materials_models_facade.py
```

期望：**3 passed** —— 34 个模型仍可从 `src.api.v1.materials` import、与
`materials_models` 同一对象、`__all__ == ["router"]` 不变。（✅）

### 1.4 路由顺序未变

模型抽取不动 handler，`/{grade}` 贪婪 catch-all 仍在最后（`router.routes` 数量与顺序
不变，23 条路由）。

## 2. health.py / dedup.py slice（计划中）

每个后续 slice 的验证模板：
1. `py_compile` 抽出的 core 模块 + 瘦身后的 router。
2. 该 router 既有测试全绿（行为保持）。
3. facade-compat 测试：被搬走的符号（含 dedup 的 5 个 Redis monkeypatch 目标、
   `_check_forced_async`、13 个模型、health 的 `hybrid_runtime_config` /
   `provider_registry_health`）仍可从原 router 模块 import 且对象一致。
4. CI 全套绿（真正绑定验证）。

## 3. 结论

materials.py slice 行为保持、可验证、facade 钉死，router 由 1146 → 886 行。health /
dedup 的安全切面已并行分析出（见 DEVELOPMENT §2），风险块明确不抽，按 smallest-safest-first
逐 slice 推进。
