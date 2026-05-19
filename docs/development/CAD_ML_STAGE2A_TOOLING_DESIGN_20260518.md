# Stage 2a — B-Rep Golden Tooling Design

Date: 2026-05-18
Status: 代码侧完备（main `7526822c`）；剩余 = 人工数据 sourcing
Single-point reference — 把散落在 commit message / NEXT_STAGES §4.4 /
scaffolder DEV·VERIFICATION / validator 的设计决策串成一张图。

## 1. 问题

Stage 2a release gate 需要 ≥50 个真实 release-eligible STEP/IGES，
组织成 `brep_golden_manifest.v1`。两个真实约束塑造了设计：

- **数据是人工瓶颈**：内部历史交付件需人去收集；公开数据有 license 雷区
- **本地无 OCC**：拓扑真实下限只能在 CI（OCC-provisioned）反推

## 2. 三个设计决策（及为什么）

### 2.1 存储 = 不用 LFS

| | |
| --- | --- |
| 原计划假设 | 单 STEP 5–50MB → 需 LFS / 外部存储 |
| 实测推翻 | 真实零件级 STEP = 64KB–433KB（`tests/fixtures/as1_oc_214.stp`=433KB） |
| 决策 | 50–100 零件 ≈ 5–150MB，**直接进 git**。LFS 反给每个 CI workflow 加 `lfs pull` + 配额，净负 |
| 例外 | 装配体级（>5MB）文件若日后引入，单独评估 |

**教训**：执行前用实测数据校准计划假设。一条 `ls -la tests/fixtures/*.stp` 推翻了整个 LFS 子工作。

### 2.2 数据源 = 内部 release-eligible + ABC 仅 coverage

| 源 | 角色 | source_type | release_eligible |
| --- | --- | --- | --- |
| 内部历史交付 | release 基准 | `internal` | ✅ true（用户确认 ≥50 可凑） |
| ABC Dataset (CC-BY-NC-SA) | 覆盖率补丁 | `public_nc`（新增） | ❌ false（强制） |
| repo fixture | smoke | `fixture` | ❌ false |

**为什么 `public_nc`**：ABC 的 NonCommercial 子句使其不能作为商业产品的
release-gating 基准。但它对"解析器/拓扑覆盖率"仍有价值。新建 source_type
`public_nc`，同时进 `ALLOWED_SOURCE_TYPES`（合法）与
`RELEASE_EXCLUDED_SOURCE_TYPES`（不计 50-floor）。`public_nc` 误标
`release_eligible=true` → validator `invalid`（硬拒，防误用）。

### 2.3 拓扑探测 = 不在脚手架做

OCC 是 CI-only。scaffolder 出保守占位 `expected_topology` + `TODO-topology`
tag；真实下限在 `brep-golden-eval.yml`（OCC-provisioned）反推。脚手架与
OCC 严格解耦 → scaffolder 本地可跑、可单测、零 OCC 依赖。

## 3. 三件套如何协作

```
data/brep_golden/
  internal/*.step      ─┐
  public_nc/*.step     ─┤
                        │  ① build_brep_golden_manifest_skeleton.py --root
                        ▼     扫目录 → 结构正确骨架（source_type 按子目录推断，
                        │     release_eligible 按 RELEASE_EXCLUDED 派生，
                        │     part_family/license/topology = TODO 占位）
   config/brep_golden_manifest.real.json (skeleton)
                        │  ② 人工补 3 类字段（grep '"TODO"' / tag TODO-*）
                        ▼
   config/brep_golden_manifest.real.json (filled)
                        │  ③ validate_brep_golden_manifest.py
                        ▼     --min-release-samples 50 --fail-on-not-release-ready
   exit 0  ⇒  ④ gh workflow run brep-golden-eval.yml (OCC eval, 反推真实 topology)
```

`config/brep_golden_manifest.real.example.json` = 填好的 3-case 字段参考
（internal/public_nc/fixture 各一），自洽验证：1 release-eligible <50 →
`insufficient_release_samples` + `errors=[]`（结构对但样本不足的中间态）。

## 4. 职责边界（严格分离，不重叠）

| 工具 | 唯一职责 | 不做 |
| --- | --- | --- |
| `build_brep_golden_manifest_skeleton.py` | 扫目录→骨架 | 不探拓扑/不猜 license/不下数据 |
| `validate_brep_golden_manifest.py` | 校验填好的 manifest | 不生成/不评测 |
| `scripts/ci/build_brep_golden_manifest_optional.sh` | CI wrapper（validate+eval） | 不构造 |
| `eval_brep_step_dir.py` | OCC 真实解析+拓扑评测 | 不校验 schema |

## 5. 人工剩余工作（代码无法代劳）

1. 收集内部 STEP/IGES ≥50 → `data/brep_golden/internal/`
2.（可选）ABC 子集 → `data/brep_golden/public_nc/`
3. 跑 scaffolder → 骨架
4. 补每 case `part_family` / `license` / 真实 `expected_topology`
5. validator exit 0 → push → dispatch `brep-golden-eval.yml`
6. CI 反推真实 topology 下限 → 回填收紧 → 再校验

## 6. 与下游 Stage 的衔接

- Stage 2a 产出 `reports/benchmark/brep_step_iges_golden/summary.json`
- Stage 2c 用它作 `--brep-summary` 喂 `export_forward_scorecard.py`
- **防假绿**：Stage 3 已加 `test_forward_scorecard_missing_component_blocked`
  ——空 brep summary → scorecard `blocked`，不会默认绿（main `b026fd8b`）

## 7. 不变式（治理）

- `public_nc` 永不计入 release floor（validator 硬拒误标）
- 未知子目录桶 → 默认 `internal` **但** `TODO-source-type` tag（不静默错分）
- 脚手架零 OCC 依赖（本地可验证）
- 无新 LFS / `.gitattributes` 改动
