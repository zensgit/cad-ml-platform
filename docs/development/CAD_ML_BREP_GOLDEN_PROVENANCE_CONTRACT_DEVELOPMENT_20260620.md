# B-Rep Golden Manifest Provenance Contract — Development

Date: 2026-06-20
Stage: 2a (golden-set 证据合约强化)
Companion: `CAD_ML_BREP_GOLDEN_PROVENANCE_CONTRACT_VERIFICATION_20260620.md`
Decision context: `CAD_ML_BREP_GOLDEN_MANIFEST_DEVELOPMENT_20260512.md`（原始合约）、
`CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` §Phase 4
PR: #485（文档与代码同 PR，遵循 Phase 0 line 29 的"文档同实现切片"约定）

## 0. 为什么

原始 manifest validator（`CAD_ML_BREP_GOLDEN_MANIFEST_*`）只验证**结构**：字段在不在、
`public_nc` 没被误标 `release_eligible`、release case 没留 TODO 占位。它无法回答两个对
"发布可信度"更关键的问题：

1. **license 是否可审计** —— release-eligible case 的 `license` 是自由文本，validator
   只查非空、非 TODO，从不核验它声称的授权是否成立。50 个 case 全标 `license: "internal"`
   也能过 release floor。
2. **topology 下限是不是真值** —— `expected_topology.faces_min` 等只校验 `int >= 0`。
   若这些下限是用**被测同一个 OCC parser** 自动反推（`derived`）出来的，那 golden set 就是
   同义反复（tautological）：parser 自己生成期望值、再用期望值验自己，永远"绿"。

结论（早期分析交付的一句话）：**release_ready ≠ 可信证据**。本切片把"漂亮的绿"升级为
"可归因、可分层、防同义反复的绿"。

诚实边界：validator **无法核验** license / topology 声称是否*为真*，只能强制声称是
**存在的、格式合法的、可归因到具名来源的**。这是 provenance **capture**，不是
verification —— 目的是让"假绿"**可追责、不静默**。

## 1. 设计

新增字段对 schema 全部**可选**（`schema_version` 保持 `v1`），仅对 `release_eligible`
case **必填** —— 非 release 的 fixture / public_nc / 失败行（及既有 `example.json`）保持有效。

| 维度 | 决策 |
| --- | --- |
| `license_status` | 受控词表 6 值：`internal` / `public_domain` / `permissive` / `proprietary_authorized` / `non_commercial` / `unverified`。字段存在即校验枚举（拼错 = schema error）；release case 必填 |
| release-usable | `{internal, public_domain, permissive, proprietary_authorized}` —— 可计入 release floor |
| release-excluded | `{non_commercial, unverified}` —— 强制 `release_eligible=false` |
| `license_source` | **所有** release-usable status（含 `internal`）必须提供非空 `license_source`（可审计引用 / 授权记录）—— 最强可归因立场：自有数据也需具名 dataset 指针。`non_commercial` / `unverified` 本就 release-excluded，不在此列 |
| 双轴一致性 | `public_nc`（source 轴）⟺ `non_commercial`（license 轴）是同一 NonCommercial 事实，强制**双向**校验。禁止两条排除轴静默分叉（如 `public_cad + non_commercial` 之前 `errors=[]`） |
| `topology_source` | `verified`（人工核验）/ `derived`（parser 反推）。release case 必填 |
| verified 牙齿 | `verified` 必须有非空 `topology_evidence` **且** `faces_min > 0`（仅 faces —— 曲面件 `solids_min=0` 合法）。否则"verified"只是改名贴标，floor 可被刷绿 |
| **verified floor** | release-eligible 中至少 `max(10, ceil(0.2 * eligible))` 个为 `verified`，**封顶 N**（不能要求多于存在数；N≥50 生产档封顶不生效，等价 spec 公式）。不足 → status `insufficient_verified_topology`，`release_ready=false` |
| floor 绑定位置 | **留在 validator 的 `release_ready`** —— 绑定门禁位置，更安全。`derived` 仍计入 release floor，但拉不动 verified floor |
| TODO 防御扩展 | 既有 TODO-field gate 增加 `license_source` / `topology_evidence` —— 半填的字面 `"TODO"` 也算未签收 |
| scaffolder | skeleton 产出占位：`license_status=unverified`（public_nc → `non_commercial`）、`topology_source=derived`、`license_source="TODO"`，使 skeleton 可一路签收到绿 |

## 2. 输出（validator report 新增分层计数）

- `verified_topology_count` / `derived_topology_count`
- `verified_topology_floor` / `verified_topology_floor_met`
- `license_status_counts`（按 status 分层）

这些是 forward scorecard B-Rep component 后续要消费的字段（见 §3 PR-B）。

## 3. 边界 / 未做（并行后续流）

| 流 | 内容 | 状态 |
| --- | --- | --- |
| **PR-B（紧接）** | forward scorecard `_brep_component` 消费 validator report 的分层计数，让 release 决策（`export_benchmark_release_decision.py` → `pick("brep")`）也感知 provenance —— 双层防御 | 待办 |
| P1-data | manifest 填 50–100 真实 STEP/IGES + 人工 license/topology 签收（Phase 4 line 237） | 人工瓶颈 |
| P4 | manufacturing reviewed labels（Phase 6 line 384），与 P1 并行 | 人工瓶颈 |
| P3 | router 瘦身（materials → health → dedup） | 后台填空 |

## 4. 兼容性

- 字段可选 + 仅 release case 必填 + `schema_version=v1` → 既有 `example.json` 不变
  （`insufficient_release_samples`, `errors=[]`）。
- `real.example.json`（real-data field reference）随合约同步更新：release case 补全
  provenance，并加 CLI 测试钉住 `errors=[]`，防止示例无声 rot。
- scaffolder 既有签收测试同步：full-signoff 现包含 provenance（license_status +
  source + verified topology + evidence）。

## 5. 涉及文件

- `scripts/validate_brep_golden_manifest.py` —— 合约 + 分层计数 + verified floor
- `scripts/build_brep_golden_manifest_skeleton.py` —— provenance 占位
- `config/brep_golden_manifest.real.example.json` —— 同步示例
- `tests/unit/test_validate_brep_golden_manifest.py` —— 合约逐条覆盖 + 边界 + 计数
- `tests/unit/test_build_brep_golden_manifest_skeleton.py` —— 签收链
- `tests/unit/test_brep_golden_manifest_ci_wrapper.py` —— release-ready fixture 同步
