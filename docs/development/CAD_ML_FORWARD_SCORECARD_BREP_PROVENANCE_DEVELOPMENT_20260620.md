# Forward Scorecard B-Rep Provenance Consumption — Development

Date: 2026-06-20
Stage: 2c (forward scorecard 消费 provenance)
Companion: `CAD_ML_FORWARD_SCORECARD_BREP_PROVENANCE_VERIFICATION_20260620.md`
Decision context: `CAD_ML_BREP_GOLDEN_PROVENANCE_CONTRACT_DEVELOPMENT_20260620.md` (#485),
`CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` §Phase 4

## 0. 为什么

#485 让 validator 报出 license/topology provenance 分层计数，并在 validator 自身的
`release_ready` 里挡住同义反复（`derived`-only）集合。但 forward scorecard 的
`_brep_component` 只读 **eval summary**（parser 在文件上跑出来的 parse/graph 计数）——
它结构上看不到 provenance（provenance 是 *manifest* 属性，不是 eval 输出）。

结果：一个 eval 看起来 `release_ready`、但 manifest 实际 `insufficient_verified_topology`
的运行，scorecard 的 brep component 仍会显示 `release_ready`。本切片把 validator report
喂进 scorecard，让 brep component 与 scorecard `overall_status` 都反映 verified-floor gap。

## 1. 设计

镜像既有 `_attach_manufacturing_review_manifest_validation` 模式（attach → flag → 原地
降级），消费 validator report 的 `ready_for_release`/`status`（**不只** `verified_topology_floor_met`）：

| 维度 | 决策 |
| --- | --- |
| 消费信号 | validator report 的 `ready_for_release`（综合信号）。`invalid` / `insufficient_release_samples` / `insufficient_verified_topology` 三者都使 `ready_for_release=false` |
| 降级 | manifest 非 release-ready 时，brep component 若为 `release_ready` → 降为 `benchmark_ready_with_gap`（不能保持 release_ready） |
| evidence flag | `brep_manifest_not_release_ready:<status>`；`insufficient_verified_topology` 额外发 `topology_verified_below_release_floor`（用户点名的 flag） |
| 附加可见性 | component 挂 `manifest_validation`：status / ready_for_release / verified / derived / floor / floor_met / eligible 计数 |
| markdown | `_component_evidence` 的 brep 分支增列 `manifest=<status>; verified=<v>/<eligible>`，镜像 manufacturing 的 `review_manifest=` |
| 向后兼容 | `build_forward_scorecard` 新增 `brep_manifest_validation`（Optional，默认 None）；空 report → no-op，既有 caller 不变 |

四个落地面：
- `src/core/benchmark/forward_scorecard.py` —— `_attach_brep_manifest_validation` + 参数 + 调用 + markdown
- `scripts/export_forward_scorecard.py` —— 新增 `--brep-manifest-validation-summary`，穿到 `build_forward_scorecard`
- `scripts/ci/build_forward_scorecard_optional.sh` —— 映射 `steps.brep_golden_manifest.validation_json` → `STEP_BREP_GOLDEN_VALIDATION_JSON` → 传 `--brep-manifest-validation-summary`
- unit（`test_forward_scorecard.py`）+ wrapper（`test_forward_scorecard_release_gate.py`）测试

## 2. 绑定边界（重要）

降级到 `benchmark_ready_with_gap` **会**拉低 scorecard 自己的 `overall_status`
（`_overall_status` 要求所有 component 均 `release_ready`）—— 这是本切片的绑定面。

但单独的 release **决策** artifact（`export_benchmark_release_decision.py` 的 `_decision`）
**容忍** brep=`benchmark_ready_with_gap`：它只在 `CRITICAL_STATUSES={blocked,…}` /
`REVIEW_STATUSES` 上挡，`benchmark_ready_with_gap` 不在其中。这是平台**有意**设计——
B-Rep 是"弱、单独报告、不阻塞总发布"的 lane（Phase 3 acceptance）。

因此本 PR **不**改 release 决策（scorecard-only，用户已确认）。Hard release-decision
gating 留作独立的未来 policy 决策，触发条件：真实 B-Rep golden set 达到 release floor
**且** 明确宣布"B-Rep capability 是产品发布必要条件"，届时单独开 policy PR hard-gate
（更清楚、更可审）。

## 3. 兼容性

新增参数可选、空即 no-op，既有 `build_forward_scorecard` / CLI / wrapper 调用不变。
CI wrapper 用既有 `add_if_exists` 约定接 step 输出，并保留 `FORWARD_SCORECARD_` env 覆盖。
