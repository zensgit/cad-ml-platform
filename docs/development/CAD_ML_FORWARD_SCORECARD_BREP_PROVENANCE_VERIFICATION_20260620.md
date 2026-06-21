# Forward Scorecard B-Rep Provenance Consumption — Verification

Date: 2026-06-20
Companion: `CAD_ML_FORWARD_SCORECARD_BREP_PROVENANCE_DEVELOPMENT_20260620.md`

**Principle**: forward_scorecard 是 import-light（纯逻辑）；wrapper 是 bash。本地静态 +
行为自测 + pytest 即可验。下列在 `.venv311`（pytest 7.4.3 / py 3.11）下验证通过。

## 1. 静态

```bash
python3 -c "import ast; ast.parse(open('src/core/benchmark/forward_scorecard.py').read())"
python3 -c "import ast; ast.parse(open('scripts/export_forward_scorecard.py').read())"
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

期望：exit 0。（已本地验证 ✅）

## 2. 行为自测（component 级，本地可跑）

```bash
.venv311/bin/python -c "
from src.core.benchmark.forward_scorecard import _brep_component, _attach_brep_manifest_validation
# eval 自身 release_ready
comp = _brep_component({'sample_size':50,'parse_success_count':50,'graph_valid_count':50})
assert comp['status']=='release_ready'
# validator: 样本够但 verified floor 未满足（同义反复 derived 集合）
_attach_brep_manifest_validation(comp, {'status':'insufficient_verified_topology',
  'ready_for_release':False,'verified_topology_count':0,'derived_topology_count':50,
  'verified_topology_floor':10,'verified_topology_floor_met':False,'release_eligible_count':50})
assert comp['status']=='benchmark_ready_with_gap'
assert 'topology_verified_below_release_floor' in comp['evidence_gaps']
# 空 report -> no-op（向后兼容）
c2 = _brep_component({'sample_size':50,'parse_success_count':50,'graph_valid_count':50})
_attach_brep_manifest_validation(c2, {})
assert c2['status']=='release_ready' and 'manifest_validation' not in c2
print('OK component downgrade + backward-compat')
"
```

期望：打印 OK。（✅）

## 3. pytest（CI 等价，3.11）

```bash
.venv311/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_forward_scorecard_missing_component_blocked.py \
  tests/unit/test_forward_scorecard_metric_status_truth_table.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

期望：全绿。新增覆盖：

- **端到端门**（不只 helper）：所有 component release_ready + brep_manifest_validation
  非 release-ready（`invalid` / `insufficient_release_samples` /
  `insufficient_verified_topology` 三态）→ `overall_status` 降为
  `benchmark_ready_with_gap`，brep component 同降。
- release-ready validator report → brep 保持 release_ready。
- 缺省（无 validation）→ 向后兼容，不挂 `manifest_validation`。
- markdown 暴露 `manifest=<status>; verified=<v>/<eligible>`。
- **wrapper 端到端**：`brep_golden_manifest.outputs.validation_json` 经 wrapper 传到
  `--brep-manifest-validation-summary`，brep component 反映降级（refinement 2）。

## 4. 绑定边界确认（设计，非缺陷）

`scripts/export_benchmark_release_decision.py` 的 `_decision` 只在 `CRITICAL_STATUSES`
/ `REVIEW_STATUSES` 上挡，`benchmark_ready_with_gap` 不在其中 → release 决策 artifact
有意容忍 B-Rep gap。本切片 scorecard-only，不改该 artifact。Hard release-decision
gating 见 DEVELOPMENT §2 的未来 policy 触发条件。

## 5. 结论

scorecard B-Rep component 现在消费 validator provenance：eval 看似绿、但 manifest 非
release-ready 时，brep component 与 scorecard `overall_status` 都降级并暴露
verified-floor gap。release 决策 artifact 的 hard-gating 是单独的、有触发条件的 policy
后续，不在本切片。
