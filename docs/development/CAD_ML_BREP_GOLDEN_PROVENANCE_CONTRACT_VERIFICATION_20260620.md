# B-Rep Golden Manifest Provenance Contract — Verification

Date: 2026-06-20
Companion: `CAD_ML_BREP_GOLDEN_PROVENANCE_CONTRACT_DEVELOPMENT_20260620.md`

**Principle**: validator / scaffolder 是 stdlib-only（不依赖 OCC），本地静态 + 行为自测
即可验；CI 用 3.11 跑 pytest。下列命令在 `.venv311`（pytest 7.4.3 / py 3.11）下验证通过。

## 1. 静态

```bash
python3 -c "import ast; ast.parse(open('scripts/validate_brep_golden_manifest.py').read())"
python3 -c "import ast; ast.parse(open('scripts/build_brep_golden_manifest_skeleton.py').read())"
python3 -c "import json; json.load(open('config/brep_golden_manifest.real.example.json'))"
```

期望：exit 0。（已本地验证 ✅）

## 2. 行为自测（OCC-free，本地可跑）

### 2.1 向后兼容：两个示例 manifest 仍有效

```bash
python3 scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.example.json --allow-missing-files
python3 scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.real.example.json
```

期望两者均 `status=insufficient_release_samples`、`errors=[]`（结构正确但未到 50 样本下限）。
`real.example.json` 还应报 `release_eligible_count=1`、`verified_topology_count=1`。（✅）

### 2.2 双轴 biconditional —— 用户复现用例现在报错

```bash
python3 -c "
from scripts.validate_brep_golden_manifest import validate_manifest
m={'schema_version':'brep_golden_manifest.v1','name':'repro','cases':[{
  'id':'x','path':'x.step','format':'step','source_type':'public_cad',
  'license_status':'non_commercial','release_eligible':False,'part_family':'b',
  'license':'cc','expected_behavior':'parse_success',
  'expected_topology':{'faces_min':1,'edges_min':0,'solids_min':0,'graph_nodes_min':1}}]}
r=validate_manifest(m,min_release_samples=1,allow_missing_files=True)
assert r['status']=='invalid', r['status']
assert any('requires source_type \`public_nc\`' in e for e in r['errors']), r['errors']
print('OK biconditional reverse')
"
```

期望：`invalid` + `license_status \`non_commercial\` requires source_type \`public_nc\``。（✅）

### 2.3 verified floor 同义反复门 —— 50 个全 `derived` 不得 release_ready

```bash
python3 -c "
from scripts.validate_brep_golden_manifest import validate_manifest
base=lambda i:{'id':f'd{i}','path':f'd{i}.step','format':'step','source_type':'internal',
  'release_eligible':True,'part_family':'blk','license':'internal','license_status':'internal',
  'license_source':'internal-dataset://x','topology_source':'derived','expected_behavior':'parse_success',
  'expected_topology':{'faces_min':1,'edges_min':0,'solids_min':0,'graph_nodes_min':1}}
m={'schema_version':'brep_golden_manifest.v1','name':'derived','cases':[base(i) for i in range(50)]}
r=validate_manifest(m,min_release_samples=50,allow_missing_files=True)
assert r['status']=='insufficient_verified_topology', r['status']
assert r['ready_for_release'] is False and r['verified_topology_count']==0
print('OK derived-only blocked, floor=',r['verified_topology_floor'])
"
```

期望：`insufficient_verified_topology`、`ready_for_release=false`、`verified_topology_floor=10`。（✅）

## 3. pytest（CI 等价，3.11）

```bash
.venv311/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_validate_brep_golden_manifest.py \
  tests/unit/test_build_brep_golden_manifest_skeleton.py \
  tests/unit/test_brep_golden_manifest_ci_wrapper.py
```

期望：全绿（合约逐条覆盖：license_status 必填/枚举/source 规则、双向一致性、verified 牙齿、
derived 计入 floor、verified floor 边界、real.example 钉死）。

邻居回归（未改动，安全网）：

```bash
.venv311/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_eval_brep_step_dir.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

期望：全绿（本切片未触及 eval 脚本与 forward scorecard —— 后者由 PR-B 接管）。

## 4. CI

- PR #485 GitHub checks 全绿（`tests (3.10)` / `tests (3.11)` / `core-fast-gate` /
  `unit-tier` 等）。
- `brep-golden-eval.yml`（OCC-provisioned）为 manual/scheduled，本切片不改其触发；
  默认 informational，不因示例 manifest reds 调度运行。

## 5. 结论

合约就位且可信：release floor 现在要求**可审计的 license** 与**人工核验的 topology 下限**，
`derived`-only 的同义反复集合在 validator 自身的 `release_ready` 处即被挡住。剩余绑定面
（forward scorecard / release 决策消费分层计数）由 PR-B 接续 —— 见 DEVELOPMENT §3。
