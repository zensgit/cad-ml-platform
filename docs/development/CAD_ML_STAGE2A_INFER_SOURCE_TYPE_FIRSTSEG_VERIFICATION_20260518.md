# Stage 2a — `_infer_source_type` First-Segment — Verification

Date: 2026-05-18
Companion: `CAD_ML_STAGE2A_INFER_SOURCE_TYPE_FIRSTSEG_DEVELOPMENT_20260518.md`

OCC-free; 本地静态 + 行为自测可完全验证（scaffolder 无 OCC 依赖）。CI 3.11 权威。

## 1. 静态

```bash
python3 -m py_compile \
  scripts/build_brep_golden_manifest_skeleton.py \
  tests/unit/test_build_brep_golden_manifest_skeleton.py
```

期望 exit 0。（已本地 ✅）

## 2. 单元级 `_infer_source_type` 真值

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import _infer_source_type
assert _infer_source_type(['internal','x'])      == ('internal', True)
assert _infer_source_type(['mystery','public_nc'])== ('internal', False)
assert _infer_source_type(['mystery','internal']) == ('internal', False)
assert _infer_source_type(['vendor','acme'])      == ('vendor', True)
assert _infer_source_type([])                     == ('internal', False)
assert _infer_source_type(['public_nc'])          == ('public_nc', True)
print('OK first-segment-only truth table')
"
```

期望 `OK first-segment-only truth table`。（已本地 ✅）

关键两行：`['mystery','public_nc'] -> ('internal', False)` 与
`['mystery','internal'] -> ('internal', False)` —— 深层桶名被忽略，
返回 `clean=False` 触发调用方 `TODO-source-type`。

## 3. skeleton 集成行为

```bash
python3 -c "
import sys, tempfile, os; sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import build_skeleton
d=tempfile.mkdtemp()
def t(r):
    p=os.path.join(d,r); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write('x')
t('mystery/public_nc/x.step'); t('mystery/internal/y.step'); t('loose.step'); t('internal/ok.step')
b={c['id']:c for c in build_skeleton(d)['cases']}
assert b['mystery_public_nc_x']['source_type']=='internal'
assert 'TODO-source-type' in b['mystery_public_nc_x']['tags']
assert b['mystery_public_nc_x']['release_eligible'] is False  # hard gate
assert b['mystery_internal_y']['source_type']=='internal'
assert 'TODO-source-type' in b['mystery_internal_y']['tags']
assert b['mystery_internal_y']['release_eligible'] is False   # default != clean
assert b['loose']['source_type']=='internal' and 'TODO-source-type' in b['loose']['tags']
assert b['loose']['release_eligible'] is False                # root-level not clean
assert b['internal_ok']['source_type']=='internal' and 'TODO-source-type' not in b['internal_ok']['tags']
assert b['internal_ok']['release_eligible'] is True            # clean inference
print('OK nested/root NOT release_eligible; clean inference eligible')
"
```

期望 `OK ...`。（已本地 ✅）

## 4. 现有 Stage 2a 测试无回归（重放断言）

```bash
python3 -c "
import sys, tempfile, os; sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import build_skeleton, summarize
from scripts.validate_brep_golden_manifest import validate_manifest
d=tempfile.mkdtemp()
def t(r):
    p=os.path.join(d,r); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write('ISO-10303-21;\n')
t('internal/bracket_a.step'); t('public_nc/abc_00001.step'); t('vendor/acme/shaft.stp')
b={c['id']:c for c in build_skeleton(d)['cases']}
assert b['vendor_acme_shaft']['source_type']=='vendor'   # 首段即桶，不变
d3=tempfile.mkdtemp()
def t3(r):
    p=os.path.join(d3,r); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write('x')
for i in range(50): t3(f'internal/part_{i}.step')
t3('public_nc/abc_0.step')
m=build_skeleton(d3, manifest_root=d3)
for c in m['cases']:
    c['part_family']='block'; c['license']='internal' if c['source_type']!='public_nc' else 'CC-BY-NC-SA-4.0'
    c['tags']=[t for t in c['tags'] if not t.startswith('TODO-')]  # full human signoff
r=validate_manifest(m, min_release_samples=50)
assert r['status']=='release_ready' and r['release_eligible_count']==50 and r['case_count']==51
print('OK no regression: vendor first-seg, skeleton->fill+signoff->validator round-trip')
"
```

期望 `OK no regression ...`。（已本地 ✅）

## 5. validator example 自洽

```bash
python3 scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.real.example.json --min-release-samples 50 \
  | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['status']=='insufficient_release_samples' and r['release_eligible_count']==1 and r['case_count']==3 and r['errors']==[]; print('OK example self-consistent')"
```

期望 `OK example self-consistent`。（已本地 ✅）

## 5b. Blocking finding 复现 + 修复验证（已本地 ✅）

```bash
python3 -c "
import sys, tempfile, os; sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import build_skeleton
from scripts.validate_brep_golden_manifest import validate_manifest
d=tempfile.mkdtemp()
for i in range(50):
    p=os.path.join(d,f'mystery/public_nc/p{i}.step'); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write('x')
m=build_skeleton(d, manifest_root=d)
for c in m['cases']:
    c['part_family']='block'; c['license']='internal'
assert all(c['release_eligible'] is False for c in m['cases'])
r=validate_manifest(m, min_release_samples=50)
assert r['status']!='release_ready' and r['release_eligible_count']==0
print('OK: 50 mystery/public_nc -> 0 eligible, NOT release_ready (was 50/release_ready)')
"
```

纵深（validator 独立拒绝 release_eligible+TODO）：

```bash
python3 -c "
import sys, tempfile, os; sys.path.insert(0,'.')
from scripts.validate_brep_golden_manifest import validate_manifest
d=tempfile.mkdtemp(); open(os.path.join(d,'p.step'),'w').write('x')
base=dict(id='c',path='p.step',format='step',source_type='internal',release_eligible=True,part_family='block',license='internal',expected_behavior='parse_success',expected_topology=dict(faces_min=1,edges_min=0,solids_min=0,graph_nodes_min=1,surface_types=['plane']))
for mut in ({'tags':['TODO-source-type']},{'license':'TODO'}):
    r=validate_manifest({'schema_version':'brep_golden_manifest.v1','name':'t','root':d,'cases':[{**base,**mut}]}, min_release_samples=1)
    assert r['status']=='invalid' and any('unfilled skeleton placeholders' in e for e in r['errors'])
print('OK: validator rejects release_eligible+TODO (tag & field)')
"
```

人工 signoff 契约（仅填字段不够，须清 tag）：见
`test_skeleton_requires_full_human_signoff_to_pass_validator`。

## 6. CI（合入后权威）

```bash
gh run list --branch main --workflow CI --limit 1 --json conclusion --jq '.[0].conclusion'
# 期望 success；含 test_build_brep_golden_manifest_skeleton.py（+2 用例）全 pass
```

## 7. 反例（应仍正确）

| 输入 | 期望 |
| --- | --- |
| `mystery/public_nc/x.step` | source_type=internal, **TODO-source-type**（深桶不被采纳） |
| `mystery/internal/y.step` | source_type=internal, **TODO-source-type**（即便默认也等于 internal，仍须 flag） |
| `loose.step`（root 级） | source_type=internal, **TODO-source-type** |
| `vendor/acme/shaft.stp` | source_type=vendor, clean（首段命中，不 flag） |

## 8. 一次性总验

```bash
#!/bin/bash
set -e
python3 -m py_compile scripts/build_brep_golden_manifest_skeleton.py tests/unit/test_build_brep_golden_manifest_skeleton.py
python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import _infer_source_type
assert _infer_source_type(['mystery','public_nc'])==('internal',False)
assert _infer_source_type(['mystery','internal'])==('internal',False)
assert _infer_source_type(['vendor','acme'])==('vendor',True)
assert _infer_source_type([])==('internal',False)
"
python3 scripts/validate_brep_golden_manifest.py --manifest config/brep_golden_manifest.real.example.json --min-release-samples 50 >/dev/null
echo "ALL GREEN"
```

期望 `ALL GREEN`。
