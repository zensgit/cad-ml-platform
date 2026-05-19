# B-Rep Golden Manifest Skeleton Scaffolder — Verification

Date: 2026-05-18
Companion: `CAD_ML_BREP_GOLDEN_SKELETON_SCAFFOLDER_DEVELOPMENT_20260518.md`

**Principle**: 本地 OCC-free 静态 + 行为自测可验证（脚本不依赖 OCC）；
CI 用 3.11 跑 pytest。

## 1. 静态

```bash
python3 -m py_compile \
  scripts/build_brep_golden_manifest_skeleton.py \
  tests/unit/test_build_brep_golden_manifest_skeleton.py
```

期望：exit 0。（已本地验证 ✅）

## 2. 行为自测（OCC-free，本地可跑）

```bash
python3 -c "
import sys, tempfile, os
sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import build_skeleton, summarize
from scripts.validate_brep_golden_manifest import validate_manifest
d=tempfile.mkdtemp()
def t(rel):
    p=os.path.join(d,rel); os.makedirs(os.path.dirname(p),exist_ok=True)
    open(p,'w').write('ISO-10303-21;\nEND-ISO-10303-21;\n')
for i in range(50): t(f'internal/part_{i}.step')
t('public_nc/abc_0.step'); t('mystery/x.iges')
m=build_skeleton(d, manifest_root=d)
s=summarize(m)
assert s['case_count']==52
assert s['source_type_counts']['public_nc']==1
assert s['needs_source_type_review']==1
for c in m['cases']:
    c['part_family']='block'
    c['license']='CC-BY-NC-SA-4.0' if c['source_type']=='public_nc' else 'internal'
r=validate_manifest(m, min_release_samples=50)
assert r['status']=='release_ready', (r['status'], r['errors'][:3])
assert r['release_eligible_count']==51
assert r['case_count']==52
print('ok')
"
```

期望：`ok`。（已本地验证 ✅ — str-root robustness 亦验证）

## 3. CLI smoke

```bash
mkdir -p /tmp/brep_demo/internal /tmp/brep_demo/public_nc
printf 'ISO-10303-21;\nEND-ISO-10303-21;\n' > /tmp/brep_demo/internal/demo.step
printf 'ISO-10303-21;\nEND-ISO-10303-21;\n' > /tmp/brep_demo/public_nc/abc.step
python3 scripts/build_brep_golden_manifest_skeleton.py \
  --root /tmp/brep_demo --output-json /tmp/brep_demo_skeleton.json
echo "exit=$?"
python3 -c "import json; d=json.load(open('/tmp/brep_demo_skeleton.json')); print(d['schema_version']); print([c['source_type'] for c in d['cases']])"
```

期望：
- `exit=0`
- `schema_version` = `brep_golden_manifest.v1`
- source_types 含 `internal` 与 `public_nc`
- stderr summary `release_eligible_count=1`（public_nc 排除）

（已本地验证 ✅）

## 4. 关键不变式（断言矩阵）

| 不变式 | 测试用例 |
| --- | --- |
| `source_type` 从首段路径推断 | `test_skeleton_infers_source_type_from_first_path_segment` |
| `public_nc` 自动 `release_eligible=False` | 同上 |
| 未知桶 → 默认 internal **且** flag | `test_unknown_bucket_defaults_internal_but_flagged` |
| TODO 字段是显式占位 | `test_todo_fields_are_explicit_placeholders` |
| 重名 stem 唯一 id | `test_duplicate_stems_get_unique_ids` |
| 空目录 0 case | `test_empty_root_yields_zero_cases` |
| summary 计数正确 | `test_summary_counts_release_eligible_and_todo_source` |
| 骨架→填→真实 validator release_ready | `test_skeleton_is_structurally_valid_against_real_validator` |

## 5. CI 验证（合入后）

PR #479（含 validator public_nc + 本 scaffolder）合入 main 后：

```bash
gh run view <ci_run_id> --json jobs \
  --jq '.jobs[] | select(.name | test("unit|pytest|tests")) | {name, conclusion}'
```

期望含 `test_build_brep_golden_manifest_skeleton.py` 与
`test_validate_brep_golden_manifest.py` 全 pass。

## 6. 反例（应失败 / 应被 flag）

| 输入 | 期望 |
| --- | --- |
| `public_nc/x.step` 手工改 `release_eligible=true` 后跑 validator | validator `invalid` + "cannot be release_eligible"（由 validator 的 public_nc 测试覆盖） |
| 空 `--root` | exit 1 + `no STEP/IGES files found` warning |
| `--root` 非目录 | exit 2 + `not a directory` |

## 7. 一次性总验

```bash
#!/bin/bash
set -e
python3 -m py_compile scripts/build_brep_golden_manifest_skeleton.py \
  tests/unit/test_build_brep_golden_manifest_skeleton.py
python3 -c "
import sys, tempfile, os
sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import build_skeleton, summarize
from scripts.validate_brep_golden_manifest import validate_manifest
d=tempfile.mkdtemp()
def t(rel):
    p=os.path.join(d,rel); os.makedirs(os.path.dirname(p),exist_ok=True)
    open(p,'w').write('x')
for i in range(50): t(f'internal/p{i}.step')
t('public_nc/a.step')
m=build_skeleton(d, manifest_root=d)
for c in m['cases']:
    c['part_family']='block'; c['license']='internal'
r=validate_manifest(m, min_release_samples=50)
assert r['status']=='release_ready', r['errors'][:2]
assert r['release_eligible_count']==50
print('ALL GREEN')
"
```

期望：`ALL GREEN`。
