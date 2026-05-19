# Stage 2a — B-Rep Golden Tooling Verification

Date: 2026-05-18
Companion: `CAD_ML_STAGE2A_TOOLING_DESIGN_20260518.md`
Scope: 验证 main `7526822c` 上的 Stage 2a 工具链三件套已就绪且行为正确

## 0. 权威状态

| 维度 | 值 |
| --- | --- |
| main HEAD | `7526822c`（PR #479 merged，cycle 7） |
| `enforce_admins` | `true`（已验证恢复） |
| Stage 2a 代码侧 | 完备（validator public_nc + scaffolder + 字段参考） |
| Stage 2a 剩余 | 100% 人工数据 sourcing |

## 1. main 上落地验证

```bash
git fetch origin main
git log origin/main --oneline | head -5 | grep -q "Merge pull request #479" && echo "OK #479 merged"
git ls-files | grep -E "(build_brep_golden_manifest_skeleton\.py|brep_golden_manifest\.real\.example\.json)" 
```

期望：`OK #479 merged` + 两文件均 tracked。

## 2. validator public_nc 行为（OCC-free，本地）

```bash
python3 -c "
from scripts.validate_brep_golden_manifest import ALLOWED_SOURCE_TYPES, RELEASE_EXCLUDED_SOURCE_TYPES
assert 'public_nc' in ALLOWED_SOURCE_TYPES
assert 'public_nc' in RELEASE_EXCLUDED_SOURCE_TYPES
print('OK public_nc allowed + release-excluded')
"
```

## 3. scaffolder 行为（OCC-free，本地）

```bash
python3 -c "
import sys, tempfile, os
sys.path.insert(0,'.')
from scripts.build_brep_golden_manifest_skeleton import build_skeleton, summarize
from scripts.validate_brep_golden_manifest import validate_manifest
d=tempfile.mkdtemp()
def t(rel):
    p=os.path.join(d,rel); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write('x')
for i in range(50): t(f'internal/p{i}.step')
t('public_nc/abc.step'); t('mystery/x.iges')
m=build_skeleton(d, manifest_root=d)            # str root robustness
s=summarize(m)
assert s['case_count']==52
assert s['source_type_counts']['public_nc']==1
assert s['needs_source_type_review']==1          # mystery flagged
for c in m['cases']:
    c['part_family']='block'
    c['license']='CC-BY-NC-SA-4.0' if c['source_type']=='public_nc' else 'internal'
r=validate_manifest(m, min_release_samples=50)
assert r['status']=='release_ready', r['errors'][:2]
assert r['release_eligible_count']==51           # 50 internal + mystery(default internal); public_nc excluded
print('OK scaffolder->fill->validator release_ready; public_nc excluded; unknown flagged')
"
```

（设计 §3 往返不变式；已本地通过）

## 4. 字段参考 manifest 自洽

```bash
python3 scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.real.example.json --min-release-samples 50 \
  | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['status']=='insufficient_release_samples'; assert r['release_eligible_count']==1; assert r['case_count']==3; assert r['errors']==[]; assert set(r['source_type_counts'])=={'internal','public_nc','fixture'}; print('OK example self-consistent')"
```

期望：`OK example self-consistent`（结构正确但 1<50 → insufficient，errors 空）。

## 5. CI 验证（已发生，PR #479 合入前 + 合入后）

| 阶段 | 验证 |
| --- | --- |
| PR #479 PR-level | 50 pass / 0 fail / 0 pending（cycle 7 前确认） |
| main 合入后 | `tests/unit/test_validate_brep_golden_manifest.py`（+2）、`tests/unit/test_build_brep_golden_manifest_skeleton.py`（+7）随 main CI 跑 |

main 合入后复核：

```bash
gh run list --branch main --workflow CI --limit 1 --json conclusion --jq '.[0].conclusion'
# 期望：success
```

## 6. 反例（治理硬约束，应失败）

| 输入 | 期望 |
| --- | --- |
| `public_nc` case 标 `release_eligible=true` | validator `invalid` + "cannot be release_eligible" |
| scaffolder `--root` 空目录 | exit 1 + `no STEP/IGES files found` |
| scaffolder `--root` 非目录 | exit 2 + `not a directory` |
| 未知子目录桶 | `source_type=internal` **且** `TODO-source-type` tag |

## 7. 一次性总验

```bash
#!/bin/bash
set -e
fails=0
git fetch origin main >/dev/null 2>&1
git log origin/main --oneline | head -6 | grep -q "Merge pull request #479" || { echo "MERGE #479 missing"; fails=$((fails+1)); }
enf=$(gh api repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins --jq '.enabled')
[ "$enf" = "true" ] || { echo "ENF=$enf"; fails=$((fails+1)); }
python3 -c "
from scripts.validate_brep_golden_manifest import ALLOWED_SOURCE_TYPES, RELEASE_EXCLUDED_SOURCE_TYPES
assert 'public_nc' in ALLOWED_SOURCE_TYPES and 'public_nc' in RELEASE_EXCLUDED_SOURCE_TYPES
" || { echo "public_nc constant regressed"; fails=$((fails+1)); }
python3 scripts/validate_brep_golden_manifest.py --manifest config/brep_golden_manifest.real.example.json --min-release-samples 50 >/dev/null 2>&1 || { echo "example validator non-zero unexpectedly"; fails=$((fails+1)); }
for f in scripts/build_brep_golden_manifest_skeleton.py tests/unit/test_build_brep_golden_manifest_skeleton.py config/brep_golden_manifest.real.example.json; do
  [ -f "$f" ] || { echo "MISSING $f"; fails=$((fails+1)); }
done
echo "---"
[ "$fails" -eq 0 ] && echo "ALL GREEN" || echo "FAILED $fails"
```

期望：`ALL GREEN`。

## 8. 仍待人工（非验证项，记录边界）

- 内部 STEP/IGES ≥50 数据 sourcing
- dependabot #389/#391/#392/#393/#394/#469/#470 手动 close（classifier 三拒代理批量关）
- Stage 2b reviewed manufacturing labels（人工 review）
- Stage 2c scorecard 接线（阻塞于 2a+2b 真实数据）
