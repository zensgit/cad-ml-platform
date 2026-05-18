# CAD ML Forward Roadmap — Next Stages Verification Protocol

Date: 2026-05-17
Companion: `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md`

**Principle**: 每个 Stage 至少包含 (1) 本地 / 静态门禁 (2) CI 门禁 (3) 期望产物路径 (4) 失败模式与回滚。**不允许"我看过了感觉对"作为验证结论**。

## 0. 验证环境约束（不可绕过）

- 本地 `python3` = 3.9.6 → 无法 import FastAPI 路由层 → **本地 pytest 不是权威**；本地仅允许 `py_compile` + YAML safe_load + 纯字符串/JSON 静态校验。
- 项目 CI target = **Python 3.11**。所有 "tests passed" 类断言必须引用具体 CI run id 或 `gh workflow run` 后 `gh run watch` 的退出码。
- 堆叠分支 CI 真实门禁不自动跑——需 `gh workflow run <wf>.yml --ref <branch>`；**新增** workflow 文件首次执行必须等其落到 `main`（GitHub Actions 注册自 default branch）。

## 1. Stage 0 — CI 卫生 hotfix 验证

### 1.1 静态验证（本地，3.9 即可）

```bash
# (A) 语法正确性
python3 -m py_compile scripts/ci/check_workflow_file_issues.py

# (B) regex 行为单测（新增 tests/unit/test_check_workflow_file_issues.py）
python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.ci.check_workflow_file_issues import _is_missing_workflow_on_ref_error
cases = {
    'HTTP 404: workflow .github/workflows/brep-golden-eval.yml not found on the default branch (https://api.github.com/...)': True,
    'could not find workflow file, try specifying a different ref': True,
    'workflow was not found on the requested ref': True,
    'failed to log in: HTTP 401': False,
    'some unrelated error': False,
}
for msg, want in cases.items():
    got = _is_missing_workflow_on_ref_error(msg)
    assert got == want, f'{msg!r} -> got {got}, want {want}'
print('ok: all 5 cases match')
"
```

期望输出：`ok: all 5 cases match`，退出 0。

### 1.2 `.gitignore` 验证

```bash
git check-ignore -v reports/benchmark/forward_scorecard/latest.json
# 期望：.gitignore:<line> reports/benchmark/    reports/benchmark/forward_scorecard/latest.json
```

### 1.3 CI 验证

提交两个 commit 后 push，等 `stress-tests.yml` 的 `workflow-file-health` job：

```bash
gh run list --workflow stress-tests.yml --branch phase3-vectors-batch-similarity-router-20260429 --limit 1
gh run view <run_id> --json conclusion,jobs --jq '.jobs[] | select(.name=="workflow-file-health") | {name, conclusion}'
```

期望：`conclusion: "success"`，PR #472 的 `workflow-file-health` 由 FAILURE → SUCCESS。

### 1.4 失败模式

| 现象 | 诊断 | 回滚 |
| --- | --- | --- |
| `workflow-file-health` 仍 fail，新错误形态 | 看 fail log 中的"HTTP 4xx"前缀，补 helper 分支 | `git revert <stage0-hotfix-sha>` |
| yaml fallback 误绿坏 workflow | `_check_yaml_parse` 漏了某 key 校验 → 加 yaml 字段断言（不是放宽 fallback） | 同上 |

## 2. Stage 1 — 堆叠链合入 main 验证

### 2.1 每层合入前的 pre-merge 门禁

对 #468 / #471 / #472 各自：

```bash
PR=468  # 重复 471、472
gh pr view "$PR" --json statusCheckRollup,mergeStateStatus,mergeable \
  --jq '{mergeStateStatus, mergeable, checks: [.statusCheckRollup[] | {name, conclusion: (.conclusion // .status // .state)}]}'
```

期望：`mergeStateStatus: "CLEAN"`，所有 check 非 `FAILURE`/`STARTUP_FAILURE`。

### 2.2 #472 合入后的"首次真实门禁"

合入 main 后必须**全部**绿（这是堆叠链上一直没真实跑过的）：

```bash
gh run list --branch main --limit 10 --json conclusion,workflowName,databaseId \
  --jq '.[] | select(.workflowName | test("CI|Tiered|Governance|Code Quality|Evaluation Report")) | {workflowName, conclusion, id: .databaseId}'
```

期望 6 个 workflowName 全部 `success`：

- `CI`
- `CI Tiered Tests`
- `Governance Gates`
- `Code Quality`（首次跑通——hand-off §2 的 "accepted gap" 解除）
- `Evaluation Report`
- `Self-Check`

### 2.3 `brep-golden-eval.yml` 注册验证

合入后：

```bash
gh workflow list --all --json name,state \
  --jq '.[] | select(.name == "B-Rep Golden Eval (OCC)")'
# 期望：state: "active"

# Smoke run（example manifest，不该失败 release gate 因 inputs 默认 fail_on_not_release_ready=false）
gh workflow run brep-golden-eval.yml --ref main
gh run watch  # 等待
```

期望：run conclusion = `success`，artifact `brep-golden-eval` 上传，其中 `reports/benchmark/brep_golden_manifest/summary.json` 的 `status` ∈ {`insufficient_release_samples`, `release_ready`}（example manifest 只有 1 个 case，第一种）。

### 2.4 dependabot PR 清理验证

合入后逐个 check：

```bash
for pr in 389 391 392 393 394 469 470; do
  gh pr view $pr --json mergeStateStatus,number,title \
    --jq '"\(.number) \(.mergeStateStatus) — \(.title)"'
done
```

期望：BLOCKED → CLEAN/UNSTABLE（取决于各自 CI）。逐个 review 后 merge 或 close。

## 3. Stage 2a — B-Rep 真实数据集验证

### 3.1 离线 manifest 校验（OCC 未装环境可跑）

```bash
python3 scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.real.json \
  --min-release-samples 50 \
  --fail-on-not-release-ready \
  --output-json reports/benchmark/brep_golden_manifest/validation.json
echo "exit=$?"
```

期望 `exit=0`。失败时打印的 JSON 检查：

- `status`: `release_ready`
- `release_eligible_count` ≥ 50
- `errors`: `[]`
- `source_type_counts` 中 `fixture`/`synthetic_demo`/`generated_mock` 之和 / 总数 ≤ 50%（健康度判断，**不是 hard fail**）

### 3.2 part_family 覆盖检查

```bash
python3 -c "
import json, collections
m = json.load(open('config/brep_golden_manifest.real.json'))
fams = collections.Counter(c.get('part_family','?') for c in m['cases'] if c.get('release_eligible'))
print(fams)
assert len(fams) >= 4, f'expected >= 4 part_family, got {len(fams)}: {fams}'
assert max(fams.values()) / sum(fams.values()) <= 0.5, f'single family dominates: {fams}'
print('ok')
"
```

### 3.3 License 字段完整性

```bash
python3 -c "
import json
m = json.load(open('config/brep_golden_manifest.real.json'))
missing = [c['id'] for c in m['cases'] if not c.get('license') or not str(c['license']).strip()]
assert not missing, f'cases without license: {missing}'
print('ok: all cases have license')
"
```

### 3.4 CI 真实跑

合入 main 后：

```bash
gh workflow run brep-golden-eval.yml --ref main \
  -f brep_golden_manifest_json=config/brep_golden_manifest.real.json \
  -f brep_golden_manifest_fail_on_not_release_ready=true \
  -f brep_golden_eval_strict=true
gh run watch
```

期望：

- run conclusion = `success`
- artifact 中 `reports/benchmark/brep_step_iges_golden/summary.json` 含字段：
  - `sample_size` ≥ 50
  - `parse_success_count` / `sample_size` ≥ 0.85
  - `graph_valid_count` / `sample_size` ≥ 0.85
  - `synthetic_geometry_not_allowed` ≠ false（strict 模式开关）

### 3.5 失败模式

| 现象 | 诊断 | 处置 |
| --- | --- | --- |
| validator `errors` 非空 | 看错误前缀（`case \`xx\`: ...`）定位 case | 修 manifest，重跑 3.1 |
| `status: insufficient_release_samples` | 真实 case < 50 | 继续 sourcing，**不可** 通过把 `fixture` 改 `release_eligible=true` 凑数（validator 会拒） |
| `parse_success_count` 低 | OCC 解析失败比例高 | 单独跑 `scripts/eval_brep_step_dir.py --strict <path>` 调试 |
| LFS quota 撞墙 | 大文件 push 失败 | 切外部存储 + fetch-on-demand（见 Development §4.4 决策点） |

## 4. Stage 2b — Reviewed manufacturing labels 验证

### 4.1 manifest 校验

```bash
python3 scripts/build_manufacturing_review_manifest.py \
  --validate-manifest data/manifests/manufacturing_review.json \
  --min-reviewed-samples 30
echo "exit=$?"
```

期望 `exit=0`。

### 4.2 路径治理

```bash
git ls-files data/manifests/ | grep -E 'manufacturing_review.*\.(json|csv)$'
# 期望：至少一条命中，且不在 reports/benchmark/

git check-ignore -v data/manifests/manufacturing_review.json
# 期望：(no output, exit 1) — 即 NOT ignored
```

### 4.3 字段完整性（source/payload/detail 三件套）

具体校验由 `--validate-manifest` 自身完成（feature 文档 `CAD_ML_REVIEW_MANIFEST_MERGE_*.md` 是单一信息源）。这里只做"产物存在 + 不在 gitignored 路径"的二次确认。

## 5. Stage 2c — Scorecard 接线验证

**前置硬条件**（必须先验证）：

1. Stage 2a 验证全过（§3 全 ✅）。
2. Stage 2b 验证全过（§4 全 ✅）。

### 5.1 本地构造 scorecard（用真实 summary）

```bash
python3 scripts/export_forward_scorecard.py \
  --brep-summary reports/benchmark/brep_step_iges_golden/summary.json \
  --manufacturing-evidence-summary data/manifests/manufacturing_review_summary.json \
  --manufacturing-review-manifest-validation-summary reports/benchmark/manufacturing_review_validation.json \
  --output-json reports/benchmark/forward_scorecard/latest.json \
  --output-md  reports/benchmark/forward_scorecard/latest.md
```

期望产物：

- `reports/benchmark/forward_scorecard/latest.json` 包含 `summary.forward_status`
- `summary.forward_status` 是 4 个枚举之一：`release_ready` / `benchmark_ready_with_gap` / `shadow_only` / `blocked`
- B-Rep 组件 status **不为** `blocked`（说明真实数据已被消费）

### 5.2 默认假绿回归测试

在 evaluation-report.yml 接线**之前**，先跑一次"故意不传 brep summary"看 B-Rep 组件是否仍判 `blocked`：

```bash
python3 scripts/export_forward_scorecard.py \
  --output-json /tmp/scorecard_no_brep.json
python3 -c "
import json
s = json.load(open('/tmp/scorecard_no_brep.json'))
brep = [c for c in s['components'] if c['name'] in ('brep_3d','brep_step_iges')]
for c in brep:
    assert c.get('status') == 'blocked', f\"expected blocked for missing brep summary, got {c.get('status')} in {c['name']}\"
print('ok: missing brep summary correctly → blocked')
"
```

期望 `ok`。失败 ⇒ 说明 scorecard 对缺省值"默认空 dict"路径漏判，**必须** 在接线前先修。

### 5.3 CI 接线后的 Evaluation Report 验证

修改 `.github/workflows/evaluation-report.yml` 加 `--brep-summary` / `--manufacturing-evidence-summary` 参数后：

```bash
gh workflow run evaluation-report.yml --ref main
gh run watch
gh run view <id> --log | grep -E "(forward_status|brep_summary|manufacturing)"
```

期望日志中能看到真实 summary 文件路径被 echo，且 scorecard 上传到 artifact。

## 6. Stage 3 — 加固单测验证

### 6.1 新增测试落地

每个新测试文件落地后：

```bash
# 本地仅做 py_compile（3.9 跑不了 pytest collection）
python3 -m py_compile tests/unit/test_decision_contract_schema.py
python3 -m py_compile tests/unit/test_decision_service_evidence.py
```

### 6.2 CI 真实跑

push 后等 `ci.yml` 收到（PR→main）：

```bash
gh run view <ci_run_id> --json jobs --jq '.jobs[] | select(.name | test("unit|pytest")) | {name, conclusion}'
```

期望全部 `success`。

### 6.3 失败模式

| 现象 | 诊断 |
| --- | --- |
| 新单测在 3.11 失败 | 看 `pytest -x -v` 输出，多半是 `decision_service` 字段名漂移 → 修 fixture |
| 旧测试因新单测引起 import 副作用挂掉 | 检查 `tests/unit/conftest.py`，确认新文件无 module-level side effect |

## 7. Stage 出门验收清单（每个 Stage 必查）

| # | 检查项 | 工具 | 通过条件 |
| --- | --- | --- | --- |
| 1 | 代码静态绿 | `py_compile` + `yaml.safe_load` | exit 0 |
| 2 | CI 跑过 | `gh run view` | conclusion success |
| 3 | 产物落盘 | `ls <expected_path>` | 文件存在且大小 > 0 |
| 4 | 无"默认值假绿" | 跑 §5.2 类型的负向测试 | 缺数据时 status = blocked |
| 5 | 文档同步 | `git diff docs/development/` | DEVELOPMENT + VERIFICATION + TODO 同 stage 更新 |

## 8. 整体验收：当 Stage 2c 完成时

```bash
# 一次性总验
gh workflow run evaluation-report.yml --ref main
gh run watch
gh run view <id> --json conclusion --jq .conclusion
# 期望：success

# 抓 scorecard 产物
gh run download <id> --name forward-scorecard
cat forward_scorecard/latest.json | python3 -c "
import json, sys
s = json.load(sys.stdin)
print('forward_status:', s['summary']['forward_status'])
print('components:')
for c in s['components']:
    print(f\"  {c['name']:30s} status={c['status']:25s} sample_size={c.get('sample_size','?')}\")
"
```

期望（最低要求）：

- `forward_status` ∈ {`benchmark_ready_with_gap`, `release_ready`}
- `brep_*` 组件 status **不为** `blocked`
- `manufacturing_*` 组件 status **不为** `blocked`

**仅当上述三条全部成立**，方可宣称 "Stages 1+2 scorecard-credible with real data"，开始考虑 Phase 7 design（仍只是 design）。
