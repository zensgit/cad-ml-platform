# CAD ML Forward Roadmap — Next Stages Development Plan

Date: 2026-05-17
Branch: `phase3-vectors-batch-similarity-router-20260429`
Successor of: `CAD_ML_FORWARD_ROADMAP_HANDOFF_20260515.md`

## 0. 状态校准（开工前先对齐）

- Phase 1–6 代码框架已完成且 fail-closed（`17a28676..c67dcbec`），CI（3.10+3.11）on `workflow_dispatch` 已绿。
- 真正阻塞 "release_ready" 流转的是**数据**，不是代码：50–100 个真实 STEP/IGES + reviewed manufacturing labels。
- B6.5 的 95.8% accuracy 是 **2D / DXF** 路径上的数字；**B-Rep 3D 路径目前无真实数据可言准确率**。`forward_scorecard` 把两条流分开统计，下游不可混称。
- 堆叠 PR 链（#468 → #471 → #472）尚未合入 main，`ci.yml` / `code-quality.yml` / `brep-golden-eval.yml` 在堆叠分支无法获得 main 上的真实门禁。

## 1. 范围（本计划交付什么）

| Stage | 交付物 | 阻塞? | 形态 |
| --- | --- | --- | --- |
| **0** | CI 卫生 hotfix（file-health regex + `.gitignore`） | 否 | 代码 |
| **1** | 堆叠链合入 main，dependabot 排队 PR 解锁 | 是（人工） | 治理 |
| **2a** | B-Rep 真实数据集 50–100 STEP/IGES + 真实 manifest | 是（数据） | 数据 |
| **2b** | Reviewed manufacturing labels（≥30） | 是（数据） | 数据 |
| **2c** | Scorecard 接入 2a/2b 真实 summary | 否（依赖 2a+2b） | 代码 |
| **3** | 加固：单测、schema test、observability | 否 | 代码 |

Phase 7（参数化/生成式）按 hand-off §4 **保持 design-only，不开工**。

## 2. Stage 0 — CI 卫生 hotfix（本次会话内可落地）

### 2.1 file-health fallback regex 扩展

**问题**：PR #472 唯一红 check 是 `workflow-file-health`（`stress-tests.yml:19`），失败行：

```
[fail] gh .github/workflows/brep-golden-eval.yml -
  HTTP 404: workflow ... not found on the default branch
```

**根因**：`scripts/ci/check_workflow_file_issues.py:43-48` 的 `_is_missing_workflow_on_ref_error` 仅匹配两种错误形态：

```python
return (
    "could not find workflow file" in text
    and "try specifying a different ref" in text
) or ("workflow was not found" in text and "ref" in text)
```

GitHub 的 "新 workflow 文件未在 default branch 注册" 走的是第三种形态 `"HTTP 404: workflow ... not found on the default branch"`，不命中任一分支，因此 `check_workflow_file_issues.py:206-213` 的 `can_fallback`（要求"所有 gh 错误都属于这一类"）失败，无 yaml fallback，整个 job 红。

**修法**（最小改动，1–3 行）：在 `_is_missing_workflow_on_ref_error` 加一个判别：

```python
def _is_missing_workflow_on_ref_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        ("could not find workflow file" in text
         and "try specifying a different ref" in text)
        or ("workflow was not found" in text and "ref" in text)
        or ("not found on the default branch" in text)
    )
```

**副作用评估**：该 helper 只控制是否触发 yaml fallback；yaml fallback 仍然校验 `name`/`on` 必备 keys（`_check_yaml_parse`），不会让"真正坏掉的 workflow 文件"假绿。等价于**把"堆叠分支上新增的合法 workflow"显式认作可兜底**。

### 2.2 `reports/benchmark/` 进 `.gitignore`

Commit-split 文档 §"Cross-cutting artifacts" 已点名 follow-up；当前是生成产物却未 ignore。新增一行：

```
reports/benchmark/
```

放在 `.gitignore` 已有 `reports/` 区段附近。

### 2.3 Stage 0 提交切分

| Commit | 内容 |
| --- | --- |
| `chore(ci): tolerate "not found on the default branch" in workflow file-health fallback` | 仅改 `scripts/ci/check_workflow_file_issues.py` + 增 1 个 unit test |
| `chore(repo): gitignore reports/benchmark generated artifacts` | 仅 `.gitignore` |

两条互独立，独立可 revert。

## 3. Stage 1 — 堆叠链合入 main

**当前堆叠状态**：

```
main
 └─ #468  phase3-vectors-crud-router-20260422 (CLEAN, 等 merge)
      └─ #471  phase3-vectors-list-router-20260429 (CLEAN)
           └─ #472  phase3-vectors-batch-similarity-router-20260429 (UNSTABLE — Stage 0 后翻 CLEAN)
```

**推进顺序**：

1. Stage 0 落地 → PR #472 翻 CLEAN。
2. 顺序 squash-merge：#468 → #471 → #472。每合一层，下游 PR 自动 rebase。
3. #472 合入后，触发的"真正门禁"：
   - `ci.yml` + `ci-tiered-tests.yml` + `governance-gates.yml` 真实跑（不再靠 `workflow_dispatch`）。
   - `code-quality.yml`（无 `workflow_dispatch`）首次跑通——这是 hand-off §2 documented 的"accepted gap"被解。
   - `brep-golden-eval.yml` 在 GitHub Actions UI 出现 → 可手动 dispatch（Stage 2a 解锁）。
4. 解除堆叠后，扫一遍 dependabot 阻塞：#389/#391/#392/#393/#394/#469/#470 共 7 个 PR 状态从 BLOCKED → 检查 mergeability，按风险分批 merge（actions-minor / docker base / pip group 各算一批）。

**风险栅**：

- ✗ 不要 `git reset --hard` 或 force-push 已 push 的堆叠分支——Stage 0 的两个 commit 应作为新 commit 追加到 `phase3-vectors-batch-similarity-router-20260429`，PR #472 自动重跑 CI。
- ✗ `code-quality.yml` 首次跑通可能暴露此前堆叠分支屏蔽的 mypy/ruff/lint 问题。**预期**：列入 follow-up，不要塞回堆叠 PR。
- ✓ 回滚锚：`pre-split-backup-20260515` 仍可用。

## 4. Stage 2a — B-Rep 真实数据集

### 4.1 目标

- ≥ 50 个 `release_eligible=true` 的 STEP/IGES 文件（`source_type ∉ {fixture, synthetic_demo, generated_mock}`），通过 `scripts/validate_brep_golden_manifest.py --min-release-samples 50 --fail-on-not-release-ready` 退出码 0。
- 覆盖 ≥ 4 个 `part_family`，避免单一形态导致 scorecard 偏置。

### 4.2 数据来源（按 ROI 排序）

| 来源 | 估算可用规模 | License | 备注 |
| --- | --- | --- | --- |
| **ABC Dataset** (Koch et al., NYU) | 750k STEP | CC-BY-NC-SA 4.0 | 大量 mechanical parts，建议作为主源；需在 manifest `license` 字段如实标注 |
| **内部历史交付** | 视客户而定 | 内部 | 按 `part_family` 分层抽样 8–10 类，每类 5–8 个；最贴近生产分布 |
| **Onshape Public** | 数十万 | 各异 | 必须逐 document 检查 license；导出脚本可参考 Onshape API `getStepExport` |
| **FreeCAD 参数化生成** | 任意 | 自有 | **仅用于覆盖率补丁**，标 `source_type=synthetic_demo`，不计入 release_eligible |

### 4.3 双 manifest 策略

```
config/
  brep_golden_manifest.example.json    # 现有，仅 1 fixture（demo 用）
  brep_golden_manifest.real.json       # 新增，≥50 release-eligible（Stage 2a 交付）
  brep_golden_manifest.synthetic.json  # 新增，FreeCAD 合成补丁（不进 release gate）
```

`brep-golden-eval.yml` 的 `workflow_dispatch.inputs.brep_golden_manifest_json` 已支持任意路径，无需改 workflow。

### 4.4 文件落地路径

- 数据文件：`data/brep_golden/{part_family}/<case_id>.{step,iges}`（**新增目录，需进 `.gitattributes` 走 Git LFS**——单个 STEP 常常 5–50 MB，50 个文件可能 1–2 GB）。
- 若仓库不接受 LFS：放外部对象存储（S3/GCS），manifest 走相对路径 + `scripts/ci/build_brep_golden_manifest_optional.sh` 增加 fetch-on-demand 逻辑（Stage 2a 的子工作）。
- **决策点**：在执行前需要先评估"仓库 LFS 配额" vs "外部存储 + CI fetch"两种方案，选其一。

### 4.5 流水线

1. 收集 → 按 license / part_family 标注 → 落到 `data/brep_golden/`。
2. 生成 `brep_golden_manifest.real.json`，每 case 填全 `id/path/format/source_type/release_eligible/part_family/license/expected_behavior/expected_topology/tags`。
3. 本地（OCC 已装环境）跑 `python3 scripts/validate_brep_golden_manifest.py --manifest config/brep_golden_manifest.real.json --min-release-samples 50 --fail-on-not-release-ready` → exit 0。
4. Push 至 main 后 dispatch `B-Rep Golden Eval (OCC)`，输入 `brep_golden_manifest_json=config/brep_golden_manifest.real.json brep_golden_manifest_fail_on_not_release_ready=true`。
5. CI 产出 `reports/benchmark/brep_step_iges_golden/summary.json` → Stage 2c 的输入。

## 5. Stage 2b — Reviewed manufacturing labels

### 5.1 目标

≥ 30 个被人工 review 过、source/payload/detail 三字段齐全的 manufacturing label，路径在跟踪目录 `data/manifests/`（**不在** gitignored 的 `reports/benchmark/`）。

### 5.2 工具

- `scripts/build_manufacturing_review_manifest.py`（3002 行，已有）
- 配套文档（feature 文档为单一信息源）：
  - `docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_*.md`
  - `docs/development/CAD_ML_REVIEW_MANIFEST_MERGE_*.md`

### 5.3 流水线

1. 从最近的 batch_classify 结果 CSV 起步：`--from-results-csv <path>`。
2. 生成 reviewer 填空模板：`--reviewer-template-csv <out>`（按文档约定）。
3. 人工 review 填好后，按 feature 文档的 apply/merge 路径产出最终 manifest。
4. 校验 + 落地：
   ```
   python3 scripts/build_manufacturing_review_manifest.py \
     --validate-manifest data/manifests/manufacturing_review.json \
     --min-reviewed-samples 30 \
     --output-csv data/manifests/manufacturing_review_summary.csv
   ```
   exit 0 ⇒ 进入 Stage 2c。

### 5.4 治理铁律

- ✗ 不要把 reviewed manifest 放在 `reports/benchmark/`（gitignored，会被 Stage 0 的 `.gitignore` 一并吞掉）。
- ✗ 不要把 apply/merge 的具体 flag 凭记忆写 PR 描述——以 `--help` 与配套 feature 文档为准。

## 6. Stage 2c — Scorecard 真实数据接线

**前置硬条件**：Stage 2a 产出 `brep_summary.json` **且** Stage 2b 产出 `manufacturing_review.json` 并通过 `--validate-manifest`。

在 `.github/workflows/evaluation-report.yml` 中给 `scripts/export_forward_scorecard.py` 传：

```
--brep-summary reports/benchmark/brep_step_iges_golden/summary.json
--manufacturing-evidence-summary data/manifests/manufacturing_review_summary.json
--manufacturing-review-manifest-validation-summary <validation_out>.json
```

**绝不**在 Stage 2a/2b 完成前接——`forward_scorecard.py` 对缺省的 summary 走"默认空 dict"路径，会让 `release_ready` 误绿（hand-off §3.D 明确风险）。

接线后期望状态：

- 2D/DXF 流 `hybrid_component.forward_status` 维持现状（取决于实际 hybrid summary）。
- B-Rep 流首次脱离 `blocked` → 进入 `shadow_only`/`benchmark_ready_with_gap`/`release_ready` 之一（取决于实际数据通过率）。
- 整体 `summary.forward_status` 取所有 component 最低 rank。

## 7. Stage 3 — 代码加固（可与 Stage 2a/2b 并行）

低风险、纯单测/observability，与数据采集并行不冲突：

| 项 | 文件 | 价值 |
| --- | --- | --- |
| `decision_contract.v1` 字段 schema 单测 | `tests/unit/test_decision_contract_schema.py`（新） | 防止字段重命名/丢失静默漂移 |
| `decision_service._prediction_evidence` / `_brep_evidence` 单测 | `tests/unit/test_decision_service_evidence.py`（新） | 当前仅集成端覆盖，纯单测补盲 |
| `readiness_registry._CHECKSUM_CACHE` 失效路径 | `tests/unit/test_model_readiness_registry.py`（已存在，扩展） | mtime/size 变化触发 re-checksum 的显式断言 |
| `forward_scorecard._metric_status` 真值表参数化 | `tests/unit/test_forward_scorecard.py`（已存在，扩展） | 4 status × sample_size/threshold 边界矩阵 |
| `forward_scorecard` 总线 component 缺失行为 | `tests/unit/test_forward_scorecard_release_gate.py`（已存在，扩展） | 验证"缺 brep_summary 时 B-Rep 流必须 blocked 而非默认 release_ready" |

每项独立 commit，独立可 revert。

## 8. 提交规范（保持现有风格）

参考 `git log main..HEAD`：

- `chore(ci): ...` — CI/repo housekeeping
- `feat: ...` — 新能力
- `fix: ...` — bug
- `refactor: ...` — 等价重构
- `docs: ...` — 文档
- 每个 commit 必须独立可 revert；尾注：

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## 9. 时间盒（建议）

| 周 | 目标 |
| --- | --- |
| W1（本周） | Stage 0 + Stage 1 开始（#468 合入） |
| W2 | Stage 1 完成（#472 合入 + dependabot 清理）；Stage 2a 数据 sourcing 开工 |
| W3–W4 | Stage 2a 完成；Stage 2b 启动 |
| W5 | Stage 2b 完成；Stage 2c 接线 → 首份"含真实 B-Rep 数据"的 forward scorecard |
| 并行 | Stage 3 单测加固随时插入 |

## 10. 出门检查（每个 Stage 完成时三问）

1. CI（3.11，PR→main 上下文）跑过么？产物在哪？
2. 这次改动有没有让任何 gate"看起来更绿但本质未验证"？（合成数据混入、缺省值假绿、跳过 strict 模式）
3. 配套验证文档是否更新到了同一 stage？（见 `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_VERIFICATION_20260517.md`）

## 11. 不做什么（显式排除）

- ✗ Phase 7 任何代码——hand-off §4，等 Stages 1+2 scorecard 真实数据通过后再开。
- ✗ 蒸馏复活——B6.5 已结论"异构集成不可替代"。
- ✗ 在堆叠分支上新增第二个未在 main 注册的 workflow——除非已计入"main 合入后 dispatch"的延迟代价。
- ✗ 把 Stage 2c 的接线提前到 2a/2b 之前。
