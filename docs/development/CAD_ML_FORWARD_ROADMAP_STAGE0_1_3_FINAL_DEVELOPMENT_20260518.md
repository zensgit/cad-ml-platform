# CAD ML Forward Roadmap — Stages 0 / 1 / 3 Final Development Report

Date: 2026-05-18
Predecessors:
- `CAD_ML_FORWARD_ROADMAP_HANDOFF_20260515.md`
- `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md`
- `CAD_ML_FORWARD_ROADMAP_STAGE1_LAND_STACK_DEVELOPMENT_20260517.md`
- `CAD_ML_FORWARD_ROADMAP_STAGE1_COMPLETION_REPORT_20260518.md`

## 1. 终态（本日完成）

Codex 的 forward roadmap (Phases 1–6) + 本会话三阶段 (0 / 1 / 3) 全部在 `main` 上 CI-verified。

```
main HEAD（按时间倒序）
b026fd8b  Merge pull request #474 — Stage 3 hardening unit tests
41971faf  test: stage 3 hardening — 5 new unit tests guarding release infrastructure
f2553f9d  Merge pull request #473 — fix(tests) stress-tests --mode auto assertion
43d01e0f  fix(tests): stress-tests workflow-file-health uses --mode auto
01f12b87  Merge pull request #472 — Phase 1–6 + Stage 0 hotfix + roadmap docs
3c5399bb  Merge pull request #471 — split vectors list router
968e5fc8  Merge pull request #468 — split vectors crud router
e9d05258  docs: stage 1 land-stack development + verification
9a63e006  chore(ci): stress-tests workflow-file-health uses --mode auto
03be32bd  docs: forward-roadmap next-stages plan + verification + TODO
b428bfa4  chore(ci): tolerate "not found on the default branch" fallback
```

## 2. Stage 0 — CI 卫生 hotfix

### 2.1 计划与实际

| 步骤 | 计划 | 实际 |
| --- | --- | --- |
| S0.1 helper regex | 加 "not found on the default branch" 分支 | ✅ commit `b428bfa4` |
| S0.2 单测 | 5-case parametric + auto-mode 集成 | ✅ 同 commit |
| S0.5–S0.7 `.gitignore` | 添加 `reports/benchmark/` | ✅ 已是历史状态 (`.gitignore:144`)，commit-split 文档过时 |
| S0.8 push 验证 | PR #472 file-health 翻绿 | ❌ 仍 FAILURE — 真问题：caller `--mode gh` 不触发 fallback |
| **S0.10 新增** | stress-tests.yml 改 `--mode auto` | ✅ commit `9a63e006` |

### 2.2 关键经验

改 helper 时必须同步验证 **caller 实际路径**。advisor 与计划文档都漏识别"caller 用 `--mode gh` 完全跳过 fallback 逻辑"。grep 调用方比改 helper 更早 done。

### 2.3 隐藏后遗症

Stage 0 commit `9a63e006` 改了 stress-tests.yml 的 mode，但没改 `tests/unit/test_stress_workflow_workflow_file_health.py:45` 的 assertion `"--mode gh" in run_script`。

- 堆叠分支 PR-level CI 是轻量集合，**没跑这个测试** → push 完看 PR #472 全绿是误判。
- 堆叠链合入 main 后 `tests (3.10)` / `tests (3.11)` / `unit-tier` 首跑直接 red。
- 修复 = PR #473（详见 §3.4）。

## 3. Stage 1 — 堆叠链合入 main

### 3.1 治理约束

Main protection 既要求 `required_approving_review_count=1` 又开 `enforce_admins=true`：

- admin token 不能 bypass review
- author = `zensgit` = git user，无法 self-approve（GitHub 硬规则）
- 自动模式 classifier 也正确拒绝 self-approve

### 3.2 采用方案 C（用户授权，执行 3 次 cycle）

每次 cycle：

```bash
gh api -X DELETE repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins
# ... admin-merge ...
gh api -X POST   repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins
gh api          repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins --jq '.enabled'
# 必须 = true
```

| Cycle | 用途 | 合入数 | `enforce_admins=false` 暴露窗口 |
| --- | --- | --- | --- |
| 1 | #468 / #471 / #472 主体堆叠链 | 3 | ~3 分钟 |
| 2 | #473 Stage 0 follow-up | 1 | ~30 秒 |
| 3 | #474 Stage 3 hardening | 1 | ~30 秒 |

每次 cycle 后 `enforce_admins=true` 在最后一步显式验证。

### 3.3 顺序合入推进

```
#468 (crud → main)               — admin merge → 968e5fc8
#471 retarget → main → CI watch → admin merge → 3c5399bb
#472 retarget → main → CI watch → admin merge → 01f12b87 (--delete-branch=true)
```

每次合入后 `git fetch && git log origin/main` 验证 merge commit 出现。

### 3.4 PR #473 — Stage 0 commit `9a63e006` 的 follow-up

| 字段 | 值 |
| --- | --- |
| 触发 | main `tests (3.10)` / `(3.11)` / `unit-tier` red |
| 根因 | `tests/unit/test_stress_workflow_workflow_file_health.py:45` 仍断言 `--mode gh` |
| 修法 | 改为 `--mode auto`，加注释链回 commit `9a63e006` 说明 |
| Merge | `f2553f9d`（Cycle 2） |
| 验证 | 新 main CI run 26011196551 + 26011196580 翻绿 |

### 3.5 main 上 6 个真实门禁首跑

首次脱离堆叠分支屏蔽，**全部** success：

| Workflow | Run ID | 备注 |
| --- | --- | --- |
| CI | 26011196551 | 含 unit-tier + tests (3.10) + tests (3.11) |
| CI Tiered Tests | 26011196580 | |
| Governance Gates | 26008774788 | |
| Code Quality | 26008774795 | **hand-off §2 accepted gap 解除 — 无 mypy/lint/docstring 债** |
| Evaluation Report | 26008774768 | |
| Self-Check | 26008774767 | |
| B-Rep Golden Eval (OCC) smoke | 26008810573 | example manifest，status=insufficient_release_samples（预期） |

Code Quality 首跑零债：hand-off §6.2 预测的 4 类失败（mypy strict / isort / docstring / dead-code）实际**全部未发生**。Codex 在做 Phase 1–6 split 时已经一并处理了。

### 3.6 Dependabot 7 个 PR — 未自动 close

7 个 BLOCKED dependabot PR (#389, #391, #392, #393, #394, #469, #470) 全部违反 Action Pin Guard policy（dependabot 写 `uses: foo/bar@v5`，repo 要求 `uses: foo/bar@<SHA>`）。

**未自动 close**：auto-mode classifier 拒绝批量 close 7 个外部 PR。推荐处置命令写入 STAGE1 完成报告 §3.3，由人工执行：

```bash
for pr in 389 391 392 393 394 469 470; do
  gh pr close $pr --comment "..."
done
gh issue create --title "dependabot config: emit SHA-pinned PRs ..."
```

## 4. Stage 3 — 5 个新单测加固 release infrastructure

### 4.1 测试文件

| 文件 | 防御的 invariant |
| --- | --- |
| `tests/unit/test_decision_contract_schema.py` | v1 key set 锁定（8 个字段；renames/drops 触发 assertion） |
| `tests/unit/test_decision_service_evidence.py` | `_prediction_evidence` / `_top_brep_hint` / `_brep_evidence` 单测覆盖，包含 None-return / label-key fallback / detail compaction / valid_3d→status |
| `tests/unit/test_model_readiness_registry_cache_invalidation.py` | `_CHECKSUM_CACHE` 在 size change AND 同 size 不同 content 时都失效 |
| `tests/unit/test_forward_scorecard_metric_status_truth_table.py` | `_metric_status` 参数化真值表：sample_size {-1,0,1,9,10,29,30,50,100} × primary score bands × secondary/low_conf/custom-threshold overrides |
| `tests/unit/test_forward_scorecard_missing_component_blocked.py` | **load-bearing**：空 brep + manufacturing summary 必须 NOT release_ready |

### 4.2 设计原则

- **无 fixture 跨文件耦合** — 每个测试文件自带最小 fixtures，便于 future-self 阅读
- **静态数据 + 显式默认** — 不依赖 monkeypatch（除 readiness cache 需触摸 module-level cache）
- **失败信息携带修法提示** — `forward_scorecard_missing_component_blocked` 的 assertion 失败时打印"这是 default-green 失败模式 Stage 2c 必须永不重启"

### 4.3 治理动作

| Item | 状态 |
| --- | --- |
| 不动生产代码（仅 `tests/unit/*` 新文件） | ✅ `git diff --stat 41971faf^..41971faf` 仅 5 个 test 文件 |
| 本地 `py_compile` 全过 | ✅ 5/5 |
| PR-level CI 全过 | ✅ failing=0 |
| 合入 main 后 CI 验证 | ✅ CI run 26018132929 success / CI-Tiered 26018132993 success / Code Quality 26018132996 success |

## 5. Stage 2a/2b/2c 状态（未启动）

`CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md` §4–6 文档化；瓶颈是数据采集 + 人工 review，非代码。Stage 3 加固在 Stage 2c 接线前提供了"默认空 → blocked"防御。

## 6. Phase 7 (parametric / generative)

按 hand-off §4 **保持 design-only**。本次会话**未变更**。

## 7. 风险与未决项

| Item | 状态 | 影响 |
| --- | --- | --- |
| 7 个 dependabot PR 待人工 close | 待处置 | 看板视觉噪声 + 升级债积累 |
| dependabot config SHA-pin 改造 | follow-up issue 待立 | 未来重发 PR 仍会触发 Action Pin Guard |
| Stage 2a 真实 STEP/IGES 数据 | 未启动 | release_ready 实质门槛 |
| Stage 2b reviewed manufacturing labels | 未启动 | 同上 |
| Stage 2c scorecard 接线 | 阻塞于 2a+2b | 不可提前 |

## 8. 单作者治理纪律（未来 PR 合入约束）

- ✅ 永久关 `enforce_admins`（方案 A） — **未采纳**，治理保留
- ✅ 永久降 `required_approving_review_count` 为 0（方案 B） — **未采纳**
- ✅ 每次合 PR 走方案 C cycle — **采纳**，本日已成功执行 3 次

每次 cycle 暴露窗口 ≤ 3 分钟，可被 `git log + GitHub audit log` 完整追溯。

## 9. 后续会话开工三句话

如果你接手这条线：

1. **main 上 Phase 1–6 + Stage 0/1/3 全部 CI-verified**，6 大门禁 + brep smoke 全绿
2. **release 真假取决于 B-Rep 真实数据 + reviewed manufacturing labels 的人工填充** — 是数据不是代码
3. **未来合 PR：找第二个 reviewer 或走方案 C cycle**；不要 self-approve、不要 force-push、不要永久关 `enforce_admins`

## 10. 文档地图（本会话沉淀）

```
docs/development/
├── CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md       (整体计划)
├── CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_VERIFICATION_20260517.md      (Stage 0–3 验证)
├── CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_TODO_20260517.md              (64 条 checklist，本日勾选 Stage 0+1.A)
├── CAD_ML_FORWARD_ROADMAP_STAGE1_LAND_STACK_DEVELOPMENT_20260517.md (Stage 1 详细)
├── CAD_ML_FORWARD_ROADMAP_STAGE1_LAND_STACK_VERIFICATION_20260517.md (Stage 1 验证)
├── CAD_ML_FORWARD_ROADMAP_STAGE1_COMPLETION_REPORT_20260518.md      (Stage 1 完成报告)
├── CAD_ML_FORWARD_ROADMAP_STAGE0_1_3_FINAL_DEVELOPMENT_20260518.md  ← 本文档
└── CAD_ML_FORWARD_ROADMAP_STAGE0_1_3_FINAL_VERIFICATION_20260518.md (本文档配套)
```

## 11. 一句话总结

**Stage 0 + 1 + 3 完整闭环；release infrastructure 完成且 fail-closed；剩余瓶颈是数据，不是代码。**
