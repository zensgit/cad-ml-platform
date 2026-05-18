# CAD ML Forward Roadmap — Next Stages TODO

Date: 2026-05-17
Owner: TBD
Companions:
- `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md`
- `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_VERIFICATION_20260517.md`

## Legend

- `[ ]` 待办；`[x]` 完成；`[~]` 进行中；`[!]` 阻塞
- 每条任务尾巴标注 `(estimate: <N>h | risk: low|med|high)`
- 任务编号 `S<stage>.<seq>`，引用便于跨文档

---

## Stage 0 — CI 卫生 hotfix（本会话内可落地，2 commit）

- [ ] **S0.1** 在 `scripts/ci/check_workflow_file_issues.py:43-48` 的 `_is_missing_workflow_on_ref_error` 加 `"not found on the default branch" in text` 分支（estimate: 0.25h | risk: low）
- [ ] **S0.2** 新增 `tests/unit/test_check_workflow_file_issues.py`，参数化覆盖 5 种错误形态（3 ✅ + 2 ❌）（estimate: 0.5h | risk: low）
- [ ] **S0.3** 本地跑验证 §1.1（B）的 5-case 内联 assert，期望 `ok: all 5 cases match`（estimate: 0.1h | risk: low）
- [ ] **S0.4** Commit: `chore(ci): tolerate "not found on the default branch" in workflow file-health fallback`（estimate: 0.1h | risk: low）
- [ ] **S0.5** `.gitignore` 添加 `reports/benchmark/`（参 commit-split §"Cross-cutting artifacts"）（estimate: 0.1h | risk: low）
- [ ] **S0.6** `git check-ignore -v reports/benchmark/forward_scorecard/latest.json` 验证生效（estimate: 0.05h | risk: low）
- [ ] **S0.7** Commit: `chore(repo): gitignore reports/benchmark generated artifacts`（estimate: 0.1h | risk: low）
- [ ] **S0.8** Push → 等 PR #472 上 `workflow-file-health` 由 FAILURE → SUCCESS（estimate: 0.2h | risk: low）
- [ ] **S0.9** 若仍红，按验证文档 §1.4 表格走诊断（estimate: var | risk: var）

**Stage 0 出门条件**：PR #472 的 `workflow-file-health` SUCCESS，整 PR 进入 CLEAN/UNSTABLE 但 UNSTABLE 不含 file-health。

---

## Stage 1 — 堆叠链合入 main（人工）

### 1.A 顺序合入

- [ ] **S1.1** 重审 #468（base=main），CI 全绿 → squash-merge（estimate: 0.5h | risk: med — `code-quality.yml` 首跑可能暴露既有 lint 债）
- [ ] **S1.2** #468 合入后，#471 自动 rebase → 重审 → squash-merge（estimate: 0.3h | risk: low）
- [ ] **S1.3** #472 自动 rebase → 等 Stage 0 落地后翻 CLEAN → squash-merge（estimate: 0.3h | risk: low）

### 1.B 首次真实门禁观察

- [ ] **S1.4** main 上等下列 6 个 workflow 全绿（见验证 §2.2）：CI / CI Tiered Tests / Governance Gates / Code Quality / Evaluation Report / Self-Check（estimate: 1h watching | risk: med — code-quality 首跑）
- [ ] **S1.5** 若 `code-quality.yml` 红，**单独开一个 PR 修**，**不要** 塞回堆叠链（estimate: var | risk: med）

### 1.C `brep-golden-eval.yml` 注册验证

- [ ] **S1.6** `gh workflow list` 看到 "B-Rep Golden Eval (OCC)" 出现且 `state: active`（estimate: 0.05h | risk: low）
- [ ] **S1.7** Smoke dispatch（example manifest，默认 `fail_on_not_release_ready=false`）→ success（estimate: 15min CI | risk: low）

### 1.D Dependabot 清理（7 个 BLOCKED PR）

- [ ] **S1.8** #469 nvidia/cuda 11.8 → 13.2.1 runtime — review，关注 base image 兼容性（estimate: 1h | risk: med — major version bump）
- [ ] **S1.9** #470 actions-minor group — review，多为 minor bump（estimate: 0.5h | risk: low）
- [ ] **S1.10** #389 pip python-minor group(37 updates) — **分批 review**，逐 package 看 changelog（estimate: 3h | risk: med — 37 个包）
- [ ] **S1.11** #391 azure/k8s-set-context 3.1 → 5 — major bump，看 k8s 部署是否有兼容性变更（estimate: 0.5h | risk: med）
- [ ] **S1.12** #392 mamba-org/setup-micromamba 1.11.0 → 3.0.0 — major bump，**关键**（影响 `brep-golden-eval.yml`）（estimate: 1h | risk: med — 已 pin 1.5.8）
- [ ] **S1.13** #393 actions/github-script 6.4.1 → 9.0.0 — major bump（estimate: 0.5h | risk: low）
- [ ] **S1.14** #394 azure/setup-helm 3.5 → 5 — major bump（estimate: 0.5h | risk: low）

**Stage 1 出门条件**：main 上 6 个真实门禁 workflow 全绿；dependabot 7 个 PR 至少处置完毕（合入或显式拒绝并 close）。

---

## Stage 2a — B-Rep 真实数据集 ≥ 50

### 2a.A 决策点（先选路）

- [ ] **S2a.1** 评估仓库 LFS 配额：现有 `git lfs ls-files` 数量 + 预算 50–100 STEP × 平均大小（estimate: 0.5h | risk: low）
- [ ] **S2a.2** **决策**：LFS 内置 vs 外部对象存储 + CI fetch（estimate: 0.5h | risk: med — 影响长期成本）
- [ ] **S2a.3** 若选外部存储：扩展 `scripts/ci/build_brep_golden_manifest_optional.sh` 加 fetch-on-demand 步骤（estimate: 4h | risk: med）

### 2a.B 数据 sourcing

- [ ] **S2a.4** ABC Dataset：抽取 30 个 STEP，覆盖 ≥ 4 个 part_family，license 标 `CC-BY-NC-SA 4.0`（estimate: 4h | risk: low）
- [ ] **S2a.5** 内部历史交付：联系业务侧，挑 20 个真实交付件，按 part_family 分层（estimate: 8h human-loop | risk: med — 依赖业务方）
- [ ] **S2a.6**（可选补丁）FreeCAD 脚本生成 20 个合成 case，标 `source_type=synthetic_demo`，放 `brep_golden_manifest.synthetic.json`（estimate: 6h | risk: low）

### 2a.C Manifest 构造

- [ ] **S2a.7** 新建 `config/brep_golden_manifest.real.json`，schema_version=`brep_golden_manifest.v1`（estimate: 2h | risk: low）
- [ ] **S2a.8** 每 case 填 `id/path/format/source_type/release_eligible/part_family/license/expected_behavior/expected_topology/tags`（estimate: 4h | risk: med — 拓扑下限要 OCC 探一遍）
- [ ] **S2a.9** 跑 `scripts/validate_brep_golden_manifest.py --min-release-samples 50 --fail-on-not-release-ready` → exit 0（estimate: 0.3h | risk: low）
- [ ] **S2a.10** 跑覆盖检查（验证 §3.2）：≥ 4 part_family，单家族占比 ≤ 50%（estimate: 0.1h | risk: low）
- [ ] **S2a.11** 跑 license 完整性（验证 §3.3）（estimate: 0.1h | risk: low）

### 2a.D CI 真实跑

- [ ] **S2a.12** PR 合入 main 后，`gh workflow run brep-golden-eval.yml -f brep_golden_manifest_json=config/brep_golden_manifest.real.json -f brep_golden_manifest_fail_on_not_release_ready=true`（estimate: 25min CI | risk: med）
- [ ] **S2a.13** 检查 `summary.json` 中 `parse_success_count/sample_size ≥ 0.85` 且 `graph_valid_count/sample_size ≥ 0.85`；strict 模式无 `synthetic_geometry_not_allowed` 误判（estimate: 0.5h | risk: med）
- [ ] **S2a.14** 失败时按验证 §3.5 表格诊断（estimate: var | risk: var）

**Stage 2a 出门条件**：dispatch run success + summary.json 三项指标达标 + manifest 在 main 上 tracked。

---

## Stage 2b — Reviewed manufacturing labels ≥ 30

### 2b.A 模板生成

- [ ] **S2b.1** 选定 source batch_classify CSV（取最近一次代表性运行）（estimate: 0.5h | risk: low）
- [ ] **S2b.2** `python3 scripts/build_manufacturing_review_manifest.py --from-results-csv <src> --reviewer-template-csv data/manifests/manufacturing_review_template.csv`（estimate: 0.3h | risk: low）

### 2b.B 人工 review

- [ ] **S2b.3** Review ≥ 30 个 row，填 source/payload/detail 三字段（estimate: 6h human-loop | risk: med — 速率取决于熟练度）

### 2b.C Apply / merge

- [ ] **S2b.4** 按 `docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_*.md` apply 模板（estimate: 0.5h | risk: low）
- [ ] **S2b.5** 按 `docs/development/CAD_ML_REVIEW_MANIFEST_MERGE_*.md` 合入主 manifest 至 `data/manifests/manufacturing_review.json`（estimate: 0.5h | risk: low）

### 2b.D 校验 + 路径治理

- [ ] **S2b.6** `python3 scripts/build_manufacturing_review_manifest.py --validate-manifest data/manifests/manufacturing_review.json --min-reviewed-samples 30` → exit 0（estimate: 0.2h | risk: low）
- [ ] **S2b.7** `git ls-files data/manifests/` 看到 manifest tracked；`git check-ignore -v <path>` 无命中（estimate: 0.1h | risk: low）
- [ ] **S2b.8** Commit `feat(data): reviewed manufacturing labels (Stage 2b)`（estimate: 0.1h | risk: low）

**Stage 2b 出门条件**：`--validate-manifest --min-reviewed-samples 30` exit 0，manifest 在 `data/manifests/` 跟踪。

---

## Stage 2c — Scorecard 真实数据接线

**硬前置**：Stage 2a 全部 ✅ **且** Stage 2b 全部 ✅。

### 2c.A 负向回归（接线前先验"假绿"机制）

- [ ] **S2c.1** 跑验证 §5.2 负向测试：缺 brep summary 时 B-Rep 组件必须 `blocked`（estimate: 0.2h | risk: med — 若失败必须先修 scorecard，不可绕过）
- [ ] **S2c.2** 同上对 manufacturing summary（estimate: 0.2h | risk: med）

### 2c.B 本地构造完整 scorecard

- [ ] **S2c.3** 跑验证 §5.1 命令，产出 `reports/benchmark/forward_scorecard/latest.{json,md}`（estimate: 0.3h | risk: low）
- [ ] **S2c.4** 检查 `summary.forward_status` 不为 `blocked`，B-Rep 组件不为 `blocked`（estimate: 0.1h | risk: low）

### 2c.C 接线 evaluation-report.yml

- [ ] **S2c.5** 在 `.github/workflows/evaluation-report.yml` 找到 `export_forward_scorecard.py` 调用步骤，添加 `--brep-summary` / `--manufacturing-evidence-summary` / `--manufacturing-review-manifest-validation-summary` 三个参数（estimate: 1h | risk: med）
- [ ] **S2c.6** 添加 `Upload forward-scorecard` artifact step（若尚未有）（estimate: 0.3h | risk: low）
- [ ] **S2c.7** Commit `feat(ci): wire real B-Rep + manufacturing summaries into forward scorecard (Stage 2c)`（estimate: 0.1h | risk: low）

### 2c.D 端到端验证

- [ ] **S2c.8** PR 合入 main 后 dispatch `evaluation-report.yml`，等 success（estimate: 30min CI | risk: med）
- [ ] **S2c.9** 下载 artifact，跑验证 §8 总验脚本（estimate: 0.3h | risk: low）
- [ ] **S2c.10** 期望 `forward_status` ∈ {`benchmark_ready_with_gap`, `release_ready`}（estimate: 0h — 观察 | risk: high — 真实数据可能不达标，需诊断）

**Stage 2c 出门条件**：evaluation-report run success + scorecard `forward_status` 不为 `blocked` + B-Rep/manufacturing 组件均非 `blocked`。

---

## Stage 3 — 代码加固（可与 2a/2b 并行）

### 3.A Decision contract / service 单测补盲

- [ ] **S3.1** 新增 `tests/unit/test_decision_contract_schema.py`，断言 `build_classification_decision_contract` 返回 keys 集合 = 已定义字段（estimate: 1h | risk: low）
- [ ] **S3.2** 新增 `tests/unit/test_decision_service_evidence.py`，针对 `_prediction_evidence` / `_brep_evidence` / `_top_brep_hint` 写 5–8 个边界 case（estimate: 2h | risk: low）

### 3.B Readiness registry 缓存失效

- [ ] **S3.3** 扩展 `tests/unit/test_model_readiness_registry.py`：写入文件 → 拿 checksum → 改文件 size → 再拿应不同；改 mtime（保 size）→ 同上（estimate: 1h | risk: low）

### 3.C Forward scorecard 边界

- [ ] **S3.4** 扩展 `tests/unit/test_forward_scorecard.py`：参数化 `_metric_status`，4 status × sample_size {0, 9, 10, 29, 30, 50, 100} × threshold 三档（estimate: 2h | risk: low）
- [ ] **S3.5** 扩展 `tests/unit/test_forward_scorecard_release_gate.py`：缺 component 时必须 `blocked`，**禁止** 默认空 dict 假绿（estimate: 1.5h | risk: med — 可能暴露既有 bug）

### 3.D Observability（可选）

- [ ] **S3.6** 在 `forward_scorecard` 关键 status 变化处 emit 一个 metric（`forward_scorecard_status` gauge，labels=`component,status`）—— **只在已有 metrics 注册中心存在时做**（estimate: 2h | risk: med — 评估必要性）

---

## 治理 / 文档

- [ ] **G.1** 本三份 MD 落地后，将其路径加入 `MEMORY.md` 入口 + memory 文件指向 Stage 2c 完成状态（estimate: 0.3h | risk: low）
- [ ] **G.2** Stage 1 完成后，更新 [[ci-stacked-pr-gates-dormant]] memory（标"已解除"或保留作历史记录）（estimate: 0.1h | risk: low）
- [ ] **G.3** 每个 Stage 完成时同步更新本 TODO 的勾选状态（estimate: 0.1h × N | risk: low）

---

## 显式不做（防自走偏）

- [ ] ~~Phase 7 任何代码~~ — hand-off §4，等 Stages 1+2 scorecard 真实绿
- [ ] ~~蒸馏复活实验~~ — B6.5 已结论
- [ ] ~~在 Stage 2a/2b 完成前接 scorecard~~ — 验证 §5 明令禁止
- [ ] ~~force-push 堆叠分支~~ — 用追加 commit 推进
- [ ] ~~把 reviewed manifest 放 `reports/benchmark/`~~ — Stage 0 的 `.gitignore` 会吃掉

---

## 进度总览（手动维护）

```
Stage 0  [ ] 0/9
Stage 1  [ ] 0/14
Stage 2a [ ] 0/14
Stage 2b [ ] 0/8
Stage 2c [ ] 0/10
Stage 3  [ ] 0/6
治理     [ ] 0/3
─────────────────────
合计     0/64
```

完成每条后把 `[ ]` 改 `[x]` 并刷新此处总览。
