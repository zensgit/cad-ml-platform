# Hybrid Superpass Mainline Sync - Design & Validation (2026-03-13)

## 1. Goal

在 `origin/main` 最新基线上完成 Hybrid Superpass 的可合并接入，避免历史分支大冲突，并保证：

- workflow_dispatch 可控（新增 superpass 开关/模式/strict 覆盖）。
- evaluation workflow 具备 superpass gate + strict fail + artifact + summary。
- Makefile 具备本地 gate、GH E2E dispatch、GH vars apply 与一键验证入口。
- 配套脚本和测试可直接执行。

## 2. Implementation Strategy

采用“最小增量移植”而非整体 cherry-pick：

1. 先迁移独立新增文件（scripts/config/tests）。
2. 再在 `main` 当前版 `Makefile` 与 `evaluation-report.yml` 上打定点补丁。
3. 最后运行脚本层 + workflow wiring + Makefile target 的回归测试。

这样可以规避 `main` 上长期演进导致的大面积冲突。

## 3. Changed Files

### Workflow / Build

- `.github/workflows/evaluation-report.yml`
- `.github/workflows/hybrid-superpass-e2e.yml`
- `Makefile`

### New Config

- `config/hybrid_superpass_targets.yaml`

### New Scripts

- `scripts/ci/check_hybrid_superpass_targets.py`
- `scripts/ci/dispatch_hybrid_superpass_workflow.py`
- `scripts/ci/apply_hybrid_superpass_gh_vars.py`

### Tests

- `tests/unit/test_check_hybrid_superpass_targets.py`
- `tests/unit/test_dispatch_hybrid_superpass_workflow.py`
- `tests/unit/test_apply_hybrid_superpass_gh_vars.py`
- `tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py`
- `tests/unit/test_hybrid_superpass_workflow_integration.py`
- `tests/unit/test_graph2d_parallel_make_targets.py` (追加 superpass make target 断言)

## 4. Workflow Wiring (What was added)

### 4.1 workflow_dispatch inputs

新增：

- `hybrid_superpass_enable`
- `hybrid_superpass_missing_mode`
- `hybrid_superpass_fail_on_failed`

### 4.2 env variables

新增：

- `HYBRID_SUPERPASS_ENABLE`
- `HYBRID_SUPERPASS_CONFIG`
- `HYBRID_SUPERPASS_OUTPUT_JSON`
- `HYBRID_SUPERPASS_MISSING_MODE`
- `HYBRID_SUPERPASS_GATE_REPORT_JSON`
- `HYBRID_SUPERPASS_CALIBRATION_JSON`
- `HYBRID_SUPERPASS_FAIL_ON_FAILED`

### 4.3 evaluate job steps

新增：

- `Check Hybrid superpass gate (optional)`
- `Evaluate Hybrid superpass strict mode (optional)`
- `Upload Hybrid superpass gate artifact`
- `Fail workflow when Hybrid superpass strict check requires blocking`

并在 `Create job summary` 中新增 superpass 状态输出（status/headline/report/missing_mode/strict flags）。

### 4.4 Dedicated Dispatch Workflow

新增轻量 dispatch 工作流：`hybrid-superpass-e2e.yml`，用于规避 `evaluation-report.yml` 体量过大导致的 `workflow_dispatch` 解析失败风险。  
`dispatch_hybrid_superpass_workflow.py` 与 Make 默认 workflow 已切换到该文件，可按需通过 `--workflow` 覆盖。

## 5. Makefile Targets

新增：

- `hybrid-superpass-gate`
- `hybrid-superpass-e2e-gh`
- `hybrid-superpass-apply-gh-vars`
- `validate-hybrid-superpass-workflow`

新增相关变量：

- `HYBRID_SUPERPASS_*`（gate / e2e / apply 三组）

## 6. Validation Commands and Results

在干净 worktree 执行：

```bash
python3 -m pytest tests/unit/test_check_hybrid_superpass_targets.py tests/unit/test_dispatch_hybrid_superpass_workflow.py tests/unit/test_apply_hybrid_superpass_gh_vars.py -q
# 15 passed

python3 -m pytest tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py tests/unit/test_hybrid_superpass_workflow_integration.py tests/unit/test_graph2d_parallel_make_targets.py -q
# 17 passed

make validate-hybrid-superpass-workflow
# 51 passed, 1 warning (PendingDeprecationWarning from python_multipart import path)
```

结论：superpass 相关脚本、Makefile 编排与 workflow 接线均通过。

## 7. Notes

- 本次以 `main` 当前结构为基线做增量补丁，避免了将旧分支里大量不相关历史变更带入。
- `dispatch_hybrid_superpass_workflow.py` 包含 remote workflow input 预检与 `--skip-remote-input-check` 兜底参数，便于排查远端分支未同步场景。
- 线上 `evaluation-report.yml` 当前存在 `workflow_dispatch` 表达式长度限制问题（HTTP 422 / Exceeded max expression length 21000）；本次通过新增专用 workflow 避免阻塞 superpass E2E。
- GitHub 仅允许 dispatch 默认分支已存在的 workflow 文件。PR 分支内新增的 `hybrid-superpass-e2e.yml` 在合入 `main` 前会返回 HTTP 404（workflow not found on default branch），脚本已给出明确提示与 reason 标记。

## 8. Suggested Next Step

1. 在该分支提交并推送。
2. 使用 `make hybrid-superpass-e2e-gh ...` 跑一次 strict fail / success 双场景 GH 验证。
3. 创建 PR 到 `main` 并在描述中附上本 MD。

## 9. Post-Merge Runtime Validation (Main Branch)

在 `main` 上执行了真实 dispatch 验证：

- 失败场景（`missing_mode=fail`, `fail_on_failed=true`）  
  run: `23038415772`  
  url: `https://github.com/zensgit/cad-ml-platform/actions/runs/23038415772`  
  结果：`failure`，与预期一致。

- 成功场景（`missing_mode=skip`, `fail_on_failed=false`）  
  run: `23038473346`  
  url: `https://github.com/zensgit/cad-ml-platform/actions/runs/23038473346`  
  结果：`success`，与预期一致。

并发 dispatch 时发现 run 归属竞态（两个客户端可能绑定同一 run）。已补充修复：

1. `hybrid-superpass-e2e.yml` 新增 `dispatch_trace_id` input，`run-name` 带 trace。
2. `dispatch_hybrid_superpass_workflow.py` 新增 trace 注入与 run 过滤匹配逻辑。
3. 新增单测覆盖 trace 构建与过滤路径。
