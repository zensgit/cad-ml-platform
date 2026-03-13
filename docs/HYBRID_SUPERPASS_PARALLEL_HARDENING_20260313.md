# Hybrid Superpass 并行强化开发与验证（2026-03-13）

## 目标
- 提升 `hybrid-superpass` 在 GitHub Actions 上的并行稳定性与可观测性。
- 增加 fail/success 双场景自动对比能力，降低“串单/误判”风险。
- 增加 nightly 巡检 workflow，并纳入 Makefile 一键化入口与单测保护。

## 本次实现

### 1. E2E workflow 兼容性强化
- 文件：`.github/workflows/hybrid-superpass-e2e.yml`
- 改动：
  - 增加 `env.FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true`，规避 Node 运行时兼容告警/漂移风险。

### 2. 双场景对比脚本（新增）
- 文件：`scripts/ci/compare_hybrid_superpass_reports.py`
- 能力：
  - 输入 fail/success 两份 dispatch JSON。
  - 提取并标准化字段：`run_id`、`conclusion`、`expected_conclusion`、`matched_expectation`、`dispatch_trace_id`、`run_url`。
  - 输出 machine JSON 总结 + Markdown 对比报告。
  - 检查 `run_id` 是否不同（并行隔离校验）。
  - `--strict` 下若任一场景不符合预期则返回非零退出码。

### 3. Nightly workflow（新增）
- 文件：`.github/workflows/hybrid-superpass-nightly.yml`
- 改动：
  - 新增 `schedule + workflow_dispatch` 双触发模式。
  - 先后 dispatch fail/success 两个场景。
  - 使用 `compare_hybrid_superpass_reports.py --strict` 生成 nightly 对比结论。
  - 上传 fail/success/compare 产物并写入 step summary。
  - 同步设置 `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true`。

### 4. Makefile 一键链路强化
- 文件：`Makefile`
- 新增目标：
  - `hybrid-superpass-compare`
  - `hybrid-superpass-e2e-dual-gh`
  - `hybrid-superpass-nightly-gh`
  - `validate-hybrid-superpass-nightly-workflow`
- 新增变量：
  - `HYBRID_SUPERPASS_DUAL_FAIL_JSON`
  - `HYBRID_SUPERPASS_DUAL_SUCCESS_JSON`
  - `HYBRID_SUPERPASS_COMPARE_FAIL_JSON`
  - `HYBRID_SUPERPASS_COMPARE_SUCCESS_JSON`
  - `HYBRID_SUPERPASS_COMPARE_OUTPUT_JSON`
  - `HYBRID_SUPERPASS_COMPARE_OUTPUT_MD`
  - `HYBRID_SUPERPASS_COMPARE_STRICT`
  - `HYBRID_SUPERPASS_NIGHTLY_WORKFLOW`
  - `HYBRID_SUPERPASS_NIGHTLY_REF`
  - `HYBRID_SUPERPASS_NIGHTLY_REPO`
  - `HYBRID_SUPERPASS_NIGHTLY_PRINT_ONLY`
- 修复：
  - `hybrid-superpass-e2e-dual-gh` 对 `--repo` 改为条件透传，避免空 repo 参数导致潜在 dispatch 风险。

### 5. 测试覆盖（新增/更新）
- 新增：
  - `tests/unit/test_compare_hybrid_superpass_reports.py`
  - `tests/unit/test_hybrid_superpass_nightly_workflow.py`
- 更新：
  - `tests/unit/test_hybrid_superpass_e2e_workflow.py`
  - `tests/unit/test_graph2d_parallel_make_targets.py`
- 覆盖点：
  - compare 脚本正常路径、run_id 冲突告警、strict 失败退出。
  - nightly workflow 触发器、权限、关键步骤与参数。
  - Make 目标参数透传与回归检查。

## 验证结果

### 命令 1
```bash
make validate-hybrid-superpass-workflow
```
- 结果：`58 passed, 1 warning`
- warning：第三方依赖 `starlette/formparsers.py` 的 `PendingDeprecationWarning`，与本次改动无关。

### 命令 2
```bash
make validate-hybrid-superpass-nightly-workflow
```
- 结果：`18 passed, 1 warning`
- warning 同上（依赖侧）。

## 风险与后续建议
- 已消除 nightly 对“占位 compare 脚本”的依赖，改为真实 compare 脚本 + strict 判定。
- 建议下一步在主仓执行一次真实手动 dispatch：
  - `make hybrid-superpass-e2e-dual-gh`
  - 核验输出的 compare JSON/MD 与 Actions run 链接一致性。

## 增量强化（同日第二轮）

### 新增 nightly dispatcher 脚本
- 文件：`scripts/ci/dispatch_hybrid_superpass_nightly_workflow.py`
- 能力：
  - 支持 `workflow_dispatch` 的完整参数透传（repo/ref/target_*）。
  - 支持 `--print-only`、`--output-json`、`--expected-conclusion`、`watch`。
  - 自动生成 `dispatch_trace_id`（`nsp-<12hex>`），并用于并发 run 识别过滤。

### Nightly workflow traceability 强化
- 文件：`.github/workflows/hybrid-superpass-nightly.yml`
- 改动：
  - 新增 `run-name`（包含可选 `dispatch_trace_id`）。
  - 新增 `workflow_dispatch.inputs.dispatch_trace_id`。
  - 新增 `NIGHTLY_TRACE_ID` 环境变量（输入优先，默认 `nsp-${{ github.run_id }}`）。
  - fail/success dispatch 分别传递 `--dispatch-trace-id "${NIGHTLY_TRACE_ID}-fail/-success"`。
  - step summary 增加 Trace ID 输出。

### Makefile 夜检入口升级
- 文件：`Makefile`
- 改动：
  - `hybrid-superpass-nightly-gh` 改为调用 `dispatch_hybrid_superpass_nightly_workflow.py`，支持等待与结果判定。
  - 新增夜检变量：`HYBRID_SUPERPASS_NIGHTLY_TARGET_*`、`HYBRID_SUPERPASS_NIGHTLY_DISPATCH_TRACE_ID`、`HYBRID_SUPERPASS_NIGHTLY_TIMEOUT/POLL/LIST/OUTPUT_JSON/EXPECTED_CONCLUSION`。
  - `validate-hybrid-superpass-nightly-workflow` 纳入 `test_dispatch_hybrid_superpass_nightly_workflow.py`。

### 增量验证结果
- `make validate-hybrid-superpass-workflow`：`59 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`24 passed, 1 warning`
