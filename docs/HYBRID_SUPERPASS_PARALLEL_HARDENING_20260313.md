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

## 增量强化（同日第三轮：并发提速）

### 并发双场景 orchestrator（新增）
- 文件：`scripts/ci/run_hybrid_superpass_dual_dispatch.py`
- 能力：
  - fail/success 两个 dispatch 子进程并发执行（`subprocess.Popen`）。
  - 两者结束后统一执行 compare（`compare_hybrid_superpass_reports.py`）。
  - 支持 `--strict`、`--print-only`、`--dispatch-trace-prefix`。
  - 输出统一汇总字段：三条命令、各子任务 exit code、overall_exit_code。
  - 自动 trace 前缀：`dsp-<12hex>`，并派生 `-fail/-success`。

### Makefile 并发入口切换
- 文件：`Makefile`
- 改动：
  - `hybrid-superpass-e2e-dual-gh` 改为并发 orchestrator 路径。
  - 保留 `hybrid-superpass-e2e-dual-gh-sequential` 作为回退模式。
  - 新增并发汇总变量：`HYBRID_SUPERPASS_DUAL_PARALLEL_SUMMARY_JSON`、`HYBRID_SUPERPASS_DUAL_DISPATCH_TRACE_PREFIX`。
  - `validate-hybrid-superpass-workflow` 纳入 orchestrator 单测。

### 增量测试
- 新增：`tests/unit/test_run_hybrid_superpass_dual_dispatch.py`
- 更新：`tests/unit/test_graph2d_parallel_make_targets.py`

### 第三轮验证结果
- `pytest tests/unit/test_run_hybrid_superpass_dual_dispatch.py tests/unit/test_graph2d_parallel_make_targets.py -q`：`21 passed, 1 warning`
- `make validate-hybrid-superpass-workflow`：`64 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`25 passed, 1 warning`

## 增量强化（同日第四轮：nightly 编排收敛）

### Nightly workflow 改为单步 orchestrator
- 文件：`.github/workflows/hybrid-superpass-nightly.yml`
- 改动：
  - 将原先 `fail dispatch + success dispatch + compare` 三步，收敛为单步 `run_hybrid_superpass_dual_dispatch.py`。
  - 继续保留 strict compare 语义，且产物路径保持兼容（`FAIL_JSON/SUCCESS_JSON/COMPARE_JSON/COMPARE_MD`）。
  - 新增 `DUAL_SUMMARY_JSON` 产物，step summary 增加 dual step outcome 与 dual summary 路径。

### 对应单测更新
- 文件：`tests/unit/test_hybrid_superpass_nightly_workflow.py`
- 改动：
  - 从三步断言改为单步 orchestrator 断言。
  - 校验 dual summary artifact 与 summary 输出字段。

### 第四轮验证结果
- `pytest tests/unit/test_hybrid_superpass_nightly_workflow.py tests/unit/test_run_hybrid_superpass_dual_dispatch.py tests/unit/test_graph2d_parallel_make_targets.py -q`：`24 passed, 1 warning`
- `make validate-hybrid-superpass-workflow`：`64 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`25 passed, 1 warning`

## 增量强化（同日第五轮：并发隔离硬门禁）

### Compare 脚本增强（严格模式可强制 run_id 不同）
- 文件：`scripts/ci/compare_hybrid_superpass_reports.py`
- 新增参数：
  - `--strict-require-distinct-run-ids`
- 行为：
  - 当 `--strict` + `--strict-require-distinct-run-ids` 同时启用时，若 fail/success `run_id` 相同（或无法判定为不同）则直接失败（exit 1）。
  - 报告中新增该策略开关状态。

### 并发 orchestrator 与流程默认启用该门禁
- 文件：
  - `scripts/ci/run_hybrid_superpass_dual_dispatch.py`
  - `Makefile`
  - `.github/workflows/hybrid-superpass-nightly.yml`
- 改动：
  - orchestrator 支持透传 `--strict-require-distinct-run-ids` 到 compare。
  - Makefile 新增 `HYBRID_SUPERPASS_COMPARE_STRICT_REQUIRE_DISTINCT_RUN_IDS ?= 1`（默认开启）。
  - nightly workflow 的 dual-dispatch 步骤默认携带该参数。

### 测试覆盖增强
- 文件：
  - `tests/unit/test_compare_hybrid_superpass_reports.py`
  - `tests/unit/test_run_hybrid_superpass_dual_dispatch.py`
  - `tests/unit/test_graph2d_parallel_make_targets.py`
  - `tests/unit/test_hybrid_superpass_nightly_workflow.py`
- 新增断言：
  - strict + require-distinct 时 run_id 相同必须失败。
  - orchestrator/Make/nightly workflow 都包含新参数透传。

### 第五轮验证结果
- `pytest tests/unit/test_compare_hybrid_superpass_reports.py tests/unit/test_run_hybrid_superpass_dual_dispatch.py tests/unit/test_graph2d_parallel_make_targets.py tests/unit/test_hybrid_superpass_nightly_workflow.py -q`：`28 passed, 1 warning`
- `make validate-hybrid-superpass-workflow`：`65 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`25 passed, 1 warning`

## 增量强化（同日第六轮：trace 配对硬门禁）

### Compare 增强（严格模式可强制 trace 成对）
- 文件：`scripts/ci/compare_hybrid_superpass_reports.py`
- 新增参数：
  - `--strict-require-trace-pair`
- 规则：
  - fail 的 `dispatch_trace_id` 必须为 `<prefix>-fail`
  - success 的 `dispatch_trace_id` 必须为 `<prefix>-success`
  - 两者 `<prefix>` 必须一致
- strict 行为：
  - 当 `--strict` + `--strict-require-trace-pair` 启用且不满足规则时，直接失败（exit 1）。
- 输出增强：
  - `checks.trace_pair_consistent`
  - `strict_require_trace_pair`
  - markdown 增加 trace 配对结论和该 strict 开关状态。

### Orchestrator / Make / Nightly 透传
- 文件：
  - `scripts/ci/run_hybrid_superpass_dual_dispatch.py`
  - `Makefile`
  - `.github/workflows/hybrid-superpass-nightly.yml`
- 改动：
  - orchestrator 新增 `--strict-require-trace-pair` 透传到 compare。
  - Makefile：
    - 新增 `HYBRID_SUPERPASS_DUAL_STRICT_REQUIRE_TRACE_PAIR ?= 1`
    - 新增 `HYBRID_SUPERPASS_COMPARE_STRICT_REQUIRE_TRACE_PAIR ?= 0`
    - dual 目标默认强制 trace 配对门禁；compare 目标可按需开启。
  - nightly workflow dual 步骤默认开启 `--strict-require-trace-pair`。

### 测试覆盖增强
- 文件：
  - `tests/unit/test_compare_hybrid_superpass_reports.py`
  - `tests/unit/test_run_hybrid_superpass_dual_dispatch.py`
  - `tests/unit/test_graph2d_parallel_make_targets.py`
  - `tests/unit/test_hybrid_superpass_nightly_workflow.py`
- 新增断言：
  - strict+trace-pair 正常配对通过、异常配对失败。
  - orchestrator/Make/nightly 都透传新参数。

### 第六轮验证结果
- `pytest tests/unit/test_compare_hybrid_superpass_reports.py tests/unit/test_run_hybrid_superpass_dual_dispatch.py tests/unit/test_graph2d_parallel_make_targets.py tests/unit/test_hybrid_superpass_nightly_workflow.py -q`：`31 passed, 1 warning`
- `make validate-hybrid-superpass-workflow`：`68 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`26 passed, 1 warning`

## 增量强化（同日第七轮：nightly 参数化加速）

### Nightly workflow 参数化
- 文件：`.github/workflows/hybrid-superpass-nightly.yml`
- 新增 `workflow_dispatch` 输入：
  - `dual_wait_timeout_seconds`（默认 `900`）
  - `dual_poll_interval_seconds`（默认 `3`）
  - `dual_list_limit`（默认 `20`）
  - `strict_require_distinct_run_ids`（默认 `true`）
  - `strict_require_trace_pair`（默认 `true`）
- dual step 改动：
  - `--wait-timeout-seconds/--poll-interval-seconds/--list-limit` 读取 env 映射参数。
  - 使用 shell 条件逻辑按 strict 策略值决定是否追加：
    - `--strict-require-distinct-run-ids`
    - `--strict-require-trace-pair`
- summary 增加上述参数与策略输出，便于排障。

### Nightly dispatcher 参数透传
- 文件：`scripts/ci/dispatch_hybrid_superpass_nightly_workflow.py`
- 新增 CLI 参数并透传到 `gh workflow run -f`：
  - `--dual-wait-timeout-seconds`
  - `--dual-poll-interval-seconds`
  - `--dual-list-limit`
  - `--strict-require-distinct-run-ids`
  - `--strict-require-trace-pair`
- `print-only` 输出格式保持兼容。

### Make 夜检入口扩展
- 文件：`Makefile`
- 新增变量：
  - `HYBRID_SUPERPASS_NIGHTLY_DUAL_WAIT_TIMEOUT`
  - `HYBRID_SUPERPASS_NIGHTLY_DUAL_POLL_INTERVAL`
  - `HYBRID_SUPERPASS_NIGHTLY_DUAL_LIST_LIMIT`
  - `HYBRID_SUPERPASS_NIGHTLY_STRICT_REQUIRE_DISTINCT_RUN_IDS`
  - `HYBRID_SUPERPASS_NIGHTLY_STRICT_REQUIRE_TRACE_PAIR`
- `hybrid-superpass-nightly-gh` 透传上述参数到 dispatcher，实现不改代码调参。

### 测试覆盖增强
- 文件：
  - `tests/unit/test_dispatch_hybrid_superpass_nightly_workflow.py`
  - `tests/unit/test_hybrid_superpass_nightly_workflow.py`
  - `tests/unit/test_graph2d_parallel_make_targets.py`
- 覆盖点：
  - 输入默认值、env 映射、命令透传、summary 输出。
  - nightly Make 目标含 dual 参数与 strict 参数透传。

### 第七轮验证结果
- `pytest tests/unit/test_compare_hybrid_superpass_reports.py tests/unit/test_run_hybrid_superpass_dual_dispatch.py tests/unit/test_dispatch_hybrid_superpass_nightly_workflow.py tests/unit/test_hybrid_superpass_nightly_workflow.py tests/unit/test_graph2d_parallel_make_targets.py -q`：`36 passed, 1 warning`
- `make validate-hybrid-superpass-workflow`：`68 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`26 passed, 1 warning`

## 增量强化（同日第八轮：dual summary 渲染与夜检可读性）

### 新增 dual summary 渲染脚本
- 文件：`scripts/ci/render_hybrid_superpass_dual_summary.py`
- 能力：
  - 读取 `dual-summary.json`（必需）和 `compare.json`（可选）。
  - 产出统一 markdown，包含：
    - Exit codes（overall/fail/success/compare）
    - Key checks（run_id/trace/strict 等）
    - fail/success run 信息
    - 可选附加 compare markdown 内容。
  - dual json 非法时返回非零，便于 CI 直接失败。

### Nightly workflow 接入渲染步骤
- 文件：`.github/workflows/hybrid-superpass-nightly.yml`
- 改动：
  - 新增 `DUAL_SUMMARY_MD` 产物路径。
  - 新增步骤 `Render nightly dual summary markdown`。
  - artifact 增加 `DUAL_SUMMARY_MD`。
  - step summary 优先附加 `DUAL_SUMMARY_MD`，提升可读性与排障效率。

### Nightly 验证链纳入渲染器单测
- 文件：
  - `Makefile`
  - `tests/unit/test_graph2d_parallel_make_targets.py`
- 改动：
  - `validate-hybrid-superpass-nightly-workflow` 新增 `test_render_hybrid_superpass_dual_summary.py`。
  - Make 目标断言同步更新。

### 测试覆盖增强
- 文件：
  - `tests/unit/test_render_hybrid_superpass_dual_summary.py`
  - `tests/unit/test_hybrid_superpass_nightly_workflow.py`
  - `tests/unit/test_dispatch_hybrid_superpass_nightly_workflow.py`
  - `tests/unit/test_graph2d_parallel_make_targets.py`

### 第八轮验证结果
- `pytest tests/unit/test_render_hybrid_superpass_dual_summary.py tests/unit/test_dispatch_hybrid_superpass_nightly_workflow.py tests/unit/test_hybrid_superpass_nightly_workflow.py tests/unit/test_graph2d_parallel_make_targets.py -q`：`29 passed, 1 warning`
- `make validate-hybrid-superpass-workflow`：`68 passed, 1 warning`
- `make validate-hybrid-superpass-nightly-workflow`：`29 passed, 1 warning`
