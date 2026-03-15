# Hybrid Superpass Dispatch 失败根因诊断增强（2026-03-15）

## 目标

在 `scripts/ci/dispatch_hybrid_superpass_workflow.py` 中补齐失败诊断能力，解决以下问题：

- workflow 结论不符合预期时，只看到 `failure`，缺少失败 job/step 线索；
- 需要手工进入 Actions 页面逐步点开 job 才能定位首个失败点；
- 输出 JSON 缺少可机器消费的诊断字段，不利于后续自动告警/聚合。

## 实现内容

### 1) 新增失败诊断提取能力

- 新增函数：
  - `_is_failed_conclusion`
  - `summarize_failed_jobs`
  - `fetch_run_failure_diagnostics`
- 诊断数据来源：
  - `gh run view <run_id> --json jobs`
- 诊断摘要输出：
  - `total_jobs`
  - `failed_job_count`
  - `failed_jobs`（包含 `job_name/job_conclusion/job_url/failed_step_name/failed_step_conclusion`）
  - `failed_jobs_truncated`
  - `available/reason`（诊断可用性与失败原因）

### 2) 主流程接入

- 当 `matched_expectation == false`（即 `overall_exit_code != 0`）时：
  - 自动抓取运行诊断并写入 `payload["failure_diagnostics"]`；
  - 控制台打印首个失败 job/step 摘要，便于第一时间定位。

## 测试补强

更新文件：`tests/unit/test_dispatch_hybrid_superpass_workflow.py`

- 新增 `test_summarize_failed_jobs_extracts_first_failed_step`
  - 校验从 jobs payload 提取首个失败 step。
- 新增 `test_main_mismatch_writes_failure_diagnostics`
  - 校验 `main` 在结论不匹配时输出 `failure_diagnostics` 到 `--output-json`。

## 验证结果

### 定向单测 + 代码风格

```bash
pytest -q tests/unit/test_dispatch_hybrid_superpass_workflow.py
flake8 --max-line-length=100 \
  scripts/ci/dispatch_hybrid_superpass_workflow.py \
  tests/unit/test_dispatch_hybrid_superpass_workflow.py
```

结果：

- `9 passed, 1 warning`
- flake8 通过

### Hybrid Superpass 回归集

```bash
make validate-hybrid-superpass-workflow
```

结果：

- `60 passed`

## 风险与回滚

- 新增诊断仅在失败路径触发，不影响成功路径行为；
- 若需回滚，可仅移除 `failure_diagnostics` 相关逻辑，不影响 dispatch/watch 核心流程。
