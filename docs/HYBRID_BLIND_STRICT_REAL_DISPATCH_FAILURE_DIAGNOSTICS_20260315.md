# Hybrid Blind Strict-Real Dispatch 失败根因诊断增强（2026-03-15）

## 目标

将 `scripts/ci/dispatch_hybrid_blind_strict_real_workflow.py` 与 superpass dispatch 对齐，补齐失败自动诊断：

- 失败时自动提取 GitHub Actions jobs 失败摘要；
- 输出 machine-readable 的 `failure_diagnostics`；
- 控制台打印首个失败 job/step，缩短排障路径。

## 实现内容

### 1) 新增诊断能力

- 新增函数：
  - `_is_failed_conclusion`
  - `summarize_failed_jobs`
  - `fetch_run_failure_diagnostics`
- 调用来源：
  - `gh run view <run_id> --json jobs`
- 输出字段：
  - `available`
  - `reason`
  - `total_jobs`
  - `failed_job_count`
  - `failed_jobs`
  - `failed_jobs_truncated`

### 2) 主流程接入

- 当 `overall_exit_code != 0`（结论不匹配）时：
  - 写入 `payload["failure_diagnostics"]`；
  - 输出首个失败 job/step 诊断日志。

## 测试与验证

### 新增单测

更新文件：`tests/unit/test_dispatch_hybrid_blind_strict_real_workflow.py`

- `test_summarize_failed_jobs_extracts_first_failed_step`
- `test_main_mismatch_writes_failure_diagnostics`

### 运行结果

```bash
pytest -q tests/unit/test_dispatch_hybrid_blind_strict_real_workflow.py
flake8 --max-line-length=100 \
  scripts/ci/dispatch_hybrid_blind_strict_real_workflow.py \
  tests/unit/test_dispatch_hybrid_blind_strict_real_workflow.py
make validate-hybrid-blind-workflow
```

结果：

- `10 passed, 1 warning`
- flake8 通过
- `make validate-hybrid-blind-workflow`：`90 passed`

## 风险与回滚

- 增强逻辑仅挂在失败路径，不改变成功路径；
- 回滚只需移除 `failure_diagnostics` 相关逻辑，不影响 dispatch/watch 主链路。
