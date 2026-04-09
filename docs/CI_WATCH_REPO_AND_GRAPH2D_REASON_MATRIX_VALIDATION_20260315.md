# CI Watch Repo Support + Graph2D Reason Matrix Validation (2026-03-15)

## 目标

在现有 CI watcher 与 Graph2D 规则增强基础上，继续推进三项并行优化：

1. 为 `watch_commit_workflows.py` 增加跨仓库 `--repo` 能力。  
2. 将失败详情诊断能力接入默认调用链（可配置关闭）。  
3. 扩展 `analyze.py` 中 Graph2D soft-override 原因分支矩阵测试覆盖。

## 变更清单

### 1) 跨仓库 watcher 支持

- 文件: `scripts/ci/watch_commit_workflows.py`
- 新增参数:
  - `--repo`（`owner/repo`）
- 生效范围:
  - `gh run list ... --repo <repo>`
  - `gh run view ... --repo <repo>`
  - print-only 输出增加 `# repo=...`
  - summary JSON 增加 `repo` 字段

### 2) 默认失败诊断能力接入 Make 调用链

- 文件: `Makefile`
- 新增变量:
  - `CI_WATCH_REPO ?=`
- 调整默认:
  - `CI_WATCH_PRINT_FAILURE_DETAILS ?= 1`
- 透传参数:
  - `--repo "$(CI_WATCH_REPO)"`
  - `--failure-details-max-runs "$(CI_WATCH_FAILURE_DETAILS_MAX_RUNS)"`
  - `--print-failure-details`（默认开启，可通过 `CI_WATCH_PRINT_FAILURE_DETAILS=0` 关闭）

### 3) Graph2D reason 矩阵测试增强

- 文件: `tests/unit/test_analyze_graph2d_gate_helpers.py`
- 新增覆盖:
  - `graph2d_result=None` 返回 `None`
  - `status=model_unavailable` -> `graph2d_unavailable`
  - reason matrix:
    - `confidence_source_not_rules`
    - `rule_version_not_v1`
    - `graph2d_excluded`
    - `graph2d_not_allowed`
    - `graph2d_drawing_type`
    - `graph2d_coarse_label`
    - `below_margin`
    - `below_threshold`
  - 完整 eligible 正例路径

## 测试与验证

```bash
pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py
```
- 结果: `42 passed`

```bash
pytest -q tests/unit/test_analyze_graph2d_gate_helpers.py
```
- 结果: `16 passed`

```bash
make validate-ci-watchers
```
- 结果: 全通过（含 watcher / dispatcher / eval-with-history / graph2d strict e2e 链路）

## 使用示例

```bash
make watch-commit-workflows \
  CI_WATCH_REPO=zensgit/cad-ml-platform \
  CI_WATCH_SHA=HEAD \
  CI_WATCH_PRINT_FAILURE_DETAILS=1 \
  CI_WATCH_FAILURE_DETAILS_MAX_RUNS=5
```

说明:
- 当检测到失败结论时，会自动输出失败 job/step 摘要，减少手工 `gh run view` 排障时间。
