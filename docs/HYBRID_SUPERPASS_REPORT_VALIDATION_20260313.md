# Hybrid Superpass 报表结构校验增强（2026-03-13）

## 目标

在现有 `Hybrid superpass` gate 之外，新增一层“报表结构校验”，避免出现：

- gate 产出 JSON 存在但字段不完整/类型错误；
- 上游 `hybrid_blind_gate` 与 `hybrid_calibration` 报表格式漂移未被发现；
- CI summary 缺少结构健康状态，定位成本高。

## 设计与实现

### 1) 新增校验脚本

- 文件：`scripts/ci/validate_hybrid_superpass_reports.py`
- 输入参数：
  - `--superpass-json`（必选）
  - `--hybrid-blind-gate-report`（可选）
  - `--hybrid-calibration-json`（可选）
  - `--output-json`（可选）
  - `--strict`（可选）
- 校验范围：
  - `superpass` 必备字段与类型：
    - `status`(str), `headline`(str), `thresholds`(dict),
      `checks`(list), `failures`(list), `warnings`(list)
  - `checks[*]` 基础结构（`name` / `passed`）
  - 可选输入提取：
    - `metrics.hybrid_accuracy`
    - `metrics.hybrid_gain_vs_graph2d`
    - `metrics_after.ece`
- 输出：
  - stdout 打印 JSON
  - 可选写入 `--output-json`
  - 统一结果结构：
    - `status: ok|warn|error`
    - `errors: []`
    - `warnings: []`
    - `summary: {}`

### 2) 接入 evaluation workflow

- 文件：`.github/workflows/evaluation-report.yml`
- 新增环境变量：
  - `HYBRID_SUPERPASS_VALIDATION_JSON`
- 新增步骤：
  - `Validate Hybrid superpass report structure (optional)`
- 行为：
  - 在 superpass gate 启用后执行；
  - 当 superpass strict 模式启用时，自动透传 `--strict` 给校验脚本；
  - 读取 gate 输出报表并进行结构校验；
  - 输出状态、headline、warning/error 数量到 step outputs；
  - strict 模式下，若结构校验返回非 0，新增阻断步骤失败 workflow；
  - 将校验 JSON 一并纳入 artifact。

### 3) 测试与 Make 集成

- 新增：
  - `tests/unit/test_validate_hybrid_superpass_reports.py`
- 更新：
  - `tests/unit/test_hybrid_superpass_workflow_integration.py`
  - `tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py`
  - `tests/unit/test_graph2d_parallel_make_targets.py`
  - `Makefile` 的 `validate-hybrid-superpass-workflow` 目标加入新测试文件

## 验证结果

### 定向单测

```bash
pytest -q \
  tests/unit/test_validate_hybrid_superpass_reports.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_graph2d_parallel_make_targets.py
```

结果：`20 passed`

### 目标回归验证

```bash
make validate-hybrid-superpass-workflow
```

结果：`56 passed`

## 回滚与风险控制

- 新增逻辑默认是“附加验证”，不会替代现有 superpass gate。
- 若需快速回退：
  - 可移除/跳过 `Validate Hybrid superpass report structure (optional)` 步骤；
  - 其余 gate/strict 逻辑不受影响。
