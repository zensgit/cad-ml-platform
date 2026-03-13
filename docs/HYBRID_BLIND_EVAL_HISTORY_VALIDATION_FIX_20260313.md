# Hybrid Blind Eval History Validation Fix (2026-03-13)

## 背景
- `evaluation-report.yml` 在 `Validate history with JSON Schema` 步骤中会扫描 `reports/eval_history/*.json`。
- 新增的 sidecar 报告 JSON（如 `hybrid_blind_drift_alert_report.json`）并非历史快照格式，导致被错误校验并阻塞流程。
- 同时，bootstrap/归档脚本在部分场景会写入非哈希 `commit`（如 `bootstrap`），不符合 `docs/eval_history.schema.json` 约束。

## 修复内容
1. `scripts/validate_eval_history.py`
- 新增 `--exclude-glob` 参数（可重复），支持显式排除 sidecar JSON。
- 默认排除：
  - `hybrid_blind_drift_alert_report.json`
  - `hybrid_blind_drift_threshold_suggestion.json`
- 输出中增加跳过文件计数，保持可观测性。

2. `scripts/ci/archive_hybrid_blind_eval_history.py`
- 新增 commit 规范化逻辑：仅允许 `[a-f0-9]{6,40}` 或 `[redacted]`。
- 非法/非哈希 commit 自动归一为 `[redacted]`，保证 schema 合法。

3. `scripts/ci/bootstrap_hybrid_blind_eval_history.py`
- 同步 commit 规范化，写出的历史快照默认使用合法 commit 值（如 `[redacted]`）。

4. `.github/workflows/evaluation-report.yml`
- 历史校验步骤显式传入：
  - `--exclude-glob hybrid_blind_drift_alert_report.json`
  - `--exclude-glob hybrid_blind_drift_threshold_suggestion.json`

## 新增/更新测试
- `tests/unit/test_validate_eval_history_exclude_sidecar_reports.py`（新增）
  - 默认排除 sidecar 报告文件
  - 未排除未知 sidecar 时应失败
  - 自定义 `--exclude-glob` 后可通过
- `tests/unit/test_archive_hybrid_blind_eval_history.py`
  - 增加非哈希 commit 自动 redaction 测试
- `tests/unit/test_bootstrap_hybrid_blind_eval_history.py`
  - 增加默认 commit redaction 测试
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - 增加 workflow 历史校验步骤 `exclude-glob` 回归断言

## 验证结果
执行：

```bash
pytest -q \
  tests/unit/test_validate_eval_history_history_sequence.py \
  tests/unit/test_validate_eval_history_hybrid_blind.py \
  tests/unit/test_validate_eval_history_exclude_sidecar_reports.py \
  tests/unit/test_archive_hybrid_blind_eval_history.py \
  tests/unit/test_bootstrap_hybrid_blind_eval_history.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

结果：
- `17 passed`（本地）

额外链路验证：

```bash
python3 scripts/ci/bootstrap_hybrid_blind_eval_history.py ... --count 2
python3 scripts/validate_eval_history.py --dir <temp_eval_history> --summary
```

结果：
- 生成 2 个快照，均通过 schema 校验。

## 影响
- 解决 sidecar 报告误入 schema 校验导致的流程阻塞。
- 保证 hybrid blind 新生成历史快照的 `commit` 字段长期符合 schema。
