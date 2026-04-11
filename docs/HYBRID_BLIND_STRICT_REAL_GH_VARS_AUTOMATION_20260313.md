# Hybrid Blind Strict-Real GH Vars Automation (2026-03-13)

## 目标
- 为 strict-real 混合盲测提供一键同步 GitHub Variables 的工具，减少手工配置错误。
- 默认 `dry-run`，仅在显式开启时执行写入。

## 新增能力
1. 新脚本：
   - `scripts/ci/apply_hybrid_blind_strict_real_gh_vars.py`
2. 新 Make 目标：
   - `hybrid-blind-strict-real-apply-gh-vars`

## 变量基线
脚本会生成并按 key 排序输出以下变量：
- `HYBRID_BLIND_ENABLE=true`
- `HYBRID_BLIND_FAIL_ON_GATE_FAILED=true`
- `HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA=true`
- `HYBRID_BLIND_DXF_DIR=<--dxf-dir>`
- `HYBRID_BLIND_DRIFT_ALERT_ENABLE=true`

## 使用方式
### Dry-run（默认）
```bash
python3 scripts/ci/apply_hybrid_blind_strict_real_gh_vars.py \
  --repo zensgit/cad-ml-platform \
  --dxf-dir /mnt/cad/real_dxf
```

### 实际写入
```bash
python3 scripts/ci/apply_hybrid_blind_strict_real_gh_vars.py \
  --repo zensgit/cad-ml-platform \
  --dxf-dir /mnt/cad/real_dxf \
  --apply
```

### Make 方式
```bash
make hybrid-blind-strict-real-apply-gh-vars \
  HYBRID_BLIND_STRICT_APPLY_REPO=zensgit/cad-ml-platform \
  HYBRID_BLIND_STRICT_APPLY_DXF_DIR=/mnt/cad/real_dxf \
  HYBRID_BLIND_STRICT_APPLY_EXECUTE=1
```

## 验证
执行：
```bash
pytest -q \
  tests/unit/test_apply_hybrid_blind_strict_real_gh_vars.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

结果：
- `24 passed`

额外回归：
```bash
pytest -q \
  tests/unit/test_build_hybrid_blind_synthetic_dxf_dataset.py \
  tests/unit/test_hybrid_blind_gate_check.py \
  tests/unit/test_hybrid_confidence_calibration_gate_check.py \
  tests/unit/test_hybrid_confidence_calibration_baseline_update.py \
  tests/unit/test_apply_hybrid_blind_strict_real_gh_vars.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

结果：
- `39 passed`
