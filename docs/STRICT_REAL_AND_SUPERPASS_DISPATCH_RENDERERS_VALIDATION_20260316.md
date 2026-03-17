# Strict-Real 与 Superpass Dispatch Renderers 验证（2026-03-16）

## 目标

为两条现有 GH dispatch 链路补齐统一的本地摘要渲染能力：

- `hybrid blind strict-real dispatch`
- `hybrid superpass dispatch`

使真实运行后的 `--output-json` 能直接转成 Markdown，便于人工审阅、artifact 化与后续 workflow 接线。
本轮还补齐了统一的 `Dispatch Verdict` 段，使 strict-real 与 superpass 摘要都先给出结论、再展开 failure diagnostics。
本轮继续补了统一的 `Dispatch Snapshot` 段，并为无效 JSON 输入补齐失败回归。

## 实现

### 1) 新增 strict-real dispatch renderer

- 文件：`scripts/ci/render_hybrid_blind_strict_real_dispatch_summary.py`
- 输入：
  - `--dispatch-json`
  - `--output-md`
- 输出字段：
  - workflow / ref / repo
  - strict-real dxf/manfiest/synth/strict flags
  - expected / actual conclusion
  - dispatch verdict / top failed jobs / top failed steps / diagnostics reason
  - dispatch snapshot / failure reason
  - overall/dispatch/watch exit code
  - run_id / run_url
  - failure diagnostics（若存在）

### 2) 新增 superpass dispatch renderer

- 文件：`scripts/ci/render_hybrid_superpass_dispatch_summary.py`
- 输入：
  - `--dispatch-json`
  - `--output-md`
- 输出字段：
  - workflow / ref / repo
  - expected / actual conclusion
  - dispatch verdict / top failed jobs / top failed steps / diagnostics reason
  - dispatch snapshot / failure reason
  - overall/watch exit code
  - run_id / run_url
  - dispatch_command
  - failure diagnostics（若存在）

### 3) Make 接线

- 文件：`Makefile`
- 新增变量：
  - `HYBRID_BLIND_STRICT_E2E_OUTPUT_MD`
  - `HYBRID_SUPERPASS_E2E_OUTPUT_MD`
- 新增目标：
  - `render-hybrid-blind-strict-real-dispatch-summary`
  - `validate-render-hybrid-blind-strict-real-dispatch-summary`
  - `render-hybrid-superpass-dispatch-summary`
  - `validate-render-hybrid-superpass-dispatch-summary`

### 4) 回归测试

- 新增：
  - `tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py`
  - `tests/unit/test_render_hybrid_superpass_dispatch_summary.py`
- 更新：
  - `tests/unit/test_hybrid_calibration_make_targets.py`

## 验证

### 单测

```bash
pytest -q \
  tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py \
  tests/unit/test_render_hybrid_superpass_dispatch_summary.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

结果：`47 passed`

### Make 验证

```bash
make validate-render-hybrid-blind-strict-real-dispatch-summary
make validate-render-hybrid-superpass-dispatch-summary
```

结果：

- `validate-render-hybrid-blind-strict-real-dispatch-summary` -> `43 passed`
- `validate-render-hybrid-superpass-dispatch-summary` -> `43 passed`

### 近真实验证 1：strict-real dispatch

```bash
python3 scripts/ci/render_hybrid_blind_strict_real_dispatch_summary.py \
  --dispatch-json reports/experiments/strict_real_dispatch_20260313_fix2.json \
  --output-md reports/experiments/20260316/strict_real_dispatch_rendered_20260316.md
```

输出文件：

- `reports/experiments/20260316/strict_real_dispatch_rendered_20260316.md`

### 近真实验证 2：superpass dispatch

```bash
python3 scripts/ci/render_hybrid_superpass_dispatch_summary.py \
  --dispatch-json reports/experiments/20260315/hybrid_superpass_e2e.json \
  --output-md reports/experiments/20260316/hybrid_superpass_dispatch_rendered_20260316.md
```

输出文件：

- `reports/experiments/20260316/hybrid_superpass_dispatch_rendered_20260316.md`

## 变更文件

- `scripts/ci/render_hybrid_blind_strict_real_dispatch_summary.py`
- `scripts/ci/render_hybrid_superpass_dispatch_summary.py`
- `Makefile`
- `tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py`
- `tests/unit/test_render_hybrid_superpass_dispatch_summary.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `docs/STRICT_REAL_AND_SUPERPASS_DISPATCH_RENDERERS_VALIDATION_20260316.md`
