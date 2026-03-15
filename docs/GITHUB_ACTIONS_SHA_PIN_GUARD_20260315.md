# GitHub Actions SHA Pin + Guard 增强（2026-03-15）

## 目标

在现有 Node24 升级基础上，进一步提升 CI 供应链安全与稳定性：

- 将关键官方 Action 从 tag 引用升级为 commit SHA 固定；
- 增加自动守卫，阻止回退到 `@v*` 等非 SHA 引用；
- 提供可本地执行的 Make 验证入口和单测回归。

## 实现内容

### 1) 全量 workflow 改为 SHA pin

对 `.github/workflows/*.yml` 中目标 Action 进行统一替换：

- `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd`（tag: `v6.0.2`）
- `actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405`（tag: `v6.2.0`）
- `actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f`（tag: `v7.0.0`）
- `actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c`（tag: `v8.0.1`）

并已将其余外部 action（docker/azure/github-script/cache/codeql 等）统一替换为
commit SHA（按当前 workflow 实际引用解析）。

### 2) 新增 pin 校验脚本

- 新文件：`scripts/ci/check_workflow_action_pins.py`
- 关键能力：
  - 扫描 workflow 中 `uses:` 行；
  - 通过策略文件 `config/workflow_action_pin_policy.json` 校验外部 action；
  - 支持 `--require-policy-for-all-external`，确保所有外部 action 都必须入策略；
  - 拒绝 `@v*`、动态引用、非 SHA 引用、未允许 SHA；
  - 输出结构化 JSON（可写 `--output-json`）。

### 3) 新增独立 CI 守卫 workflow

- 新文件：`.github/workflows/action-pin-guard.yml`
- 在 `push/pull_request` 针对 workflow/script/Make 变更自动执行；
- 使用 SHA pin 的 checkout/setup-python；
- 执行 `python scripts/ci/check_workflow_action_pins.py`。

### 4) 新增测试与 Make 入口

- 新增单测：
  - `tests/unit/test_check_workflow_action_pins.py`
  - `tests/unit/test_action_pin_guard_workflow.py`
- 新增策略文件：
  - `config/workflow_action_pin_policy.json`
- 更新：
  - `tests/unit/test_graph2d_parallel_make_targets.py`
  - `Makefile` 新增目标：`validate-workflow-action-pins`

## 验证结果

### 定向回归

```bash
pytest -q \
  tests/unit/test_check_workflow_action_pins.py \
  tests/unit/test_action_pin_guard_workflow.py \
  tests/unit/test_graph2d_parallel_make_targets.py
```

结果：`14 passed`

### 脚本与 Make 验证

```bash
.venv/bin/python scripts/ci/check_workflow_action_pins.py --workflows-dir .github/workflows
make validate-workflow-action-pins
```

结果：

- 脚本返回 `status: ok`, `violations_count: 0`
- `make validate-workflow-action-pins` 通过（脚本 + 单测）

### 现有关键回归未受影响

```bash
pytest -q tests/unit/test_*workflow*.py
make validate-hybrid-superpass-workflow
make validate-hybrid-blind-workflow
```

结果：

- `191 passed`（workflow 相关）
- `60 passed`
- `90 passed`

## 风险与回滚

- 风险：后续 Action 升级需要同步更新允许 SHA（策略设计内行为）。
- 回滚路径：
  - 单文件回滚 workflow 的 pin；
  - 或调整 `check_workflow_action_pins.py` 默认允许 SHA。
