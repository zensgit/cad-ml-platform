# GitHub Actions Node24 兼容迁移（2026-03-15）

## 背景

GitHub Actions 运行日志已出现 Node20 弃用提示。为降低 2026-06-02 后的运行风险，本次统一升级常用官方 Action 版本。

## 版本依据（官方 release）

- `actions/checkout` 最新：`v6.0.2`
- `actions/setup-python` 最新：`v6.2.0`
- `setup-python v6` release 明确包含 Node24 兼容依赖升级。

## 变更范围

全量替换 `.github/workflows/*.yml` 中以下引用：

- `actions/checkout@v4` -> `actions/checkout@v6`
- `actions/setup-python@v4` -> `actions/setup-python@v6`
- `actions/setup-python@v5` -> `actions/setup-python@v6`

## 验证

### 1) workflow 单测回归

```bash
pytest -q tests/unit/test_*workflow*.py
```

结果：`187 passed`

### 2) 关键集成回归

```bash
make validate-hybrid-superpass-workflow
make validate-hybrid-blind-workflow
```

结果：

- `60 passed`
- `90 passed`

## 风险与回滚

- 风险：第三方 Action 次级依赖可能在少量 workflow 中触发行为变化（低概率）。
- 回滚：可针对单个 workflow 回退至旧版本，不影响其他 workflow。
