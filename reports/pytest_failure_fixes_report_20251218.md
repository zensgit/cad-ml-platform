# Pytest 失败用例修复报告

日期：2025-12-18

## 背景

在本机环境执行 `cad-ml-platform` 全量测试（`.venv/bin/python -m pytest -q`）时出现 3 个失败用例，失败信息集中为：

- `PermissionError: [Errno 1] Operation not permitted`

这些失败与 Dedup2D（Phase 1/2）业务逻辑无直接关系，根因是测试/脚本默认尝试写入仓库目录（`reports/`、`config/`、`data/`），但当前执行环境对仓库目录写入受限。

## 失败用例清单（修复前）

1. `tests/ocr/test_golden_eval_report.py::test_run_golden_evaluation_generates_report`
   - 子进程脚本 `tests/ocr/golden/run_golden_evaluation.py` 固定写入：
     - `reports/ocr_evaluation.md`
     - `reports/ocr_calibration.md`
   - 在仓库目录不可写时直接 `PermissionError`，导致脚本无 stdout 输出，smoke test 失败。

2. `tests/unit/test_format_matrix_exempt.py::test_matrix_exempt_project`
   - 用例写入 `config/format_validation_matrix.yaml` 以模拟 format matrix。
   - 在仓库目录不可写时 `PermissionError`。

3. `tests/unit/test_faiss_recovery_persistence_reload.py::test_faiss_recovery_persistence_reload`
   - 用例调用 `similarity._persist_recovery_state()`，默认写入 `data/faiss_recovery_state.json`。
   - 在仓库目录不可写时写入失败被吞掉（best-effort），随后 reload 读到旧值导致断言失败。

## 修复目标

- 让上述测试 **不依赖仓库目录可写**（使用 `tmp_path` / 环境变量重定向输出）。
- 保持默认行为不变：在正常可写环境下仍可写回仓库 `reports/` 等目录。

## 修复内容

### 1) Golden evaluation 脚本支持可配置输出路径 + 权限回退

文件：`tests/ocr/golden/run_golden_evaluation.py`

- 新增环境变量：
  - `OCR_GOLDEN_EVALUATION_REPORT_PATH`
  - `OCR_GOLDEN_CALIBRATION_REPORT_PATH`
- 新增 `_open_report()`：当目标路径写入失败（`PermissionError`）时，回退写入到 `tempfile.gettempdir()`，并在 stderr 输出 warning。

### 2) Golden smoke test 改用 tmp_path 并向子进程注入 env

文件：`tests/ocr/test_golden_eval_report.py`

- 使用 `tmp_path` 生成 `ocr_evaluation.md` / `ocr_calibration.md`
- `subprocess.run(..., env=...)` 将脚本输出重定向到临时目录
- 断言报告文件在 `tmp_path` 下存在

### 3) Format matrix 测试改用 tmp_path + FORMAT_VALIDATION_MATRIX

文件：`tests/unit/test_format_matrix_exempt.py`

- 将 format matrix 写入 `tmp_path/format_validation_matrix.yaml`
- 通过 `monkeypatch.setenv("FORMAT_VALIDATION_MATRIX", ...)` 指向临时路径
- 用 `monkeypatch.setenv("FORMAT_STRICT_MODE", "1")` 替代直接修改 `os.environ`

### 4) Faiss recovery state 持久化测试改为写入 tmp_path

文件：`tests/unit/test_faiss_recovery_persistence_reload.py`

- 在调用 `_persist_recovery_state()` 前，通过 `monkeypatch.setenv("FAISS_RECOVERY_STATE_PATH", tmp_path/...)` 强制写入临时目录
- 直接断言持久化文件存在，确保后续 `load_recovery_state()` 读到的是本次写入的数据

## 验证结果

### 定向回归

命令：

```bash
.venv/bin/python -m pytest -q \
  tests/ocr/test_golden_eval_report.py::test_run_golden_evaluation_generates_report \
  tests/unit/test_format_matrix_exempt.py::test_matrix_exempt_project \
  tests/unit/test_faiss_recovery_persistence_reload.py::test_faiss_recovery_persistence_reload
```

结果：✅ `3 passed`

### 全量回归

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：✅ `3530 passed, 42 skipped`（仍有少量 warning，未影响测试通过）

## 影响面评估

- 仅调整测试与测试脚本的输出路径策略，不影响生产代码路径。
- 默认输出路径仍是仓库 `reports/`（仅当通过 env 覆盖或遇到权限错误时才回退/重定向）。

