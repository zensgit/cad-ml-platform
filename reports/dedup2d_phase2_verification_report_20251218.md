# Dedup2D Phase 2 校验报告（cad-ml-platform ↔ dedupcad-vision）

日期：2025-12-18

## 结论摘要

- 本次变更的核心目标（Phase 2：`cad-ml-platform` 采用 Redis+ARQ 异步 Job，并在“API/Worker 不共享磁盘”的前提下可正确读取 v2 几何 JSON）已通过 **lint + 定向测试** 验证。
- `dedupcad-vision` 本次未做代码修改；已对接口契约进行静态复核（`cad-ml-platform` 调用 `/api/v2/search` 与 `dedupcad-vision` 实现一致）。
- 全量测试已通过（`3530 passed, 42 skipped`）；原先因执行环境对仓库目录写入受限导致的 3 个失败用例，已通过“脚本输出路径可配置 + 测试改用 tmp_path”修复。

## 代码范围

### cad-ml-platform

本次校验覆盖的关键模块/能力：

- **2D 查重（异步）Redis backend + ARQ worker**
  - Job 提交、查询、取消、容量上限（`JOB_QUEUE_FULL`）、完成态结果读取
  - Active set 清理（防止 worker 异常导致队列容量永久占用）
- **v2 几何 JSON 存储后端抽象（filesystem / redis / hybrid）**
  - 解决 “API Server 与 Worker 不共享磁盘” 的生产不确定性
- **dedupcad-vision 调用超时上调**
  - 默认 60s（降低几何计算/精查导致的误判超时概率）

### dedupcad-vision

- 未做代码变更；仅静态核对接口：
  - v2 查重入口：`/api/v2/search`（`dedupcad-vision/src/caddedup_vision/api/routes_progressive.py`）

## Git 状态（校验时）

### cad-ml-platform

- Branch：`feat/dedup2d-phase1-tenant-isolation`
- HEAD：`d51ac9204064edba2b6354081ea7ea8b9431e711`
- Working tree：有未提交改动（Phase 2 代码 + 本次补强）
  - Modified：
    - `.env.example`
    - `deployments/docker/docker-compose.yml`
    - `src/api/v1/dedup.py`
    - `src/core/dedupcad_precision/__init__.py`
    - `src/core/dedupcad_precision/store.py`
    - `src/core/dedupcad_vision.py`
    - `tests/test_dedup_2d_proxy.py`
    - （以及 Phase 1 的 `src/core/dedupcad_2d_jobs.py`、`src/main.py` 等）
  - Untracked：
    - `src/core/dedupcad_2d_jobs_redis.py`
    - `src/core/dedupcad_2d_pipeline.py`
    - `src/core/dedupcad_2d_worker.py`
    - `tests/unit/test_dedup_geom_store.py`
    - `tests/unit/test_dedup_2d_jobs_redis.py`
    - `handoff/`（保留，不在本次校验中改动）

### dedupcad-vision

- Branch：`fix/vision-progressive-deps`
- HEAD：`0b6cce2d58ec84d4453b8b14af1ec9cd984332e0`
- Working tree：干净（无改动）

## 校验环境

- OS：`Darwin Kernel Version 25.1.0`（arm64）
- Python（venv）：`Python 3.11.13`（`cad-ml-platform/.venv/bin/python`）
- 说明：当前执行环境对仓库目录的写入存在 `EPERM` 限制，导致少量测试（需要写入 `reports/`、`config/`、`data/`）失败。

## 关键变更点（可追溯）

### 1) Geom store：支持 Redis/Hybrid（解决“不共享磁盘”）

- 代码：
  - `src/core/dedupcad_precision/store.py`
    - `RedisGeomJsonStore`：key 设计 `{DEDUPCAD_GEOM_STORE_REDIS_KEY_PREFIX}:geom:{file_hash}`
    - `HybridGeomJsonStore`：优先 filesystem，miss 再读 redis
    - `create_geom_store()`：基于 `DEDUPCAD_GEOM_STORE_BACKEND` 选择后端
  - `src/core/dedupcad_precision/__init__.py`：导出 `create_geom_store` 等
- 注入点：
  - API：`src/api/v1/dedup.py:get_geom_store()` → `create_geom_store()`
  - Worker：`src/core/dedupcad_2d_worker.py` → `create_geom_store()`
- 配置：
  - `.env.example` 增加：
    - `DEDUPCAD_GEOM_STORE_BACKEND=filesystem|redis|hybrid`
    - `DEDUPCAD_GEOM_STORE_REDIS_URL`
    - `DEDUPCAD_GEOM_STORE_REDIS_KEY_PREFIX`
    - `DEDUPCAD_GEOM_STORE_REDIS_TTL_SECONDS`

### 2) Redis job backend：取消语义 & active set 清理

- 代码：`src/core/dedupcad_2d_jobs_redis.py`
  - `_prune_active_set()`：清理 `active` set 中已完成/丢失的 job（防止“永久满队列”）
  - `cancel_dedup2d_job_for_tenant()`：对 `pending` 且未开始的 job，直接置 `canceled` + 从 active set 移除（避免 abort 后 job 永久 pending）

### 3) 调用 dedupcad-vision：默认超时上调为 60s

- `src/core/dedupcad_vision.py`：默认 `timeout_seconds=60.0`
- `.env.example` / `deployments/docker/docker-compose.yml` 同步为 60s

## 执行的校验命令与结果

### 1) Lint

命令：

```bash
make lint
```

结果：✅ 通过（flake8 src）

### 2) Dedup2D 定向测试（推荐的最小回归集合）

命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_dedup_2d_proxy.py \
  tests/unit/test_dedup_geom_store.py \
  tests/unit/test_dedup_2d_jobs_redis.py
```

结果：✅ 通过（`30 passed`）

### 3) 全量测试（供参考）

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：✅ `3530 passed, 42 skipped, 9 warnings in 46.05s`

注（修复前的失败原因，均与工作区写入限制相关）：

1. `tests/ocr/test_golden_eval_report.py::test_run_golden_evaluation_generates_report`
   - 子进程脚本 `tests/ocr/golden/run_golden_evaluation.py` 尝试写入 `reports/ocr_evaluation.md`
   - 触发：`PermissionError: [Errno 1] Operation not permitted`
2. `tests/unit/test_faiss_recovery_persistence_reload.py::test_faiss_recovery_persistence_reload`
   - `similarity._persist_recovery_state()` 需要写入 `data/faiss_recovery_state.json`，写入受限导致持久化未生效，进而断言失败（读到旧值）
3. `tests/unit/test_format_matrix_exempt.py::test_matrix_exempt_project`
   - 用例需要写入 `config/format_validation_matrix.yaml`，写入受限导致直接 `PermissionError`

#### 修复内容（已验证）

为避免测试依赖仓库目录可写（`reports/`、`config/`、`data/`），已做如下修复：

- `tests/ocr/golden/run_golden_evaluation.py` 支持环境变量覆盖输出路径：
  - `OCR_GOLDEN_EVALUATION_REPORT_PATH`
  - `OCR_GOLDEN_CALIBRATION_REPORT_PATH`
- `tests/ocr/test_golden_eval_report.py` 通过 `subprocess.run(..., env=...)` 将 report 输出重定向到 `tmp_path`
- `tests/unit/test_format_matrix_exempt.py` 改为写入 `tmp_path` 并设置 `FORMAT_VALIDATION_MATRIX`
- `tests/unit/test_faiss_recovery_persistence_reload.py` 改为在持久化前设置 `FAISS_RECOVERY_STATE_PATH` 指向 `tmp_path`

## 手工验收建议（可选）

建议用 docker-compose 启动 `cad-ml-platform + redis + dedup2d-worker`，并单独启动 `dedupcad-vision`：

```bash
docker-compose -f deployments/docker/docker-compose.yml up -d
```

关键验收点：

- `DEDUP2D_ASYNC_BACKEND=redis` 时：
  - `POST /api/v1/dedup/2d/search?async=true` 返回 `job_id`
  - `GET /api/v1/dedup/2d/jobs/{job_id}` 最终返回 `completed/failed/canceled` + `result`
- 若 `DEDUPCAD_GEOM_STORE_BACKEND=redis|hybrid`：
  - Worker 侧能读取到 `geom:{file_hash}` 并完成 L4 精查（`precision_*` 字段存在）

## 风险与后续建议

- 当前 Phase 2 job payload 仍会把 `file_bytes` base64 存 Redis：适合小文件/开发；大文件建议下一步改为 **FileRef（S3/MinIO 或共享卷）**。
- Phase 3（Webhook 回调）可按业务需要再引入；在“无回调”的情况下，轮询 + Redis 结果已经可满足大多数场景。
