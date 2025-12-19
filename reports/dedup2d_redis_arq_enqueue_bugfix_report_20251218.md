# Dedup2D Redis/ARQ 入队失败修复报告

日期：2025-12-18

## 背景

在进行 Dedup2D（`DEDUP2D_ASYNC_BACKEND=redis`）端到端验证时，发现任务提交后长期停留在 `pending`，Worker 未执行，导致：

- `/api/v1/dedup/2d/jobs/{job_id}` 一直返回 `status=pending`
- Phase 3 Webhook 回调也不会触发（`callback_status` 卡在 `pending`）

## 根因定位

### 现象（Worker 日志）

ARQ Worker 直接报错并标记 job failed：

- `TypeError: dedup2d_run_job() got an unexpected keyword argument '_job_timeout'`

### 根因

`arq==0.26.3` 的 `ArqRedis.enqueue_job()` **不支持** `_job_timeout` 参数（其签名只接受：
`_job_id / _queue_name / _defer_until / _defer_by / _expires / _job_try`）。

此前在 `cad-ml-platform/src/core/dedupcad_2d_jobs_redis.py` 中入队时传入了 `_job_timeout=...`：

- 该参数被当成普通 kwargs 原样传入任务函数 `dedup2d_run_job()`，从而触发 TypeError
- 任务函数未执行到“更新 job hash 状态”的逻辑，最终表现为 API 侧长期 `pending`

## 修复内容

### 1) 移除错误的 `_job_timeout` 入队参数

- 文件：`cad-ml-platform/src/core/dedupcad_2d_jobs_redis.py`
- 变更：`enqueue_job(...)` 调用中移除 `_job_timeout=...`
- 说明：Job 运行超时应由 Worker 侧 `WorkerSettings.job_timeout` 控制（`src/core/dedupcad_2d_worker.py`），无需在 enqueue 阶段传入。

### 2) 增加回归测试防止再次引入

- 文件：`cad-ml-platform/tests/unit/test_dedup_2d_jobs_redis.py`
- 新增：`test_submit_job_does_not_pass_job_timeout_kwarg`
- 断言：`submit_dedup2d_job()` 触发的 `enqueue_job()` kwargs 中不包含 `_job_timeout`

## 验证

### 单测

- `./.venv/bin/python -m pytest -q tests/unit/test_dedup_2d_jobs_redis.py -q` ✅

### 端到端

使用新增脚本验证（见 `cad-ml-platform/scripts/e2e_dedup2d_webhook.py`）：

- Redis + API + ARQ Worker + Fake dedupcad-vision + Callback Receiver
- 任务能从 `pending -> completed`
- `callback_status` 能变为 `success`

## 影响范围评估

- 仅影响 `DEDUP2D_ASYNC_BACKEND=redis` 路径
- in-process job store 不受影响
- 修复属于“阻塞型”问题：不修复将导致 Redis backend 在真实 Worker 环境中完全不可用

