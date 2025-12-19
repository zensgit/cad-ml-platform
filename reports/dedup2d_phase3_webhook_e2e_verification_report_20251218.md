# Dedup2D Phase 3 Webhook 端到端验证报告

日期：2025-12-18

## 验证目标

验证 `cad-ml-platform` 在 Redis + ARQ backend 下，Dedup2D 异步任务的“结果回传”链路可用：

1. API 接收 `async=true` + `callback_url`
2. Redis job/payload 写入成功，ARQ 入队成功
3. ARQ Worker 能消费任务并调用 dedupcad-vision
4. Worker 写回 job 结果（`completed`）并发送 webhook 回调
5. 回调方收到 payload，且 HMAC 签名可验证

## 验证方式

使用新增 E2E 脚本：

- 脚本：`cad-ml-platform/scripts/e2e_dedup2d_webhook.py`
- 启动组件（均为本机临时端口）：
  - `redis-server`
  - Fake `dedupcad-vision`（FastAPI + uvicorn）
  - Callback Receiver（FastAPI + uvicorn，写入 `callback_received.json`）
  - Minimal API Server（FastAPI + uvicorn，仅挂载 `/api/v1/dedup` router）
  - ARQ Worker（`arq src.core.dedupcad_2d_worker.WorkerSettings`）

## 执行命令

在 `cad-ml-platform` 目录下：

```bash
./.venv/bin/python scripts/e2e_dedup2d_webhook.py --keep-dir --startup-timeout 90 --job-timeout 60
```

## 关键配置（脚本内固定）

- `DEDUP2D_ASYNC_BACKEND=redis`
- `DEDUP2D_REDIS_KEY_PREFIX=dedup2d_e2e_<ts>`（隔离命名空间）
- `DEDUP2D_ARQ_QUEUE_NAME=<prefix>:queue`（确保 API 与 Worker 一致）
- 为本地回调允许 HTTP + 私网（仅测试用）：
  - `DEDUP2D_CALLBACK_ALLOW_HTTP=1`
  - `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=0`
- HMAC：
  - `DEDUP2D_CALLBACK_HMAC_SECRET=e2e_secret`

## 结果

### 控制台输出（节选）

```
tmp_dir: /var/.../dedup2d_webhook_e2e_sqnnw1m8
redis: redis://127.0.0.1:57843/0
api: http://127.0.0.1:57846
vision: http://127.0.0.1:57844
callback: http://127.0.0.1:57845/hook
key_prefix: dedup2d_e2e_1766068651
queue_name: dedup2d_e2e_1766068651:queue
OK: job completed + callback received + signature verified
```

### 验收结论

- ✅ Job 状态从 `pending` 变为 `completed`
- ✅ `/api/v1/dedup/2d/jobs/{job_id}` 返回 `callback_status=success`
- ✅ Callback Receiver 收到 payload 并写入 `callback_received.json`
- ✅ `X-Dedup-Signature` 可使用 `e2e_secret` 完整校验通过

## 备注

- 本 E2E 仅用于本机验证链路，故放开了 HTTP 与私网回调限制；生产环境建议保持默认（`https` + 阻断私网）。
- 若需要排障，可查看脚本输出的 `tmp_dir` 下的 `logs/*.log`。

## 回归测试（补充）

- `./.venv/bin/python -m pytest -q` ✅（`3537 passed, 42 skipped`）
