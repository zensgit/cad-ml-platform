# Dedup2D Phase 3：Webhook 回调实现报告

日期：2025-12-18

## 目标

为 `cad-ml-platform` 的 2D 查重异步任务（Redis+ARQ backend）提供“结果回传”能力：

- 客户端提交 `async=true` 任务时可选传入 `callback_url`
- Worker 完成任务后向该 URL POST 结果（best-effort）
- 支持基础 SSRF 防护与可选 HMAC 签名
- 轮询接口 `/api/v1/dedup/2d/jobs/{job_id}` 增加回调状态字段，便于排障

## 代码变更

### 新增

- `src/core/dedup2d_webhook.py`
  - `validate_dedup2d_callback_url()`：URL 校验（scheme/userinfo/allowlist/private network）
  - `sign_dedup2d_webhook()`：HMAC-SHA256 签名头
  - `send_dedup2d_webhook()`：异步发送 + 重试/退避

### 修改

- `src/core/dedupcad_2d_jobs_redis.py`
  - `submit_dedup2d_job(..., callback_url=...)`：校验并持久化 callback 信息
  - `get_dedup2d_job_for_tenant()`：回传 callback 状态到 `job.meta`
  - `cancel_dedup2d_job_for_tenant()`：若 pending 即取消，标记 `callback_status=skipped`

- `src/core/dedupcad_2d_worker.py`
  - Job 完成/失败/取消后尝试发送 callback，并将结果写回 job hash：
    - `callback_status`、`callback_attempts`、`callback_http_status`、`callback_last_error`

- `src/api/v1/dedup.py`
  - `POST /api/v1/dedup/2d/search` 新增 query 参数 `callback_url`
  - 当 `DEDUP2D_ASYNC_BACKEND != redis` 时，传 callback_url 返回 400（避免静默忽略）
  - `GET /api/v1/dedup/2d/jobs/{job_id}` 响应新增 callback 状态字段

- `.env.example`
  - 新增 Phase 3 webhook 配置项（见下）

## API 行为

### 1) 提交异步任务（带 callback）

- Endpoint：`POST /api/v1/dedup/2d/search?async=true&callback_url=...`
- 限制：仅支持 `DEDUP2D_ASYNC_BACKEND=redis`

若 backend 非 redis：

```json
{
  "detail": {
    "error": "CALLBACK_UNSUPPORTED",
    "message": "callback_url requires DEDUP2D_ASYNC_BACKEND=redis"
  }
}
```

### 2) Webhook payload（POST JSON）

Worker 会 POST 如下 JSON（字段对齐 job 生命周期；result 仅在 completed 时存在）：

```json
{
  "job_id": "…",
  "tenant_id": "…",
  "status": "completed|failed|canceled",
  "started_at": 1734567890.1,
  "finished_at": 1734567892.3,
  "result": { "...": "Dedup2DSearchResponse" },
  "error": "…"
}
```

### 3) Webhook headers（可选签名）

默认会带：

- `Content-Type: application/json`
- `User-Agent: cad-ml-platform/dedup2d-webhook`
- `X-Dedup-Job-Id: <job_id>`
- `X-Dedup-Tenant-Id: <tenant_id>`（如果有）

若配置 `DEDUP2D_CALLBACK_HMAC_SECRET`，会额外带：

- `X-Dedup-Signature: t=<unix_ts>,v1=<hmac_sha256_hex>`
- `X-Dedup-Signature-Version: v1`

签名串：`"{ts}.{job_id}." + body_bytes`

## 配置项

见 `.env.example`：

- `DEDUP2D_CALLBACK_ALLOW_HTTP`：默认 0（只允许 https；开发可开 http）
- `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS`：默认 1（阻断私网/loopback/link-local IP）
- `DEDUP2D_CALLBACK_RESOLVE_DNS`：默认 0（若开启，会做 DNS 解析并阻断解析到私网 IP 的域名）
- `DEDUP2D_CALLBACK_ALLOWLIST`：可选 hostname 白名单（逗号分隔）
- `DEDUP2D_CALLBACK_HMAC_SECRET`：可选 HMAC secret
- `DEDUP2D_CALLBACK_TIMEOUT_SECONDS`：单次回调超时
- `DEDUP2D_CALLBACK_MAX_ATTEMPTS`：最大重试次数
- `DEDUP2D_CALLBACK_BACKOFF_BASE_SECONDS` / `DEDUP2D_CALLBACK_BACKOFF_MAX_SECONDS`：退避策略

## 可观测性（轮询接口返回）

`GET /api/v1/dedup/2d/jobs/{job_id}` 增加：

- `callback_status`: `pending|success|failed|skipped|null`
- `callback_attempts`
- `callback_http_status`
- `callback_finished_at`
- `callback_last_error`

## 重要语义说明

- 回调是 best-effort：回调失败不会影响 job 的最终状态（仍可通过 poll 取结果）
- 若任务在 **pending 且未开始** 即被取消，当前实现会标记 `callback_status=skipped`（避免 “pending forever” 的误导）

## 测试与回归

- `make lint` ✅
- `.venv/bin/python -m pytest -q` ✅（`3535 passed, 42 skipped`）

