# Dedup2D ↔ DedupCAD-Vision 接口契约（对齐版）

目标：明确 `cad-ml-platform`（编排/异步/存储/回调）与 `dedupcad-vision`（2D 搜索引擎）之间的请求/响应、超时与错误语义，避免“能调用但结果/回传丢失”。

## 角色与调用方向

### 推荐生产拓扑（单向依赖）

- `cad-ml-platform` → `dedupcad-vision`
  - `cad-ml-platform` 接收上传、入队、持久化、回调
  - Worker 调用 `dedupcad-vision /api/v2/search`

> 说明：`dedupcad-vision` 内置 `MLPlatformClient` 默认禁用（见 `ML_PLATFORM_ENABLED=false`），建议不要形成循环依赖。

## 1) dedupcad-vision（被调方）

### 1.1 Health

- `GET /health`
- `200` JSON：必须包含 `status` 字段，`cad-ml-platform` 用于健康检查/告警。

### 1.2 2D Search（核心）

- `POST /api/v2/search`
- Content-Type: `multipart/form-data`
- Form 字段：
  - `file`：图纸图像文件（推荐 `png/jpg`）
  - `mode`：`fast|balanced|precise`（字符串）
  - `max_results`：整数
  - `compute_diff`：`true|false`
  - `enable_ml`：`true|false`（可选，需 vision 端已配置 L3）
  - `enable_geometric`：`true|false`（可选，需 vision 端已配置 L4）

#### 响应（JSON）

必须包含以下字段（缺失会导致 `cad-ml-platform` 解析/透传不稳定）：

- `success: bool`
- `total_matches: int`
- `duplicates: list[MatchItem]`
- `similar: list[MatchItem]`
- `final_level: int`
- `timing: {total_ms,l1_ms,l2_ms,l3_ms,l4_ms}`
- `level_stats: object`
- `warnings: list[str]`
- `error: str|null`

`MatchItem`（最小契约）：

- `drawing_id: str`
- `file_hash: str`
- `file_name: str`
- `similarity: float`
- `confidence: float`
- `match_level: int`
- `verdict: str`
- `levels: object`
- 可选：`diff_image_base64`, `diff_regions`

## 2) cad-ml-platform（对外 API）

### 2.1 提交查重

- `POST /api/v1/dedup/2d/search`
- Query：
  - `async=true|false`
  - `mode=fast|balanced|precise`
  - `max_results=<int>`
  - `callback_url=<url>`（可选）
- Form：
  - `file`（图像文件）
  - `geom_json`（可选，用于 precision；启用时通常强制 async）

#### 同步响应（async=false 且未触发 forced-async）

直接返回 `dedupcad-vision /api/v2/search` 的 JSON 结果（透传）。

#### 异步响应（async=true 或 forced-async）

```json
{
  "job_id": "<uuid>",
  "status": "pending",
  "poll_url": "/api/v1/dedup/2d/jobs/<job_id>",
  "forced_async_reason": null
}
```

### 2.2 查询 Job

- `GET /api/v1/dedup/2d/jobs/{job_id}`
- `200`：
  - `status`：`pending|in_progress|completed|failed|canceled`
  - `result`：完成时为 `dedupcad-vision` JSON（同上）
  - `error`：失败时给出字符串

### 2.3 取消 Job

- `POST /api/v1/dedup/2d/jobs/{job_id}/cancel`
- 权限：同租户可取消；跨租户返回 `403 JOB_FORBIDDEN`

### 2.4 列表

- `GET /api/v1/dedup/2d/jobs?status=<opt>&limit=<opt>`
- 返回 TTL 内的近期 jobs（包含已完成），按 `created_at` 倒序

## 3) Webhook 回调（可选）

当提交时携带 `callback_url`，worker 在 job 完成后 best-effort POST：

- Headers：
  - `Content-Type: application/json`
  - `X-Dedup-Job-Id: <job_id>`
  - `X-Dedup-Tenant-Id: <tenant_id>`（如可用）
  - `X-Dedup-Signature: t=<ts>,v1=<hex>`（当配置 `DEDUP2D_CALLBACK_HMAC_SECRET` 时）

- Body：`{"job_id": "...", "tenant_id": "...", "status": "completed", "result": {...}}`（以实现为准）

安全默认：

- 仅允许 `https`
- 默认阻断私网/loopback（SSRF 防护）

开发环境需要本地回调时（仅 dev）：

- `DEDUP2D_CALLBACK_ALLOW_HTTP=1`
- `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=0`

## 4) 超时与重试建议（默认可调整）

- `DEDUPCAD_VISION_TIMEOUT_SECONDS`: 60（单次调用）
- Job 最大运行：`DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS`: 300
- 回调：`DEDUP2D_CALLBACK_TIMEOUT_SECONDS`: 10，`DEDUP2D_CALLBACK_MAX_ATTEMPTS`: 3

## 5) 重要提示：dedupcad-vision → cad-ml-platform 的 ML 调用

`dedupcad-vision` 的 `MLPlatformClient` 默认通过 `/api/v1/analyze` 上传文件获取特征/分类（见 `src/caddedup_vision/integrations/ml_platform.py`）。

注意：`cad-ml-platform /api/v1/analyze` **不支持 PNG/JPG**（会返回 `UNSUPPORTED_FORMAT`），因此如果 `dedupcad-vision` 的 L3 输入是图像文件，则该方向调用无法直接启用。

推荐做法：

- 生产保持 `ML_PLATFORM_ENABLED=false`（避免循环依赖 + 避免格式不兼容）
- 若必须启用：需要新增/对齐“图像特征/语义分析”端点或让 vision 侧以 CAD 原文件作为 L3 输入（而非 PNG）

补充：`/api/v1/analyze` 的 `results.features` 现在包含 `combined` 字段（与 `flatten()` 同序），供 dedupcad-vision 融合语义特征使用。

补充：`/api/v1/vectors/register` 与 `/api/v1/vectors/search` 已提供用于 dedupcad-vision 的向量注册与相似检索。
