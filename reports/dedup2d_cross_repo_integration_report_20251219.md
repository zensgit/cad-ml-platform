# Dedup2D 跨仓库联调验收报告（cad-ml-platform ↔ dedupcad-vision）

日期：2025-12-19

## 目标

验证 `cad-ml-platform` 的 Dedup2D 异步 Job（Redis + ARQ Worker）能够**稳定调用真实** `dedupcad-vision` 的 `/api/v2/search`，并完成：

- 上传 → 入队 → Worker 执行 → 调用 Vision → 写回结果 → Job 查询/列表

## 结论（摘要）

- ✅ `dedupcad-vision` Docker 镜像可启动并对外提供 `/health`、`/api/v2/search`
- ✅ `cad-ml-platform` docker-compose（API/Worker/Redis/MinIO/Prometheus/Grafana）可启动并通过监控自检
- ✅ `cad-ml-platform` 异步 Dedup2D Job 能调用真实 `dedupcad-vision` 并 `pending → in_progress → completed`
- ✅ `GET /api/v1/dedup/2d/jobs` 在 Redis backend 下能列出近期已完成 job（本次联调发现缺陷后已修复并补测试）

## 本机环境与端口

为避免本机已有服务端口冲突，`cad-ml-platform` compose 使用了可配置端口（见 Phase4 分支 commit）：

- cad-ml API: `http://localhost:18000`
- Prometheus: `http://localhost:19091`
- Grafana: `http://localhost:3001`
- Redis: `localhost:16379`（仅 host 映射，容器内仍 `redis:6379`）
- MinIO(S3): `http://localhost:19000`（console: `19001`）

`dedupcad-vision` 通过单独 docker 容器暴露：

- Vision: `http://localhost:58001`（容器内部 8000）

## 关键修复（联调中发现）

### 1) dedupcad-vision Docker 镜像无法启动（已修复）

问题与修复：

- `ImportError: libGL.so.1`：`Dockerfile` runtime stage 增加 `libgl1`
- `ImportError: email-validator is not installed`：补齐 `email-validator` 依赖（`pyproject.toml` + `requirements.txt`）
- `sqlite3.OperationalError: unable to open database file`：VOLUME 导致 `/app/data` 等目录在匿名卷中权限不对；在 `Dockerfile` 中预创建 `/app/data /app/indexes /app/logs` 并 `chown`，确保匿名卷继承 owner

对应仓库变更：`dedupcad-vision` 分支 `fix/vision-progressive-deps` commit `67966c3`

### 2) cad-ml-platform Redis backend 的 jobs 列表不返回已完成 job（已修复）

现象：

- 提交 job 并完成后，`GET /api/v1/dedup/2d/jobs` 返回空列表

根因：

- Redis backend 的 `list_dedup2d_jobs_for_tenant` 仅从 `{prefix}:active` 集合读取（active set 只包含 pending/in_progress）

修复：

- 新增 per-tenant ZSET 索引：`{prefix}:tenant:{tenant_id}:jobs`（score=created_at）
- submit 时写入 ZSET；list 时从 ZSET 读取并懒清理过期 job；从而包含 TTL 内的已完成 job
- 新增单测：`tests/unit/test_dedup2d_job_list_redis.py`

对应仓库变更：`cad-ml-platform` 分支 `feat/dedup2d-phase4-production-ready` commit `3f6844a`

## 验收步骤与结果

### A) 启动 dedupcad-vision（真实服务）

构建：

```bash
cd dedupcad-vision
docker build -t dedupcad-vision:local -f Dockerfile .
```

运行：

```bash
docker rm -f dedupcad-vision-e2e || true
docker run -d --name dedupcad-vision-e2e -p 58001:8000 dedupcad-vision:local
curl -sf http://localhost:58001/health
```

验证 `/api/v2/search`：

```bash
curl -sSf -X POST "http://localhost:58001/api/v2/search" \
  -F "file=@dedupcad-vision/tests/test_data/test_drawing1.png;type=image/png" \
  -F "mode=balanced" -F "max_results=10" -F "compute_diff=false" \
  -F "enable_ml=false" -F "enable_geometric=false"
```

结果：✅ 返回结构包含 `success/duplicates/similar/final_level/timing/level_stats`

### B) 启动 cad-ml-platform（compose + MinIO + 监控）

```bash
cd cad-ml-platform
CAD_ML_API_PORT=18000 \
CAD_ML_API_METRICS_PORT=19090 \
CAD_ML_REDIS_PORT=16379 \
CAD_ML_PROMETHEUS_PORT=19091 \
CAD_ML_MINIO_PORT=19000 \
CAD_ML_MINIO_CONSOLE_PORT=19001 \
docker compose -p cad-ml-platform-phase4 \
  -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  up -d --build
```

监控验活：

- ✅ `GET http://localhost:18000/health` 返回 `healthy`
- ✅ `GET http://localhost:19091/api/v1/targets`：`cad-ml-api` target `up`
- ✅ `GET http://localhost:3001/api/search?query=Dedup2D`：可检索到 `Dedup2D Dashboard`

### C) 提交 Dedup2D 异步 Job（真实 dedupcad-vision）

说明：compose 内默认 `DEDUPCAD_VISION_URL=http://host.docker.internal:58001`，因此只要 host 端口 `58001` 有服务即可联通。

提交：

```bash
curl -sSf -X POST \
  "http://localhost:18000/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true" \
  -H "X-API-Key: tenant_abc123" \
  -F "file=@dedupcad-vision/tests/test_data/test_drawing2.png;type=image/png"
```

轮询：

```bash
curl -sSf -H "X-API-Key: tenant_abc123" \
  "http://localhost:18000/api/v1/dedup/2d/jobs/<job_id>"
```

结果：✅ job 状态 `pending → in_progress → completed`，`result` 为 `dedupcad-vision /api/v2/search` 的 JSON。

### D) 验证 jobs 列表

```bash
curl -sSf -H "X-API-Key: tenant_abc123" \
  "http://localhost:18000/api/v1/dedup/2d/jobs?limit=5"
```

结果：✅ 返回 `completed` job（TTL 内可见）

## 备注：Webhook 回调（Phase 3）

- 默认策略：仅允许 `https` 且阻止私网回调（SSRF 防护），因此本机联调一般不直接测 callback。
- 本机端到端 webhook 已由脚本覆盖：
  - `scripts/e2e_dedup2d_webhook.py`
  - `scripts/e2e_dedup2d_webhook_minio.py`

如需在本地 compose 场景强制测 callback，需要在 API/Worker 增加环境变量（仅限开发环境）：

- `DEDUP2D_CALLBACK_ALLOW_HTTP=1`
- `DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS=0`

