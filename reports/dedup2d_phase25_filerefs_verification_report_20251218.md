# Dedup2D Phase 2.5：FileRef（上传文件外置）实现与验证报告

日期：2025-12-18

## 背景与目标

当前 Dedup2D Redis/ARQ 异步模式下，Job payload 之前会把上传文件以 `file_bytes_b64` 的形式内嵌到 Redis：

- 对大文件不友好（Redis 内存/网络/序列化开销）
- 不利于水平扩容（高并发时 Redis 压力会被放大）
- 与“API/Worker 不共享磁盘”的生产不确定性冲突

本阶段目标：

1. 将 Job payload 从“内嵌文件内容”升级为“只存引用（FileRef）”
2. 支持两种存储后端：
   - `local`：本地/共享卷（开发/单机/同卷）
   - `s3`：S3/MinIO（生产推荐；API/Worker 不需要共享磁盘）
3. Worker 在任务完成/失败/取消后（可配置）清理上传文件，避免磁盘堆积

## 代码变更摘要

### 新增

- `src/core/dedup2d_file_storage.py`
  - `Dedup2DFileRef`：file 引用结构（`backend=local|s3`）
  - `LocalDedup2DFileStorage`：写入/读取/删除本地文件
  - `S3Dedup2DFileStorage`：写入/读取/删除 S3 对象（需要 `boto3`）
  - `create_dedup2d_file_storage()`：按环境变量创建后端

### 修改

- `src/core/dedupcad_2d_jobs_redis.py`
  - `submit_dedup2d_job()`：
    - 写入外部存储，生成 `file_ref`
    - Redis payload 改为存 `file_ref`（不再存 `file_bytes_b64`）
    - enqueue 失败时 best-effort 删除已写入的文件
  - `cancel_dedup2d_job_for_tenant()`：
    - 若 pending 且未开始即取消：best-effort 删除 payload 中引用的文件（避免泄漏）

- `src/core/dedupcad_2d_worker.py`
  - 从 payload 读取 `file_ref`，通过 file storage 加载 bytes
  - 任务结束后按 `DEDUP2D_FILE_STORAGE_CLEANUP_ON_FINISH` 清理上传文件

- `.env.example` / `deployments/docker/docker-compose.yml`
  - 增加并示例化 Phase 2.5 新增环境变量（见下）

## 新增环境变量

`.env.example` 已补齐示例：

- `DEDUP2D_FILE_STORAGE=local|s3`
- `DEDUP2D_FILE_STORAGE_DIR=data/dedup2d_uploads`
- `DEDUP2D_FILE_STORAGE_CLEANUP_ON_FINISH=1`

S3/MinIO（backend=s3）：

- `DEDUP2D_S3_BUCKET=...`
- `DEDUP2D_S3_PREFIX=dedup2d/uploads`
- `DEDUP2D_S3_ENDPOINT=...`（MinIO 常用）
- `DEDUP2D_S3_REGION=...`（可选）

注：`backend=s3` 需要安装 `boto3`（当前仓库依赖中为可选依赖）。

## 端到端验证

复用并增强了已有 E2E 脚本（同时覆盖 Phase 3 webhook）：

- 脚本：`scripts/e2e_dedup2d_webhook.py`
- 组件：Redis + API + ARQ Worker + fake vision + callback receiver
- 额外校验：上传文件写入外置存储后，在 job 完成时能够被 worker 清理（脚本会检查 uploads 目录没有残留文件）

执行命令：

```bash
./.venv/bin/python scripts/e2e_dedup2d_webhook.py --keep-dir --startup-timeout 90 --job-timeout 60
```

预期输出包含：

- `OK: job completed + callback received + signature verified`
- 且不会抛出 `unexpected leftover upload files`（表示清理成功）

## 回归测试

- `./.venv/bin/python -m flake8 src` ✅
- `./.venv/bin/python -m pytest -q` ✅（`3537 passed, 42 skipped`）

## 结论

- ✅ Redis job payload 已不再内嵌文件内容，改为 `file_ref`
- ✅ 支持 `local`/`s3` 两种 file storage（S3 需要 boto3）
- ✅ Worker 可按配置自动清理上传文件，E2E 已验证无残留文件
- ✅ 全量测试通过，改动可安全进入下一阶段（Phase 4：worker/运维部署增强）

