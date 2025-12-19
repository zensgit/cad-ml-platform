# Dedup2D Phase 4（7 天）验收 / 校验报告

日期：2025-12-19

## 校验范围

对 Claude 宣称完成的 Phase 4（Day1–Day7）结果做本机可重复验收，覆盖：

- Rolling upgrade 兼容（worker 兼容旧 payload）
- S3/MinIO 文件存储链路（FileRef + boto3）
- 文件保留与 GC 脚本
- Helm/K8s 模板可渲染
- 可观测性（metrics/alerts/dashboard）与 metrics contract 测试
- API 易用性（forced-async + GET /2d/jobs）
- 端到端链路（Redis + ARQ + callback webhook）

## 结论（摘要）

- ✅ 新增单测、关键集成测试、全量回归全部通过
- ✅ 本地文件存储 E2E 通过（job 完成 + webhook 回调 + HMAC 校验 + 上传文件清理）
- ✅ MinIO(S3) 文件存储 E2E 通过（同上 + S3 对象清理）
- ✅ Helm chart 在 3 种配置下均可 `helm template` 成功渲染
- ✅ 发现 1 处“forced-async 导致旧 proxy 测试期望同步返回”引起的测试失败，已修复（见下文）

## 仓库状态（验收时）

仓库：`cad-ml-platform`

- 分支：`feat/dedup2d-phase1-tenant-isolation`
- HEAD：`d51ac92`
- Phase 4 相关变更均在工作区（未提交），建议在合并/交付前统一 commit（便于追溯与 CI 复现）。

## 关键验收项对照

| Day | 目标 | 验收方式 | 结果 |
|-----|------|----------|------|
| 1 | Worker 兼容旧 payload（`file_bytes_b64` fallback） | `tests/unit/test_dedup_2d_jobs_redis.py` + `tests/unit/test_dedupcad_2d_worker` 相关覆盖 | ✅ |
| 2 | S3/MinIO 存储（FileRef） | `tests/unit/test_dedup2d_file_storage_s3.py` + MinIO E2E | ✅ |
| 3 | 上传文件保留/GC | `tests/unit/test_dedup2d_gc.py` + `scripts/dedup2d_uploads_gc.py` | ✅ |
| 4 | Helm/K8s 部署模板 | `helm template`（默认 / local / s3+secret） | ✅ |
| 5 | 可观测性（metrics/alerts/dashboard） | `tests/test_metrics_contract.py` | ✅ |
| 6 | API 易用性（forced-async + GET /2d/jobs） | `tests/unit/test_dedup2d_api_usability.py` + proxy 集成覆盖 | ✅ |
| 7 | E2E + 周报 | `scripts/e2e_dedup2d_webhook.py` + `scripts/e2e_dedup2d_webhook_minio.py` + `claudedocs/phase4_dedup2d_summary.md` | ✅ |

## 执行的校验命令与结果

### 1) 新增单测（Phase 4）

```bash
./.venv/bin/python -m pytest -q \
  tests/unit/test_dedup2d_api_usability.py \
  tests/unit/test_dedup2d_file_storage_s3.py \
  tests/unit/test_dedup2d_gc.py
```

结果：✅ `46 passed`

### 2) Dedup2D 关键集成测试

```bash
./.venv/bin/python -m pytest -q \
  tests/test_dedup_2d_proxy.py \
  tests/test_metrics_contract.py \
  tests/unit/test_dedup_2d_jobs_redis.py \
  tests/unit/test_dedup2d_webhook.py
```

结果：✅ `72 passed, 2 skipped`

### 3) 全量回归

```bash
./.venv/bin/python -m pytest -q
```

结果：✅ `3609 passed, 42 skipped`

### 4) E2E（本地文件存储）

```bash
./.venv/bin/python scripts/e2e_dedup2d_webhook.py --startup-timeout 90 --job-timeout 60
```

结果：✅ `OK: job completed + callback received + signature verified`

### 5) E2E（MinIO / S3 文件存储）

新增脚本（本次验收补齐）：

- `scripts/e2e_dedup2d_webhook_minio.py`

命令：

```bash
./.venv/bin/python scripts/e2e_dedup2d_webhook_minio.py --startup-timeout 120 --job-timeout 60
```

结果：✅ `OK: job completed + callback received + signature verified + minio cleaned`

说明：

- E2E 使用 fake vision 服务验证“HTTP 契约 + Redis/ARQ 调度 + 回调 + 存储”，不依赖真实 dedupcad-vision 的索引/数据。
- MinIO 通过 docker 启动；bucket 在脚本中用 boto3 创建；并验证 cleanup_on_finish 后 prefix 下无残留对象。

### 6) Helm 模板渲染

本机未预装 helm，验收时下载 helm `v3.16.4`（darwin/arm64）临时执行：

- 默认 values
- `--set dedup2d.storage.mode=local`
- `--set dedup2d.storage.mode=s3` + inline secret（覆盖 S3 secret 模板）

结果：✅ 三种渲染均成功（`helm template OK`）

## 验收中发现的问题与修复

### 问题：forced-async 触发导致 proxy 测试返回结构变化

现象：

- `tests/test_dedup_2d_proxy.py` 多个用例收到的是 async 响应（`{job_id,status,poll_url,...}`），导致断言访问 `duplicates` 时报 `KeyError: 'duplicates'`。

根因：

- `src/api/v1/dedup.py` 的 forced-async 配置在 import 时读取环境变量并固定为默认“开启”，当 `geom_json` 或 `mode=precise` 时会强制走 async 分支。
- proxy 测试本意是验证“同步 pipeline + precision 复核”的输出结构，与 forced-async 行为冲突。

修复：

1) `src/api/v1/dedup.py`
   - forced-async 配置改为在 `_check_forced_async()` 调用时读取环境变量（避免 import-time 固化，便于测试/部署按 env 调整）
2) `tests/test_dedup_2d_proxy.py`
   - 增加 `autouse` fixture，默认关闭 forced-async（proxy 测试专注验证同步输出；forced-async 行为由 `tests/unit/test_dedup2d_api_usability.py` 覆盖）

修复后：✅ 相关测试通过，且 forced-async 单测仍通过。

## 备注 / 后续建议

- 建议把当前工作区变更整理为一组可追溯 commits（Phase 4 含 chart/alerts/dashboard/scripts/tests），便于 CI 与回滚。
- `claudedocs/phase4_dedup2d_summary.md` 中个别环境变量/脚本命名与代码存在出入（例如 GC 脚本实际为 `scripts/dedup2d_uploads_gc.py`），建议在交付前对齐文档与真实实现。

