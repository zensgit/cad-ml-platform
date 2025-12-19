# Dedup2D（cad-ml-platform ⇄ dedupcad-vision）端到端校验报告

日期：2025-12-18

## 目标

在当前代码状态下，对 Dedup2D 的关键链路做“端到端 + 回归”验证，回答：

- 查重核心调用链是否闭环（API → Redis/ARQ → Vision → 回写结果 / 可选回调）
- Vision 与 ML Platform 的接口契约是否一致（避免“调用正常但回传丢失/超时/格式不兼容”）
- 两仓库单测/集成测试是否通过（避免隐藏回归）

## 结论（摘要）

- ✅ `cad-ml-platform` Dedup2D Redis/ARQ 异步链路 E2E 通过（含 webhook 回调签名校验、上传文件清理校验）
- ✅ `cad-ml-platform` 全量测试通过：`3537 passed, 42 skipped`
- ✅ `dedupcad-vision` 全量测试通过：`2088 passed, 19 skipped`
- ✅ Vision 接口与 ML Platform 的调用参数/路径匹配：`POST /api/v2/search`（multipart + Form 字段）
- ✅ Worker 支持 rolling upgrade：既能消费新 payload（`file_ref`），也兼容旧 payload（`file_bytes_b64`）
- ⚠️ 仅观察到若干 warning（不影响通过）；见“观察到的 warnings”

## 校验范围与关键接口契约

### 1) cad-ml-platform → dedupcad-vision（2D 视觉召回）

- 调用封装：`src/core/dedupcad_vision.py`
- 目标接口：`POST /api/v2/search`
- 传输方式：multipart 文件字段 `file` + Form 字段：
  - `mode`（fast/balanced/precise）
  - `max_results`
  - `compute_diff`
  - `enable_ml`
  - `enable_geometric`

Vision 侧对应实现：`dedupcad-vision/src/caddedup_vision/api/routes_progressive.py` 的 `@router.post("/search")`（挂载到 `/api/v2/search`）。

### 2) cad-ml-platform 内部异步（Redis + ARQ）

- API 入口：`cad-ml-platform/src/api/v1/dedup.py`
  - `POST /api/v1/dedup/2d/search?async=true`：提交 job，返回 `job_id`
  - `GET /api/v1/dedup/2d/jobs/{job_id}`：轮询 job 状态/结果
  - `POST /api/v1/dedup/2d/jobs/{job_id}/cancel`：取消
- Redis backend：`cad-ml-platform/src/core/dedupcad_2d_jobs_redis.py`
  - payload 默认存 `file_ref`（不再内嵌大文件到 Redis）
  - 灰度兼容：可选开启 `DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64=1` 让 payload 同时携带 `file_bytes_b64`
- Worker：`cad-ml-platform/src/core/dedupcad_2d_worker.py`
  - 同时支持读取 `file_ref` 与 `file_bytes_b64`（滚动升级安全）

## 执行的验证步骤与结果

### A. cad-ml-platform：关键测试集（Dedup2D/Redis/Webhook）

命令：

```bash
./.venv/bin/python -m pytest -q \
  tests/unit/test_dedup_2d_jobs_redis.py \
  tests/unit/test_dedup2d_webhook.py \
  tests/test_dedup_2d_proxy.py
```

结果：✅ `34 passed`

### B. cad-ml-platform：全量回归

命令：

```bash
./.venv/bin/python -m pytest -q
```

结果：✅ `3537 passed, 42 skipped`

### C. cad-ml-platform：端到端 E2E（Redis + ARQ + 回调）

命令：

```bash
./.venv/bin/python scripts/e2e_dedup2d_webhook.py --startup-timeout 90 --job-timeout 60
```

结果：✅ 输出 `OK: job completed + callback received + signature verified`

说明：

- E2E 使用 fake vision 服务验证 HTTP 契约与全链路调度（不依赖 dedupcad-vision 的数据/索引状态）
- 同时覆盖：
  - Redis 入队/出队 + 结果回写
  - callback webhook（含 HMAC 签名校验）
  - `file_ref` 外置上传文件在 job 完成后清理（`DEDUP2D_FILE_STORAGE_CLEANUP_ON_FINISH=1`）

### D. dedupcad-vision：全量回归

命令：

```bash
python3 -m pytest -q
```

结果：✅ `2088 passed, 19 skipped`

## 观察到的 warnings（不阻塞）

### cad-ml-platform

- Pydantic 字段保护命名空间 warning（`model_version` 与 `model_` 前缀冲突）
- 若干 pytest mark 未注册（`slow`/`performance`）
- 个别资源/协程清理相关 warning（不影响测试通过）

### dedupcad-vision

- SciPy/NumPy 版本兼容性 warning（测试仍通过）

建议：上线前可单独开一个“warning clean-up”分支逐项收敛，但不建议在 Phase 4 主线里混合处理。

## 已有修复/报告索引（供审阅/追溯）

- `reports/dedup2d_redis_arq_enqueue_bugfix_report_20251218.md`
- `reports/dedup2d_phase25_filerefs_verification_report_20251218.md`
- `reports/dedup2d_phase3_webhook_e2e_verification_report_20251218.md`

## 下一步建议（与生产部署不确定性对齐）

由于你提到“Redis/API/Worker 是否在同一 Pod 共享磁盘暂不清楚”，建议按“不共享磁盘”作为生产默认假设：

- 文件：优先走 `DEDUP2D_FILE_STORAGE=s3`（MinIO/S3），避免依赖同 Pod/PVC 的调度耦合
- Redis：共享 Redis 没问题，但必须用 `DEDUP2D_REDIS_KEY_PREFIX` 隔离命名空间
- 部署：Worker 推荐独立 Deployment（可水平扩容），API 多副本时务必使用 Redis backend

