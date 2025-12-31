# Dedup2D（cad-ml-platform）PR 与生产上线 Checklist

## 1) PR Checklist（合并前）

### 代码与兼容性

- [ ] Redis backend：`DEDUP2D_ASYNC_BACKEND=redis` 时 API/Worker 可独立扩容（无 in-process 状态依赖）
- [ ] Rolling upgrade：如需灰度兼容旧 worker，确认已按顺序使用：
  - [ ] 先升级 worker（支持 `file_ref` 与 `file_bytes_b64`）
  - [ ] 再按需开启 `DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64=1`（可选）
  - [ ] 全量升级后关闭该 flag（回到 `file_ref`-only）
  - [ ] 观察 `dedup2d_payload_format_total` 与 `dedup2d_legacy_b64_fallback_total`，确认旧格式逐步归零
- [ ] 不引入循环依赖：生产推荐由 `cad-ml-platform` 调用 `dedupcad-vision`（Vision 端调用 ML 平台默认关闭）

### 必跑测试

在 `cad-ml-platform` 仓库：

- [ ] 单测：`./.venv/bin/python -m pytest -q`
- [ ] 关键覆盖：
  - [ ] `./.venv/bin/python -m pytest -q tests/unit/test_dedup2d_job_list_redis.py`
  - [ ] `./.venv/bin/python -m pytest -q tests/test_dedup_2d_proxy.py`
  - [ ] `./.venv/bin/python -m pytest -q tests/unit/test_dedup2d_webhook.py`
  - [ ] `./.venv/bin/python -m pytest -q tests/test_metrics_contract.py`
- [ ]（可选）E2E（本机）：`./.venv/bin/python scripts/e2e_dedup2d_webhook.py`
- [ ]（可选）E2E（MinIO）：`./.venv/bin/python scripts/e2e_dedup2d_webhook_minio.py`

### 配置/文档

- [ ] `charts/cad-ml-platform/values.yaml` 的 dedup2d 配置段已更新（若新增 env/行为）
- [ ] 监控规则与 dashboard 跟随代码变更（指标名/label 不破坏 contract）

## 2) 生产上线 Checklist（部署前）

### 部署形态（强建议）

- [ ] `cad-ml-platform` API 与 `dedup2d-worker` 拆分 Deployment（不同扩缩容曲线）
- [ ] 不假设共享磁盘：生产建议 `dedup2d.storage.mode=s3`（MinIO/AWS S3）
- [ ] Redis 独立部署（或托管 Redis），确保持久化与备份策略

### 关键配置（建议值）

#### Helm values（示例）

```yaml
dedup2d:
  worker:
    enabled: true
    replicaCount: 2
    resources:
      requests:
        cpu: 200m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 2Gi

  storage:
    mode: s3
    s3:
      bucket: dedup2d-uploads
      prefix: uploads
      endpoint: http://minio:9000
      region: us-east-1
      existingSecret: cad-ml-s3-credentials
      retentionSeconds: 86400
      cleanupOnFinish: true

  redis:
    url: redis://redis:6379/0
    keyPrefix: dedup2d
    queueName: dedup2d:queue

  job:
    maxConcurrency: 4
    maxJobs: 1000
    ttlSeconds: 86400
    timeoutSeconds: 300

  vision:
    url: http://dedupcad-vision:58001
    timeoutSeconds: 60

  callback:
    allowHttp: false
    blockPrivateNetworks: true
    resolveDns: false
    allowlist: "your-callback-domain.example.com"
    hmacSecret: ""  # 建议用 Secret 注入
    timeoutSeconds: 10
    maxAttempts: 3
```

### 安全与风控

- [ ] 回调（Webhook）默认仅 `https`，并启用 allowlist（推荐）
- [ ] 配置 `DEDUP2D_CALLBACK_HMAC_SECRET`（建议通过 K8s Secret 注入）
- [ ] `DEDUP2D_ASYNC_MAX_JOBS` 设置合理上限并结合 `Retry-After`（避免队列打爆）
- [ ]（建议）对 `/api/v1/dedup/2d/search` 配置网关层限流/上传大小限制

### 可观测性

- [ ] Prometheus 能 scrape `cad-ml-platform` `/metrics`
- [ ] Grafana 预置 `Dedup2D Dashboard` 可见
- [ ] Alert rules 已加载并在 staging 验证触发/静默策略
- [ ] `dedupcad_vision_requests_total` / `dedupcad_vision_circuit_state` 指标稳定（无持续熔断）

## 3) 回滚/降级 Checklist

- [ ] Worker 异常：先将 `dedup2d.worker.enabled=false`（仅停止异步处理，API 仍可工作但 async job 会堆积/失败）
- [ ] Vision 不可用：根据业务策略对 dedup 入口做熔断/降级（返回可解释错误码，避免长时间阻塞）
- [ ] 必要时降级到单实例 in-process：
  - [ ] `DEDUP2D_ASYNC_BACKEND=inprocess`
  - [ ] `WORKERS=1`（避免多 worker 状态不一致）
