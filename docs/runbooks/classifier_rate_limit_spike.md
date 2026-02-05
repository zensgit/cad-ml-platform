# Classifier Rate Limit Spike Runbook

## 触发条件
- 告警: `ClassifierRateLimitedHigh` 触发
- `rate(classification_rate_limited_total[5m])` > 5/min 持续 10 分钟

## 可能原因
1. 请求突增（批量任务或重试风暴）
2. 客户端未做退避重试
3. 限流配置过低（`CLASSIFIER_RATE_LIMIT_PER_MIN`/`CLASSIFIER_RATE_LIMIT_BURST`）
4. 上游代理将多个客户端视为同一 IP（NAT/网关）

## 诊断步骤
1. 查看 `classification_rate_limited_total` 的增长速率
2. 查看网关或负载均衡器的客户端 IP 分布
3. 确认最近是否调整了限流环境变量
4. 抽样分析请求日志（是否重复请求同一文件）

## 缓解措施
| 场景 | 操作 |
|------|------|
| 突增流量 | 临时提升 `CLASSIFIER_RATE_LIMIT_BURST` |
| 正常负载但被限流 | 逐步提升 `CLASSIFIER_RATE_LIMIT_PER_MIN` |
| NAT 聚合 | 在网关层加入真实客户端 IP 透传 |
| 重试风暴 | 在客户端增加指数退避与抖动 |

## 回滚策略
- 恢复原限流配置并观察 10 分钟。

## 相关指标
- `classification_rate_limited_total`
