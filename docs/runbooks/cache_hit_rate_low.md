# Cache Hit Rate Low Runbook

## 触发条件
- 告警: `CacheHitRateLow` 或 `FeatureCacheHitRateLowSlidingHour` 触发
- 命中率 <30% 且请求量阈值满足 (5m 或 sliding 1h)

## 可能原因
1. TTL 过短导致缓存刚建立即失效
2. 容量不足频繁驱逐 (查看 `feature_cache_evictions_total` 与 `feature_cache_size`)
3. 特征版本频繁变化 (v2/v3 切换) 导致键失效
4. 输入文件内容高变化度 (哈希不同) 缓存命中自然低
5. 冷启动阶段尚未达到稳定命中率

## 诊断步骤
1. Grafana 查看命中率时间序列与 `feature_cache_evictions_total` 曲线
2. 检查最近是否执行特征版本迁移或模型热更新
3. 抽样请求查看文件名 + 哈希是否高度唯一
4. 是否有异常错误码上涨 (解析失败、格式验证失败) 影响缓存使用

## 缓解措施
| 场景 | 操作 |
|------|------|
| TTL 太短 | 增大 `FEATURE_CACHE_TTL_SECONDS` (渐进调优) |
| 容量不足 | 增大 `FEATURE_CACHE_CAPACITY` 并观察驱逐下降 |
| 版本切换频繁 | 固定 `FEATURE_VERSION`，延后大版本迁移批次 |
| 高唯一度 | 评估是否需要引入重复检测或分层缓存策略 |

## 后续优化建议
- 增加缓存命中率自适应调参 (依据滑窗命中率动态调整 TTL)
- 增加细粒度指标 (按格式/材料分组命中率) 定位低命中来源

## 回滚策略
- 若调大的 TTL 或容量造成内存压力，恢复原值并观察 10 分钟。

## 相关指标
- `feature_cache_hits_last_hour` / `feature_cache_miss_last_hour`
- `feature_cache_evictions_total`
- `feature_cache_size`

