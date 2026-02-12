# Classifier Cache Hit Rate Low Runbook

## 触发条件
- 告警: `ClassifierCacheHitRateLow` 触发
- `cad_ml_classification_cache_hit_ratio` < 20% 持续 15 分钟

## 可能原因
1. 输入图纸唯一度过高，重复率低
2. 缓存容量过小导致频繁驱逐
3. 缓存被频繁清空（人工/脚本触发）
4. 分类模型版本或特征提取发生变化，导致缓存键失效
5. 限流策略收紧导致请求模式突变（批量请求减少）

## 诊断步骤
1. 查看 `classification_cache_hits_total` 与 `classification_cache_miss_total` 的速率
2. 检查 `classification_cache_size` 是否长期接近上限
3. 审计 `/cache/clear` 触发记录与调用方来源
4. 统计是否存在大量一次性批量请求（导致命中偏低）
5. 检查近期模型或特征版本变更记录

## 缓解措施
| 场景 | 操作 |
|------|------|
| 容量不足 | 增大缓存容量（代码内 `LRUCache(max_size=...)` 或后续配置化） |
| 命中偏低 | 评估是否需要按文件名/路径引入二级缓存或分组策略 |
| 频繁清空 | 限制 `/cache/clear` 使用或增加审批流程 |
| 版本切换 | 等待新版本稳定后再观测命中率 |

## 回滚策略
- 如果扩容导致内存压力，恢复原 max_size 并观察 10 分钟。

## 相关指标
- `classification_cache_hits_total`
- `classification_cache_miss_total`
- `classification_cache_size`
- `cad_ml_classification_cache_hit_ratio`
