# Orphan Vectors Runbook

## Definition
向量存在于向量存储（内存 / Faiss / Redis）但缺少对应缓存分析结果键 `analysis_result:{id}`。

## Impact
- 相似度检索返回缺少上下文的向量，导致后续使用失败或降级。
- TTL / 冷清理策略可能失效，指标与分布统计偏差。

## Metrics & Alerts
- `vector_orphan_total`: 每次扫描新增孤儿数增量。
- `vector_orphan_scan_last_seconds`: 距离最近扫描的秒数。
- Alert: `OrphanVectorSpike` 在5分钟窗口新增孤儿数 > 总向量数的10%。
- Recording Rule: `cad_ml:vector_store_total` 用于占比基线。

## Common Causes
- 缓存过期策略过短导致分析结果被提前回收。
- 迁移或重启时未同步恢复缓存条目。
- 分析异常提前终止但向量已注册。
- 手动清理 Redis 键未清理内存向量。

## Investigation Steps
1. 检查 Redis 中 `analysis_result:*` 键过期策略 (TTL)。
2. 查看最近的错误日志是否存在解析 / 分析中途失败。
3. 对孤儿向量ID执行手动查询确认是否仍被业务侧使用。
4. 检查近期批量删除向量或迁移脚本执行情况。

## Remediation
- 执行待实现端点 `/api/v1/analyze/vectors/orphans?cleanup=1&threshold=0.1` 清理超阈值孤儿。（规划中）
- 调整缓存 TTL (`ANALYSIS_RESULT_TTL_SECONDS`) 与向量 TTL 对齐，避免生命周期错配。
- 在分析完成前推迟向量注册（若频繁出现半途失败，可调整注册顺序）。

## Prevention
- 向量注册与分析结果写入保持事务性（未来改造：先写结果再注册向量）。
- 迁移与重启时执行一致性校验任务，检测孤儿并清理。
- 设置最小 TTL 下限用于关键键，避免配置过低。

## Future Enhancements
- 自动清理策略：孤儿占比持续 >X% 触发自动删除并计入 `vector_cold_pruned_total{reason="orphan_cleanup"}`。
- 审计端点列出最近 N 个孤儿ID与注册时间戳。

