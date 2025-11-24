## Faiss Index Stale Runbook

### 触发条件
告警 `FaissIndexStale`: 指标 `faiss_index_age_seconds > 3600` 持续 10 分钟。

### 可能原因
1. 长时间未执行导出/导入 (无重启或手动维护操作)。
2. 自动重建条件未满足（无大量删除 / 维度变化）。
3. 后台导出任务异常或被关闭。
4. Faiss 后端降级为内存未记录导入事件。

### 诊断步骤
1. 检查指标 `faiss_index_size` 是否在预期范围；为 0 时考虑是否仍在使用内存向量。
2. 查看日志中是否有 `faiss_export_total{status="error"}` 或初始化错误。
3. 验证环境变量：`VECTOR_STORE_BACKEND=faiss` 是否正确设置。
4. 若 `_FAISS_PENDING_DELETE` 大量积累未触发重建，手动调用重建端点（若已实现）。
5. 通过相似度查询端点确认查询延迟是否升高 (`vector_query_latency_seconds{backend="faiss"}`)。

### 处理措施
1. 手动触发重建或导出：调用管理端点（计划：`/api/v1/analyze/vectors/faiss/rebuild`）。
2. 若索引丢失或损坏，删除旧的索引文件并重建全量向量。
3. 增加定期导出任务或缩短自动重建阈值 (`FAISS_MAX_PENDING_DELETE`)。
4. 如果无需 Faiss，可将 backend 切换回 `memory` 暂缓告警。

### 预防建议
1. 设置定期导出间隔并记录 `faiss_index_age_seconds` 将其重置。
2. 在部署流程中添加索引导入验证步骤，如维度不匹配时自动重建。
3. 添加更精细的年龄分层告警（例如 2h 警告、6h 严重）。

### 相关指标
- `faiss_index_size`
- `faiss_export_total{status}` / `faiss_import_total{status}`
- `faiss_rebuild_total{status}` / `faiss_rebuild_duration_seconds`
- `vector_query_latency_seconds{backend="faiss"}`

### 回滚策略
若 Faiss 不可用，系统自动退回内存向量查询；确保业务关键路径仍可运行，仅性能下降。

