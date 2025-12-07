# 特征迁移指南：v4 → v5

v4 的体积分量主导（>99% 模长）会造成“只看体积”的相似度偏差。v5 引入 26 维（24 几何不变 + 2 语义），具备旋转/等比缩放不变性，并通过分量均衡避免单分量主导。

## 关键结论
- v5 维度：24（几何）+ 2（语义）= 26。
- 不支持 v4 → v5 直接就地升级（无损不可逆）。必须基于源 CAD 重新抽取。
- 度量：
  - `feature_version_usage_total{version}`：按版本抽取量。
  - `feature_upgrade_attempt_failed_total{from_version="v4",to_version="v5"}`：错误升级尝试。
  - `feature_v5_norm_max_component_ratio`：分量占比护栏，建议告警阈值 0.85。

## 迁移流程（推荐）
1. 写双：新入库仅写 v5；保留 v4 索引只读。
2. 批量重抽取：离线任务读取源 CAD，跑 v5 抽取并注册向量。
3. 验证：
   - 版本采用度面板（v5 比例）。
   - 质量护栏：最大分量占比直方图 < 0.8。
   - p95 延迟面板（`feature_extraction_latency_seconds{version="v5"}`）。
4. 切主：v5 覆盖率达到阈值（例如 ≥80%）后切换检索读路径；保留 v4 Z 天以回滚。

## 工具
- 脚本：`scripts/migrate_v4_to_v5.py`（dry-run 统计；--apply 将拒绝并提示重新抽取）。
- API：`GET /api/v1/vectors/migrate/preview` 可评估维度变化，但对 v5 会报错，属于预期。

## 告警建议
- 采用率低：部署后 24h 内 v5 占比 < 60%。
- 升级失败速率：`increase(feature_upgrade_attempt_failed_total{to_version="v5"}[15m]) > 0`。
- 分量占比异常：`max(feature_v5_norm_max_component_ratio) > 0.85` 连续 10 分钟。

## 回滚
- 环境切换：`FEATURE_VERSION=v4`（新入库回退到 v4）。
- 保留旧索引：直接读取 v4 索引进行检索，期间暂停迁移作业。

