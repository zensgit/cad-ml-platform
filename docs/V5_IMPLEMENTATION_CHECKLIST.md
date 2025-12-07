# v5 Feature Versioning 实施检查清单

> **目标**: 安全、有序地部署 v5 特征提取系统  
> **预计总时长**: 2-4 小时（分阶段执行）  
> **建议执行顺序**: 按章节顺序，完成一个再进行下一个

---

## ✅ 准备阶段（10 分钟）

### 环境检查
- [ ] 确认 CAD ML Platform 服务正常运行
  ```bash
  curl http://localhost:8000/api/v1/health
  # 应返回 200 OK
  ```

- [ ] 确认 Prometheus 正常运行
  ```bash
  curl http://localhost:9090/-/healthy
  # 应返回 "Prometheus is Healthy."
  ```

- [ ] 确认 Grafana 正常运行
  ```bash
  curl http://localhost:3000/api/health
  # 应返回 {"database": "ok"}
  ```

### 代码验证
- [ ] 所有测试通过
  ```bash
  pytest tests/unit/test_golden_set_v5.py \
         tests/unit/test_feature_version_and_degenerate.py \
         tests/unit/test_feature_version_counts.py -v
  # 应显示 31 passed
  ```

- [ ] 基准测试脚本可运行
  ```bash
  /opt/homebrew/opt/python@3.13/bin/python3.13 scripts/benchmark_v4_vs_v5.py
  # 应显示性能对比结果
  ```

---

## 📊 阶段 1: 监控部署（30 分钟）

### 1.1 配置 Prometheus
- [ ] 复制告警规则到 Prometheus 配置目录
  ```bash
  sudo cp ops/prometheus/alerts/feature_version_alerts.yml \
          /etc/prometheus/rules/
  ```

- [ ] 验证规则语法
  ```bash
  promtool check rules /etc/prometheus/rules/feature_version_alerts.yml
  # 应显示 "SUCCESS: X rules found"
  ```

- [ ] 重载 Prometheus 配置
  ```bash
  curl -X POST http://localhost:9090/-/reload
  # 或者
  sudo systemctl reload prometheus
  ```

- [ ] 验证规则已加载
  ```bash
  curl http://localhost:9090/api/v1/rules | jq '.data.groups[].name'
  # 应包含 "feature_version_alerts"
  ```

### 1.2 导入 Grafana Dashboard
- [ ] 打开 Grafana UI: http://localhost:3000
- [ ] 导航到：**Dashboards** → **New** → **Import**
- [ ] 点击 **Upload JSON file**
- [ ] 选择文件：`docs/grafana/feature_versioning_dashboard.json`
- [ ] 选择数据源：Prometheus
- [ ] 点击 **Import**
- [ ] 验证 Dashboard 加载成功（可能暂时无数据，这是正常的）

### 1.3 验证指标可见
- [ ] 检查 Prometheus 能抓取到新指标
  ```bash
  curl http://localhost:9090/api/v1/label/__name__/values | \
    jq -r '.data[]' | grep feature_version
  ```
  预期输出：
  ```
  feature_version_counts
  feature_upgrade_attempt_failed_total
  feature_upgrade_length_mismatch_total
  feature_register_length_mismatch_total
  feature_extraction_latency_seconds
  ```

- [ ] 查询当前版本分布
  ```bash
  curl -G http://localhost:9090/api/v1/query \
    --data-urlencode 'query=feature_version_counts' | jq
  ```

**🎯 阶段 1 完成标志**：
*   Grafana Dashboard 可访问
*   至少能看到部分 Panel（即使数据为空）
*   Prometheus 能查询到 `feature_version_counts`

---

## 🧪 阶段 2: 小规模迁移测试（1-2 小时）

### 2.1 准备测试数据
- [ ] 创建测试文件列表（10-20 个文件）
  ```bash
  cat > test_migration.txt << EOF
  # 替换为您的实际文件
  test_part_001,/path/to/test_part_001.dxf
  test_part_002,/path/to/test_part_002.step
  EOF
  ```

- [ ] 验证文件存在
  ```bash
  while IFS=, read -r doc_id path; do
    [[ "$doc_id" =~ ^# ]] && continue
    [ -f "$path" ] || echo "❌ Missing: $path"
  done < test_migration.txt
  ```

### 2.2 安装迁移工具依赖
- [ ] 安装 Python 依赖
  ```bash
  pip install aiohttp tqdm
  ```

- [ ] 验证迁移脚本可运行
  ```bash
  python scripts/migrate_to_v5.py --help
  # 应显示使用说明
  ```

### 2.3 执行 Dry Run
- [ ] 运行 Dry Run 模式
  ```bash
  python scripts/migrate_to_v5.py \
    --file-list test_migration.txt \
    --dry-run
  ```

- [ ] 检查输出，确认：
  - [ ] 文件路径正确
  - [ ] doc_id 匹配现有向量
  - [ ] 无异常错误

### 2.4 执行测试迁移
- [ ] 运行实际迁移（带备份）
  ```bash
  python scripts/migrate_to_v5.py \
    --file-list test_migration.txt \
    --backup \
    --batch-size 3  # 保守设置
  ```

- [ ] 观察输出：
  - [ ] 进度条正常显示
  - [ ] 成功率 > 80%
  - [ ] 日志文件生成：`backups/migration_v5/migration_*.log`

### 2.5 验证迁移结果
- [ ] 检查某个向量的版本
  ```bash
  curl http://localhost:8000/api/v1/vectors/test_part_001 | \
    jq '.meta.feature_version'
  # 应输出: "v5"
  ```

- [ ] 在 Grafana Dashboard 观察变化
  - [ ] v5 采用率上升（应该从 0% 变为 ~10-20%，取决于测试文件数量）
  - [ ] 版本分布饼图显示 v5 切片

- [ ] 检查备份文件
  ```bash
  ls -lh backups/migration_v5/*/
  # 应显示备份的 JSON 文件
  ```

### 2.6 回归测试
- [ ] 运行基准测试，确保系统正常
  ```bash
  /opt/homebrew/opt/python@3.13/bin/python3.13 scripts/benchmark_v4_vs_v5.py
  ```

- [ ] 验证相似度搜索仍然工作
  ```bash
  curl -X POST http://localhost:8000/api/v1/similarity/search \
    -H "Content-Type: application/json" \
    -d '{"vector": [1.0, 0.5, ...], "top_k": 5}'
  # 应返回结果列表
  ```

**🎯 阶段 2 完成标志**：
*   至少 80% 的测试文件迁移成功
*   Dashboard 显示 v5 向量
*   系统功能正常（API 响应、搜索工作）

---

## 🚀 阶段 3: 全量迁移（时间取决于数据量）

**⚠️ 仅在阶段 2 成功后执行！**

### 3.1 生成完整文件列表
选择适合您的方法：

#### 方法 A: 从数据库导出
```sql
-- MySQL 示例
SELECT CONCAT(doc_id, ',', file_path)
INTO OUTFILE '/tmp/full_migration.txt'
FROM cad_documents
WHERE feature_version < 'v5' OR feature_version IS NULL;
```

#### 方法 B: 从文件系统扫描
```bash
find /data/cad -type f \( -name "*.dxf" -o -name "*.step" -o -name "*.iges" \) | \
  while read path; do
    doc_id=$(basename "$path" | sed 's/\.[^.]*$//')
    echo "$doc_id,$path"
  done > full_migration.txt
```

- [ ] 文件列表已生成
- [ ] 文件数量确认：`wc -l full_migration.txt`

### 3.2 分批迁移策略
- [ ] 计算总文件数和预计时间
  ```bash
  total=$(wc -l < full_migration.txt)
  echo "总文件数: $total"
  echo "预计耗时（按 2 files/sec）: $((total / 2 / 60)) 分钟"
  ```

- [ ] 决定是否分批执行
  - [ ] 如果 < 100 文件：一次性迁移
  - [ ] 如果 100-1000 文件：分 2-3 批
  - [ ] 如果 > 1000 文件：分 5-10 批，每批错峰执行

### 3.3 执行迁移
- [ ] 启动迁移（建议在低峰期）
  ```bash
  nohup python scripts/migrate_to_v5.py \
    --file-list full_migration.txt \
    --backup \
    --batch-size 5 \
    > migration.out 2>&1 &
  ```

- [ ] 监控进度
  ```bash
  # 方法 1: 查看日志
  tail -f backups/migration_v5/migration_*.log
  
  # 方法 2: 观察 Grafana Dashboard
  # v5 采用率应该持续上升
  ```

### 3.4 验证完成
- [ ] 检查迁移报告
  ```bash
  cat backups/migration_v5/migration_*.log | grep "MIGRATION COMPLETE" -A 10
  ```

- [ ] 确认成功率
  - [ ] 成功率 > 95%：✅ 优秀
  - [ ] 成功率 80-95%：🟡 可接受，检查失败原因
  - [ ] 成功率 < 80%：🔴 需要调查

- [ ] 在 Grafana 确认
  - [ ] v5 采用率 > 90%
  - [ ] 无持续的升级失败告警

**🎯 阶段 3 完成标志**：
*   绝大多数向量已迁移到 v5
*   Dashboard 显示健康状态
*   系统性能稳定

---

## 📈 后续优化（可选）

### A. API 契约升级
如果需要前端访问 v5 独有特征（如 `fill_ratio`）：
- [ ] 阅读 Phase 3 计划（待开发）
- [ ] 更新 Pydantic 响应模型
- [ ] 更新 API 文档
- [ ] 通知前端团队

### B. 性能调优
- [ ] 分析 v5 提取延迟
  ```promql
  histogram_quantile(0.95, 
    rate(feature_extraction_latency_seconds_bucket{version="v5"}[5m])
  )
  ```

- [ ] 如果 P95 > 500ms，考虑优化凸包计算

### C. 清理旧版本
- [ ] 在 v5 稳定运行 1-2 周后
- [ ] 考虑删除 v1-v3 向量（保留 v4 作为对照）
- [ ] 更新告警规则

---

## 🆘 故障排查

### 问题 1: Dashboard 无数据
**检查**:
```bash
# 1. Prometheus 能否访问服务
curl http://localhost:9090/targets

# 2. 服务是否暴露指标
curl http://localhost:8000/metrics | grep feature
```

### 问题 2: 迁移失败率高
**检查**:
```bash
# 1. 查看详细错误
grep "✗" backups/migration_v5/migration_*.log | head -20

# 2. 检查服务日志
docker logs cad-ml-platform | tail -50
```

### 问题 3: 性能下降
**检查**:
```bash
# 1. 对比 v4 和 v5 延迟
curl http://localhost:9090/api/v1/query \
  --data-urlencode 'query=feature_extraction_latency_seconds{quantile="0.95"}'
```

---

## 📞 支持资源

| 问题类型 | 参考文档 |
|----------|----------|
| Dashboard 部署 | `docs/FEATURE_VERSIONING_DASHBOARD_GUIDE.md` |
| 迁移工具使用 | `docs/MIGRATION_TO_V5_GUIDE.md` |
| v5 特征说明 | `docs/FEATURE_EXTRACTION_V5.md` |
| 告警规则 | `ops/prometheus/alerts/feature_version_alerts.yml` |
| Sprint 总结 | `docs/FEATURE_VERSIONING_SPRINT_SUMMARY.md` |

---

**建议执行时间表**：
*   **今天**: 完成阶段 1（监控部署）
*   **明天**: 完成阶段 2（小规模测试）
*   **本周内**: 完成阶段 3（全量迁移）

祝部署顺利！ 🚀
