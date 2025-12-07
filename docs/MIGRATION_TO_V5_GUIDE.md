# v5 Feature Migration Guide

> **目标**: 安全地将现有 v1-v4 向量迁移到 v5

---

## 📋 前置条件

1.  **依赖安装**:
    ```bash
    pip install aiohttp tqdm
    ```

2.  **服务运行**: CAD ML Platform API 必须可访问（默认 `http://localhost:8000`）

3.  **文件清单**: 准备 CAD 文件路径映射表（见下文）

---

## 🚀 快速开始

### 步骤 1: 准备文件清单

创建 `file_mapping.txt`，格式为 `doc_id,file_path`：

```
# 示例文件清单
part_12345,/data/cad/drawings/part_12345.dxf
assembly_67890,/data/cad/assemblies/asm_67890.step
plate_001,/data/cad/plates/plate_001.dxf
bracket_002,/data/cad/brackets/bracket_002.iges
```

**注意**：
*   `doc_id`: 向量存储中的唯一标识符
*   `file_path`: CAD 文件的绝对路径
*   以 `#` 开头的行为注释

### 步骤 2: 试运行（Dry Run）

在实际迁移前，先检查配置：

```bash
python scripts/migrate_to_v5.py \
  --file-list file_mapping.txt \
  --dry-run
```

输出示例：
```
[DRY RUN] Would migrate part_12345 from /data/cad/drawings/part_12345.dxf
[DRY RUN] Would migrate assembly_67890 from /data/cad/assemblies/asm_67890.step
...
```

### 步骤 3: 执行迁移（带备份）

```bash
python scripts/migrate_to_v5.py \
  --file-list file_mapping.txt \
  --backup \
  --batch-size 5
```

输出示例：
```
============================================================
CAD ML Platform - v5 Migration Tool
============================================================
API Base: http://localhost:8000/api/v1
Backup Dir: backups/migration_v5/20251128_154500
Log File: backups/migration_v5/migration_20251128_154500.log

Starting migration of 1247 files...
Dry run: False
Backup: True
Resume: False

Migrating: 100%|████████████████████| 1247/1247 [05:23<00:00, 3.86file/s]

============================================================
MIGRATION COMPLETE
============================================================
Total:     1247
Success:   1245 ✓
Failed:    2 ✗
Skipped:   0 -
Elapsed:   323.45s
Throughput: 3.85 files/sec
```

---

## 🔧 高级用法

### 断点续传

如果迁移中断（网络故障、服务重启等），可以恢复：

```bash
python scripts/migrate_to_v5.py \
  --file-list file_mapping.txt \
  --resume
```

脚本会跳过已完成的文件，只处理剩余部分。

### 自定义备份目录

```bash
python scripts/migrate_to_v5.py \
  --file-list file_mapping.txt \
  --backup \
  --backup-dir /mnt/backups/v5_migration
```

### 调整并发度

```bash
# 提高并发（服务器性能强）
python scripts/migrate_to_v5.py \
  --file-list file_mapping.txt \
  --batch-size 20

# 降低并发（避免过载）
python scripts/migrate_to_v5.py \
  --file-list file_mapping.txt \
  --batch-size 3
```

### 回滚到旧版本

```bash
python scripts/migrate_to_v5.py --rollback
```

**注意**: 当前版本的回滚功能仅保留备份文件，完整的自动恢复功能开发中。
手动恢复：使用 `backups/migration_v5/<timestamp>/*.json` 中的备份数据。

---

## 📊 监控迁移进度

### 实时监控

在迁移过程中，打开 Grafana Dashboard (`docs/grafana/feature_versioning_dashboard.json`)：

*   **v5 采用率**: 实时上升
*   **版本分布饼图**: 看到 v5 占比增加
*   **升级失败监控**: 如果有问题，立即发现

### 日志查看

```bash
# 实时跟踪日志
tail -f backups/migration_v5/migration_*.log

# 检查失败记录
grep "✗" backups/migration_v5/migration_*.log
```

---

## ⚠️ 常见问题

### Q1: "File not found" 错误？
**A**: 检查 `file_mapping.txt` 中的路径是否正确：
```bash
# 验证文件是否存在
while IFS=, read -r doc_id path; do
  [ -f "$path" ] || echo "Missing: $path"
done < file_mapping.txt
```

### Q2: "Analysis failed: 500" 错误？
**A**: 可能是服务过载或 CAD 文件损坏。建议：
*   降低 `--batch-size`
*   检查服务日志：`docker logs cad-ml-platform`
*   跳过损坏文件，手动处理

### Q3: 迁移速度太慢？
**A**: 优化策略：
*   提高 `--batch-size`（前提是服务器承受得住）
*   确保 Redis 缓存可用（减少重复计算）
*   使用 SSD 存储 CAD 文件

### Q4: 如何生成 `file_mapping.txt`？
**A**: 如果您有数据库记录，可以导出：
```sql
-- MySQL 示例
SELECT doc_id, file_path 
INTO OUTFILE '/tmp/file_mapping.txt'
FIELDS TERMINATED BY ','
FROM cad_documents
WHERE feature_version < 'v5';
```

或者用脚本扫描文件系统：
```bash
# 扫描目录，生成映射表
find /data/cad -type f \( -name "*.dxf" -o -name "*.step" \) | while read path; do
  doc_id=$(basename "$path" | sed 's/\.[^.]*$//')
  echo "$doc_id,$path"
done > file_mapping.txt
```

---

## 🛡️ 安全建议

1.  **始终启用备份**: `--backup` 是您的保险
2.  **先测试一小批**: 用 10-20 个文件测试流程
3.  **监控服务健康度**: 观察 CPU/内存/磁盘
4.  **错峰迁移**: 避开业务高峰期
5.  **验证结果**: 迁移后抽查几个向量，对比新旧特征

---

## 📈 性能预期

基于内部测试（单核 CPU, 8GB RAM）：

| 文件类型 | 平均耗时 | 吞吐量 |
|----------|----------|--------|
| DXF (简单) | 150ms | ~6.7 files/sec |
| DXF (复杂) | 800ms | ~1.2 files/sec |
| STEP (中等) | 500ms | ~2.0 files/sec |
| STL (大型)  | 1.2s  | ~0.8 files/sec |

**实际吞吐量取决于**：
*   服务器性能（CPU 核心数）
*   CAD 文件复杂度
*   网络延迟（如果 API 是远程的）
*   并发度配置

---

## ✅ 迁移后验证

迁移完成后，执行以下检查：

### 1. 确认 v5 采用率
```bash
curl http://localhost:8000/api/v1/health/extended | jq '.feature_versions'
# 期望看到: {"v5": 1245, "v4": 2, ...}
```

### 2. 抽查特征向量
```bash
# 检查某个向量的版本
curl http://localhost:8000/api/v1/vectors/part_12345 | jq '.meta.feature_version'
# 应该输出: "v5"
```

### 3. 对比新旧特征
```bash
# 如果保留了备份，可以对比
diff <(jq -S . backups/migration_v5/<timestamp>/part_12345.json) \
     <(curl -s http://localhost:8000/api/v1/vectors/part_12345 | jq -S .)
```

### 4. 运行基准测试
```bash
# 验证相似度搜索仍然正常
python scripts/benchmark_v4_vs_v5.py
```

---

**迁移愉快！如遇问题，查看日志或联系开发团队。**

*文档版本: 1.0 | 更新时间: 2025-11-28*
