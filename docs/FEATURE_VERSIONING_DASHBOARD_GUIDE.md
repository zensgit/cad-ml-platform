# Feature Versioning Dashboard 部署指南

> **目标**: 快速启动 Grafana Dashboard 监控 v5 特征迁移进度

---

## 📋 前置条件

1.  **Prometheus** 已安装并运行
2.  **Grafana** 已安装（推荐 v9.0+）
3.  CAD ML Platform 服务已启动，`/metrics` 端点可访问

---

## 🚀 快速部署（5 分钟）

### 步骤 1: 配置 Prometheus

将 `ops/prometheus/prometheus.yml` 复制到您的 Prometheus 配置目录：

```bash
# 方法 A: 直接替换（谨慎！）
cp ops/prometheus/prometheus.yml /etc/prometheus/prometheus.yml

# 方法 B: 合并配置（推荐）
# 将 scrape_configs 中的内容追加到现有配置
```

重启 Prometheus 使配置生效：
```bash
# Docker 方式
docker restart prometheus

# Systemd 方式
sudo systemctl restart prometheus
```

验证指标可见：
```bash
curl http://localhost:9090/api/v1/label/__name__/values | grep feature_version
# 应该看到: feature_version_counts, feature_upgrade_attempt_failed_total, etc.
```

---

### 步骤 2: 导入 Grafana Dashboard

#### 方法 A: UI 导入（推荐）
1.  打开 Grafana (http://localhost:3000)
2.  导航到 **Dashboards** → **Import**
3.  点击 **Upload JSON file**
4.  选择 `docs/grafana/feature_versioning_dashboard.json`
5.  选择 Prometheus 数据源
6.  点击 **Import**

#### 方法 B: API 导入（自动化）
```bash
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @docs/grafana/feature_versioning_dashboard.json
```

---

### 步骤 3: 验证 Dashboard

访问 Dashboard 后，您应该看到：
*   ✅ **饼图显示版本分布**（如果有向量数据）
*   ✅ **v5 采用率百分比**
*   ✅ **延迟对比图**（v4 vs v5）

如果图表为空：
1.  检查 Prometheus 是否能抓取到指标: `http://localhost:9090/targets`
2.  检查 `/metrics` 端点是否有数据: `curl http://localhost:8000/metrics | grep feature`
3.  确保至少有一个向量注册过（触发指标更新）

---

## 🎨 Dashboard 布局说明

### 第一行（概览）
*   **饼图**: 各版本占比（一目了然迁移进度）
*   **v5 采用率**: 关键指标，目标 >75%
*   **总向量数**: 数据规模监控
*   **升级失败总数**: 健康度仪表盘

### 第二行（详细监控）
*   **v5 升级失败趋势**: 按源版本细分
*   **长度不匹配警告**: 数据完整性监控

### 第三行（性能）
*   **延迟对比**: v4 vs v5 性能差异
*   **版本健康摘要表**: 每个版本的详细统计

### 第四行（深度分析）
*   **延迟热力图**: 发现异常值
*   **失败类型分解**: 问题定位

### 第五行（迁移追踪）
*   **Legacy 向量统计**: v1-v3 待迁移数量
*   **v4 向量统计**: 存在"体积主导"问题的向量

---

## 📊 关键指标解读

| 指标名称 | 正常范围 | 告警阈值 | 含义 |
|----------|----------|----------|------|
| **v5 Adoption Rate** | >50% | <25% 🔴 | v5 渗透率低，需加速迁移 |
| **Upgrade Failures Total** | <10 | >100 🔴 | 升级过程出现严重问题 |
| **Length Mismatch Rate** | 0 | >5/min 🟡 | 数据完整性风险 |
| **v5 P95 Latency** | <1ms | >10ms 🟡 | 性能退化 |

---

## ⚠️ 常见问题排查

### 问题 1: Dashboard 显示 "No Data"
**原因**: Prometheus 未抓取到指标或服务未启动
**解决**:
```bash
# 检查 Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="cad-ml-platform")'

# 检查服务 /metrics 端点
curl http://localhost:8000/metrics | head -20
```

### 问题 2: 只看到 v4 向量，没有 v5
**原因**: 尚未启用 v5 或未重新提取特征
**解决**:
```bash
# 设置环境变量
export FEATURE_VERSION=v5

# 重启服务
systemctl restart cad-ml-platform
```

### 问题 3: "feature_version_counts" 指标不存在
**原因**: 代码版本过旧或未合并 Day 1 的代码
**解决**:
```bash
# 检查代码中是否有指标定义
grep -r "feature_version_counts" src/utils/analysis_metrics.py

# 如果没有，拉取最新代码
git pull origin main
```

---

## 🔧 高级配置

### 自定义刷新间隔
编辑 Dashboard JSON，修改 `refresh` 字段：
```json
"refresh": "5s"  // 可改为 "10s", "1m" 等
```

### 添加自定义告警
在 Dashboard 的任意 Panel 中：
1.  点击 Panel 标题 → **Edit**
2.  切换到 **Alert** 标签
3.  定义告警条件（如 v5 采用率 < 25%）
4.  配置通知渠道（Slack/Email）

### 导出为 PDF 报告
Grafana Enterprise 功能：
```bash
# 需要 Grafana Enterprise 或 Image Renderer 插件
grafana-cli plugins install grafana-image-renderer
systemctl restart grafana-server
```

---

## 📚 相关文档

*   **告警规则**: `ops/prometheus/alerts/feature_version_alerts.yml`
*   **指标定义**: `src/utils/analysis_metrics.py`
*   **API 文档**: `docs/FEATURE_EXTRACTION_V5.md`
*   **Sprint 总结**: `docs/FEATURE_VERSIONING_SPRINT_SUMMARY.md`

---

## ✅ 部署验收清单

- [ ] Prometheus 能抓取到 `feature_version_counts` 指标
- [ ] Grafana Dashboard 导入成功
- [ ] 至少看到 1 个 Panel 有数据（如版本分布饼图）
- [ ] 告警规则已加载（检查 Prometheus `/rules` 页面）
- [ ] v5 采用率显示正常（即使是 0%）

---

**部署完成后，您可以实时观察 v5 的迁移进度，并在出现异常时快速响应！**

*文档更新时间: 2025-11-28*
