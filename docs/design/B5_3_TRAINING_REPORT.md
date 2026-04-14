# B5.3 实施报告：生产监控 + 低置信度队列

**日期**: 2026-04-14  
**阶段**: B5.3 — PredictionMonitor + LowConfidenceQueue + HybridClassifier 集成  
**基线**: B5.2 — TextContentConfig 上线，单次 I/O 优化，INT8 量化脚本就绪  
**目标**: 生产监控上线，低置信度主动学习入口就绪

---

## 1. 实施概要

| 任务 | 状态 | 说明 |
|------|------|------|
| `src/ml/monitoring/prediction_monitor.py` | ✓ 已创建 | 滑动窗口置信度监控，漂移告警 |
| `src/ml/monitoring/__init__.py` 导出 | ✓ 已更新 | PredictionRecord / PredictionMonitor 已导出 |
| `src/ml/low_conf_queue.py` | ✓ 已创建 | 低置信度 CSV 队列，主动学习入口 |
| HybridClassifier 集成 | ✓ 已实现 | classify() 末尾自动 record + maybe_enqueue |
| `tests/unit/test_monitoring.py` | ✓ 已创建 | 28 个测试，覆盖全部功能 |
| `tests/unit/test_low_conf_queue.py` | ✓ 已创建 | 26 个测试，覆盖全部功能 |
| **测试结果** | **✓ 54/54 全通过** | — |

---

## 2. 技术实现

### 2.1 PredictionMonitor（src/ml/monitoring/prediction_monitor.py）

**核心设计**：滑动窗口（默认 1000 条）+ 告警去重冷却期（默认 300s）

```
PredictionRecord（dataclass）
  ├── timestamp: float          time.monotonic()
  ├── predicted_class: str      top1 预测类别
  ├── top1_confidence: float    top1 概率 [0,1]
  ├── confidence_margin: float  top1 - top2 差距
  ├── text_hit: bool            文字分类器是否命中
  ├── filename_used: bool       文件名分类器是否参与
  └── latency_ms: float         端到端推理延迟（ms）

PredictionMonitor
  ├── 滑动窗口属性
  │     ├── n                   当前窗口记录数
  │     ├── low_conf_rate       低置信度比例
  │     ├── text_hit_rate       文字命中率
  │     ├── filename_used_rate  文件名使用率
  │     ├── avg_confidence      平均置信度
  │     ├── avg_margin          平均差距
  │     ├── avg_latency_ms      平均延迟
  │     ├── p95_latency_ms      P95 延迟
  │     └── class_distribution  类别分布（Top-10）
  ├── 漂移检测
  │     ├── check_drift()       → bool（供健康检查接口调用）
  │     └── _check_drift()      内部自动触发（每次 record() 后）
  └── 告警规则（两个阈值）
        ├── low_conf_rate > 10% → WARNING: DRIFT ALERT [low_conf]
        └── text_hit_rate < 5%  → INFO: TEXT SIGNAL 丢失
```

**关键 Bug 修复**：`_can_alert()` 原用 `0.0` 作为"从未告警"哨兵，但当 `time.monotonic() < alert_cooldown_sec`（系统刚启动时）会错误地抑制首次告警。改用 `None` 作哨兵，首次调用无条件触发：

```python
def _can_alert(self, key: str) -> bool:
    now = time.monotonic()
    last = self._last_alert_time.get(key)          # None = 从未告警
    if last is None or now - last >= self.alert_cooldown_sec:
        self._last_alert_time[key] = now
        return True
    return False
```

**环境变量控制**：
```bash
export MONITOR_WINDOW_SIZE=2000       # 加大窗口（高流量场景）
```

---

### 2.2 LowConfidenceQueue（src/ml/low_conf_queue.py）

**文件格式**：Append-only CSV，字段如下：

| 字段 | 填写方 | 说明 |
|------|--------|------|
| `file_hash` | 系统 | MD5 前 12 字符，去重键 |
| `filename` | 系统 | 原始 DXF 文件名 |
| `predicted_class` | 系统 | Top-1 预测类别 |
| `confidence` | 系统 | Top-1 置信度（保留 4 位小数） |
| `source` | 系统 | 决策分支（hybrid/graph2d/…） |
| `timestamp` | 系统 | YYYY-MM-DD HH:MM:SS |
| `reviewed_label` | **人工** | 人工标注正确类别 |
| `notes` | **人工** | 可选备注 |

**关键功能**：
- `maybe_enqueue()` — 低于阈值自动入队（默认 0.50），同会话去重
- `size()` / `pending_review()` — 队列统计
- `reviewed_entries()` — 返回已标注记录（供主动学习增量训练使用）
- `dxf_file_hash(bytes)` — 便捷 MD5 工具函数

**环境变量控制**：
```bash
export LOW_CONF_QUEUE_PATH=data/review_queue/low_conf.csv
export LOW_CONF_QUEUE_THRESHOLD=0.50     # 入队置信度阈值
```

---

### 2.3 HybridClassifier 集成

在 `__init__()` 末尾初始化（惰性导入，避免循环依赖）：

```python
from src.ml.monitoring.prediction_monitor import PredictionMonitor
from src.ml.low_conf_queue import LowConfidenceQueue

self.monitor = PredictionMonitor(window_size=_monitor_window)
self.low_conf_queue = LowConfidenceQueue(queue_path=_queue_path, threshold=_queue_threshold)
```

在 `classify()` 末尾（`_attach_explanation()` 之后、`return result` 之前）：

```python
# B5.3: Record to monitor + low-confidence queue
_top1 = float(result.confidence or 0.0)
self.monitor.record(
    predicted_class=result.label or "unknown",
    top1_confidence=_top1,
    confidence_margin=_margin,
    text_hit="text_content_predicted" in result.decision_path,
    filename_used=filename_pred is not None,
)
if file_bytes:
    self.low_conf_queue.maybe_enqueue(
        file_hash=dxf_file_hash(file_bytes),
        filename=filename,
        predicted_class=result.label or "unknown",
        confidence=_top1,
        source=result.source.value,
    )
```

**容错设计**：整个 record/enqueue 块包在 `try/except` 中，任何异常仅打 DEBUG 日志，不影响主流程返回。

---

## 3. 测试覆盖

### 3.1 test_monitoring.py（28 个测试）

| 测试类 | 覆盖场景 |
|--------|---------|
| `TestWindowBehaviour` | 空窗口、记录计数、窗口上限、滑动丢弃最旧 |
| `TestStatistics` | 各统计属性的边界值和混合情况 |
| `TestDriftDetection` | 窗口太小不触发、高置信度不触发、两种漂移触发 |
| `TestAlertCooldown` | 首次告警、冷却抑制、过期后重新触发、不同 key 独立 |
| `TestSummary` | 必需字段、JSON 可序列化、空窗口、top5 上限 |
| `TestReset` | 清空记录、清空告警时间戳、重置后可继续记录 |

### 3.2 test_low_conf_queue.py（26 个测试）

| 测试类 | 覆盖场景 |
|--------|---------|
| `TestMaybeEnqueue` | 低于/等于/高于阈值、自定义阈值覆盖、多文件入队 |
| `TestDeduplication` | 同 hash 不重复、不同 hash 均入队、禁用去重 |
| `TestSizeAndPending` | 空队列、全待审、已审核排除 |
| `TestReviewedEntries` | 无审核返回空、人工填写后返回正确行 |
| `TestCsvIntegrity` | 头行存在、无重复头行、字段内容正确 |
| `TestDxfFileHash` | 返回类型、长度、确定性、不同内容不同 hash |

---

## 4. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/ml/monitoring/prediction_monitor.py` | ✓ 新建 | PredictionMonitor + PredictionRecord |
| `src/ml/monitoring/__init__.py` | ✓ 修改 | 导出 PredictionMonitor/PredictionRecord |
| `src/ml/low_conf_queue.py` | ✓ 新建 | LowConfidenceQueue + dxf_file_hash |
| `src/ml/hybrid_classifier.py` | ✓ 修改 | classify() 集成监控 + 队列 |
| `tests/unit/test_monitoring.py` | ✓ 新建 | 28 个测试，54/54 全通过 |
| `tests/unit/test_low_conf_queue.py` | ✓ 新建 | 26 个测试，54/54 全通过 |

---

## 5. 生产运维说明

### 5.1 定时摘要日志

建议在生产环境中每 N 次推理（或每分钟）打印监控摘要：

```python
import json, logging
logger = logging.getLogger(__name__)

def maybe_log_summary(clf, every_n=1000):
    if clf.monitor.n % every_n == 0 and clf.monitor.n > 0:
        summary = clf.monitor.summary()
        logger.info("MONITOR: %s", json.dumps(summary, ensure_ascii=False))
```

### 5.2 健康检查接口集成

```python
# 在健康检查接口中暴露 drift_detected
def health_check(clf):
    return {
        "status": "degraded" if clf.monitor.check_drift() else "ok",
        "monitor": clf.monitor.summary(),
    }
```

### 5.3 低置信度队列审核工作流

```bash
# 查看待审核数量
python3 -c "
from src.ml.low_conf_queue import LowConfidenceQueue
q = LowConfidenceQueue()
print(f'总计: {q.size()}，待审核: {q.pending_review()}')
"

# 审核：用编辑器打开 data/review_queue/low_conf.csv
# 在 reviewed_label 列填写正确类别，保存

# 将已审核样本追加到训练 manifest（B5.4 主动学习触发）
python3 scripts/append_reviewed_to_manifest.py \
    --queue data/review_queue/low_conf.csv \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output data/manifests/unified_manifest_v3.csv
```

---

## 6. B5.3 验收标准

| 指标 | 目标 | 状态 |
|------|------|------|
| PredictionMonitor 实现 | 代码完整，含全部统计属性 | ✓ 完成 |
| 低置信度告警 | rate > 10% → WARN 日志 | ✓ 完成 |
| 文字信号告警 | hit_rate < 5% → INFO 日志 | ✓ 完成 |
| 告警冷却去重 | 5 分钟内不重复 | ✓ 完成 |
| summary() JSON 可序列化 | 结构完整 | ✓ 完成 |
| LowConfidenceQueue 实现 | 会话去重，Append-only CSV | ✓ 完成 |
| HybridClassifier 集成 | classify() 自动 record + enqueue | ✓ 完成 |
| 集成容错 | monitor/queue 异常不影响主流程 | ✓ 完成 |
| 单元测试全通过 | 54/54 | ✓ 完成 |

---

## 7. 里程碑追踪

| 里程碑 | 内容 | 目标 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + 模型 v3 | 91.0% | ✓ |
| B5.1 | 文字融合（三路） | 94.1% avg | ✓ |
| B5.2a | 文字缓存集成 | 单次 I/O | ✓ |
| B5.2b | INT8 量化 | 待运行验证 | ⏳ |
| **B5.3** | **监控 + 低置信度队列** | **生产监控上线** | **✓ 完成** |
| B5.4 | 主动学习增量训练 | ≥ 93% 无文件名 | 待实施 |

---

*报告生成: 2026-04-14*
