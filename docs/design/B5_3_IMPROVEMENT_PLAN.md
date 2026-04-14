# B5.3 提升计划：生产监控 + 漂移检测

**日期**: 2026-04-14  
**基线**: B5.2 — P50 < 150ms，INT8 精度损失 < 0.5pp  
**目标**: 生产就绪 — 置信度监控 + 分布漂移告警 + 低置信度队列

---

## 1. 监控需求分析

### 1.1 生产风险场景

| 场景 | 风险 | 检测指标 |
|------|------|---------|
| 新 DXF 格式 / 工具生成 | 图结构分布偏移 | Top-1 置信度下降 |
| 新零件类别出现（未见过） | 模型强行预测已知类 | 置信度 < 0.5 且均匀分布 |
| GBK 编码变体导致文字提取失败 | 文字信号静默丢失 | 文字命中率骤降 |
| 数据管道变更（节点维度变化） | 推理异常 | 错误率 |
| 季节性业务变化（订单结构） | 类别分布漂移 | 类别频率偏移 |

### 1.2 监控指标体系

```
Layer 1 - 实时指标（每次推理）
  ├── top1_confidence       最高置信度
  ├── confidence_margin     top1 - top2 差距
  ├── text_hit              是否有文字信号（bool）
  └── predicted_class       预测类别

Layer 2 - 窗口统计（滑动窗口 1000 次）
  ├── low_conf_rate         置信度 < 0.6 比例
  ├── text_hit_rate         文字命中率
  ├── class_distribution    各类预测频率
  └── avg_confidence        平均置信度

Layer 3 - 告警触发
  ├── low_conf_rate > 10%   → WARN: 置信度漂移
  ├── text_hit_rate < 5%    → INFO: 文字信号丢失
  └── class_freq > 3σ       → WARN: 类别分布异常
```

---

## 2. B5.3 实施方案

### 2.1 PredictionMonitor（核心监控类）

```python
# src/ml/monitoring.py

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """单次推理记录"""
    timestamp: float
    predicted_class: str
    top1_confidence: float
    confidence_margin: float   # top1 - top2
    text_hit: bool             # 文字分类器是否命中
    filename_used: bool        # 文件名是否参与融合
    latency_ms: float


class PredictionMonitor:
    """监控推理置信度分布，检测模型漂移。

    设计原则：
    - 滑动窗口（默认 1000 次），避免历史数据污染当前状态
    - 低置信度阈值 0.6：24 类分布下 top1 < 0.6 通常表示高度不确定
    - 告警去重：同类告警 5 分钟内不重复触发
    """

    LOW_CONF_THRESHOLD: float = 0.60
    DRIFT_ALERT_RATE: float = 0.10      # 低置信度比例 > 10% 触发告警
    TEXT_HIT_ALERT_RATE: float = 0.05   # 文字命中率 < 5% 触发信息日志
    ALERT_COOLDOWN_SEC: int = 300       # 同类告警最小间隔（秒）

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._records: deque[PredictionRecord] = deque(maxlen=window_size)
        self._last_alert: dict[str, float] = {}

    # ── 记录 ──────────────────────────────────────────

    def record(
        self,
        predicted_class: str,
        top1_confidence: float,
        confidence_margin: float = 0.0,
        text_hit: bool = False,
        filename_used: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """记录一次推理结果，并触发漂移检测。"""
        rec = PredictionRecord(
            timestamp=time.time(),
            predicted_class=predicted_class,
            top1_confidence=top1_confidence,
            confidence_margin=confidence_margin,
            text_hit=text_hit,
            filename_used=filename_used,
            latency_ms=latency_ms,
        )
        self._records.append(rec)
        self._check_drift()

    # ── 统计属性 ───────────────────────────────────────

    @property
    def n(self) -> int:
        return len(self._records)

    @property
    def low_conf_rate(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.top1_confidence < self.LOW_CONF_THRESHOLD) / self.n

    @property
    def text_hit_rate(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.text_hit) / self.n

    @property
    def avg_confidence(self) -> float:
        if not self._records:
            return 0.0
        return statistics.mean(r.top1_confidence for r in self._records)

    @property
    def avg_latency_ms(self) -> float:
        if not self._records:
            return 0.0
        return statistics.mean(r.latency_ms for r in self._records)

    @property
    def class_distribution(self) -> dict[str, float]:
        if not self._records:
            return {}
        counts = Counter(r.predicted_class for r in self._records)
        return {cls: count / self.n for cls, count in counts.most_common()}

    # ── 漂移检测 ───────────────────────────────────────

    def _should_alert(self, alert_key: str) -> bool:
        """告警去重：同类告警冷却期内不重复触发。"""
        now = time.time()
        last = self._last_alert.get(alert_key, 0)
        if now - last >= self.ALERT_COOLDOWN_SEC:
            self._last_alert[alert_key] = now
            return True
        return False

    def _check_drift(self) -> None:
        """检测漂移并触发告警（仅在窗口达到 100 次后开始检测）。"""
        if self.n < 100:
            return

        # 低置信度漂移
        rate = self.low_conf_rate
        if rate > self.DRIFT_ALERT_RATE and self._should_alert("low_conf"):
            logger.warning(
                "DRIFT ALERT: %.1f%% of last %d predictions have confidence < %.0f%%"
                "  (threshold: %.0f%%)",
                rate * 100, self.n, self.LOW_CONF_THRESHOLD * 100,
                self.DRIFT_ALERT_RATE * 100,
            )

        # 文字命中率骤降
        txt_rate = self.text_hit_rate
        if txt_rate < self.TEXT_HIT_ALERT_RATE and self._should_alert("text_hit"):
            logger.info(
                "TEXT SIGNAL: only %.1f%% of last %d predictions have keyword hits "
                "(expected ~15%% on typical DXF batches)",
                txt_rate * 100, self.n,
            )

    def summary(self) -> dict:
        """返回当前监控摘要（适合定时 dump 到日志或监控系统）。"""
        return {
            "n": self.n,
            "avg_confidence": round(self.avg_confidence, 4),
            "low_conf_rate": round(self.low_conf_rate, 4),
            "text_hit_rate": round(self.text_hit_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "top5_classes": list(self.class_distribution.items())[:5],
        }
```

### 2.2 与 HybridClassifier 集成

```python
# src/ml/hybrid_classifier.py（新增监控集成）

import time
from src.ml.monitoring import PredictionMonitor

class HybridClassifier:
    def __init__(self, ...):
        ...
        self.monitor = PredictionMonitor(window_size=1000)

    def predict(self, dxf_bytes: bytes, filename: str) -> dict:
        t0 = time.perf_counter()

        # 1. 图特征 + 文字内容（单次 ezdxf 读取）
        # 2. 三路融合
        result = self._predict_impl(dxf_bytes, filename)

        latency_ms = (time.perf_counter() - t0) * 1000

        # 记录推理结果供监控
        probs = result.get("probabilities", {})
        sorted_probs = sorted(probs.values(), reverse=True)
        top1 = sorted_probs[0] if sorted_probs else 0.0
        margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else top1

        self.monitor.record(
            predicted_class=result.get("predicted_class", "unknown"),
            top1_confidence=top1,
            confidence_margin=margin,
            text_hit=result.get("text_hit", False),
            filename_used=result.get("filename_used", False),
            latency_ms=latency_ms,
        )

        return result
```

### 2.3 定时摘要日志

```python
# 建议在生产部署中每 N 次推理（或每分钟）打印监控摘要

import json

def log_monitor_summary(clf: HybridClassifier, every_n: int = 1000) -> None:
    if clf.monitor.n % every_n == 0 and clf.monitor.n > 0:
        summary = clf.monitor.summary()
        logger.info("MONITOR [n=%d]: %s", summary["n"], json.dumps(summary, ensure_ascii=False))
```

---

## 3. 低置信度队列（主动学习入口）

### 3.1 方案概述

当推理置信度 < 0.5 时，将样本 hash 写入"待标注队列"，供人工审核后追加到训练集：

```python
# src/ml/low_conf_queue.py

import csv
import hashlib
from pathlib import Path

class LowConfidenceQueue:
    """收集低置信度预测供人工审核，支持主动学习。"""

    def __init__(self, queue_path: str = "data/review_queue/low_conf.csv"):
        self.path = Path(queue_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def maybe_enqueue(
        self,
        dxf_bytes: bytes,
        filename: str,
        predicted_class: str,
        confidence: float,
        threshold: float = 0.50,
    ) -> bool:
        """如置信度低于阈值，将记录写入队列。返回是否入队。"""
        if confidence >= threshold:
            return False
        file_hash = hashlib.md5(dxf_bytes).hexdigest()[:12]
        self._append({
            "file_hash": file_hash,
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.4f}",
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        })
        return True

    def _ensure_header(self) -> None:
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["file_hash", "filename", "predicted_class", "confidence", "timestamp"]
                )
                writer.writeheader()

    def _append(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file_hash", "filename", "predicted_class", "confidence", "timestamp"]
            )
            writer.writerow(row)
```

---

## 4. 验收标准

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| PredictionMonitor 上线 | 代码完整，集成到 HybridClassifier | 代码审查 |
| 低置信度告警 | rate > 10% → WARN 日志 | 集成测试（注入低置信样本） |
| 文字信号告警 | hit_rate < 5% → INFO 日志 | 集成测试 |
| 监控摘要 | 每 1000 次可导出 summary dict | 单元测试 |
| 低置信度队列 | confidence < 0.5 自动入队 | 单元测试 |
| 队列 CSV 格式 | 正确写入，不重复头部 | 单元测试 |

---

## 5. 实施步骤

```
Week 3 (B5.3a): 监控核心
  → 创建 src/ml/monitoring.py (PredictionMonitor)
  → 单元测试 tests/unit/test_monitoring.py
  → 集成到 HybridClassifier.predict()

Week 4 (B5.3b): 低置信度队列
  → 创建 src/ml/low_conf_queue.py (LowConfidenceQueue)
  → 集成到推理流程
  → 定时摘要日志

Week 5 (B5.4): 主动学习闭环
  → 人工审核 → 标注 → 追加到训练 manifest
  → 触发增量训练（fine-tune v3 → v4）
  → 目标：acc ≥ 93%（原始 val set，无文件名场景）
```

---

## 6. 里程碑追踪

| 里程碑 | 内容 | 目标 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + 模型 v3 | 91.0% | ✓ |
| B5.1 | 文字融合（三路） | 94.1% avg | ✓ |
| B5.2a | 文字缓存集成 | 单次 I/O | ✓ 代码完成 |
| B5.2b | INT8 量化 | < 0.5pp 损失 | 待运行 |
| **B5.3a** | **PredictionMonitor** | **生产监控上线** | 待实施 |
| **B5.3b** | **低置信度队列** | **主动学习入口** | 待实施 |
| B5.4 | 主动学习增量训练 | ≥ 93%（无文件名） | 待规划 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
