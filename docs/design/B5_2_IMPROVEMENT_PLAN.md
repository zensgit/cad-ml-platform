# B5.2 提升计划：推理性能优化 + 生产就绪

**日期**: 2026-04-14  
**基线**: B5.1 — 有文字场景 ≥ 93%，无文字场景 91%  
**目标**: 端到端推理 < 100ms，并发 ≥ 10 req/s，生产监控上线

---

## 1. 性能现状分析

### 1.1 推理延迟分解（估计）

| 步骤 | 预计耗时 | 占比 |
|------|---------|------|
| DXF 文件 I/O | ~20ms | 14% |
| DXF → Graph 转换（ezdxf 解析） | ~80ms | 56% |
| DXF 文字提取（新增 B5.1） | ~15ms | 11% |
| Graph2D 推理（GNN forward） | ~15ms | 11% |
| 文件名分类 | ~1ms | 1% |
| 文字关键词匹配 | ~1ms | 1% |
| 融合 + 排序 | ~1ms | 1% |
| **总计（冷启动，无缓存）** | **~133ms** | — |

> **关键瓶颈**：DXF→Graph 转换（ezdxf 解析）占 56%，已在 B4.1 用图缓存解决 70%+ 命中率
> **新增瓶颈**：文字提取 15ms（需要第二次 ezdxf 读取）

### 1.2 优化机会

| 优化 | 预计收益 | 难度 |
|------|---------|------|
| 文字+图形同步提取（单次 ezdxf 读取） | -15ms | 低 |
| Graph 缓存中附加文字缓存 | -30ms（重复文件） | 低 |
| INT8 模型量化 | -6ms (GNN推理) | 中 |
| 异步并行推理（asyncio） | -20ms（并发场景） | 中 |
| ONNX Runtime 导出 | -10ms | 高 |

---

## 2. B5.2 优化方案

### 2.1 文字内容附加到图缓存（优先级 P0）

**问题**: 当前文字提取需要第二次读取 DXF 文件，与 Graph 缓存不共享。  
**解决**: 在预处理阶段同时提取文字并缓存。

```python
# scripts/preprocess_dxf_to_graphs.py 新增文字缓存字段
def preprocess_dxf(dxf_path: str, cache_path: str):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # 图特征
    x, edge_index, edge_attr = dxf_to_graph(msp, ...)
    
    # 文字内容（同一次 ezdxf 读取）
    from src.ml.text_extractor import _extract_from_doc
    text_content = _extract_from_doc(doc)  # 不重新读文件
    
    torch.save({
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "label": label,
        "text": text_content,  # 新增
    }, cache_path)
```

**效果**: 消除重复 DXF 读取，节省 ~30ms（有缓存命中时）

### 2.2 文字+图形并行提取（P1）

```python
# src/ml/hybrid_classifier.py
import asyncio

async def predict_async(self, dxf_bytes: bytes, filename: str) -> dict:
    """Parallel extraction of graph features + text content."""
    
    # Parse DXF once, share doc object
    import io, ezdxf
    doc = ezdxf.read(io.BytesIO(dxf_bytes))
    
    # Parallel: graph conversion + text extraction (CPU-bound → thread pool)
    loop = asyncio.get_event_loop()
    graph_future = loop.run_in_executor(None, self._graph_from_doc, doc)
    text_future = loop.run_in_executor(None, self._text_from_doc, doc)
    
    graph, text = await asyncio.gather(graph_future, text_future)
    
    # GNN inference
    g2d_probs = self.graph2d_clf.predict_from_graph(graph)
    
    # Text classification
    text_probs = self.text_clf.predict_probs(text)
    
    # Filename classification
    fn_probs = self.filename_clf.predict(filename)
    
    # Fusion
    return self.fusion_engine.fuse(g2d_probs, text_probs, fn_probs)
```

### 2.3 INT8 模型量化（P2）

```python
# scripts/quantize_graph2d_model.py

import torch
from scripts.evaluate_graph2d_v2 import load_model

def quantize_model(checkpoint_path: str, output_path: str):
    model, label_map = load_model(checkpoint_path)
    
    # Dynamic quantization (no calibration data needed)
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers only
        dtype=torch.qint8,
    )
    
    # Save quantized checkpoint
    torch.save({
        "arch": "GraphEncoderV2",
        "model_state": quantized.state_dict(),
        "label_map": label_map,
        "quantized": True,
    }, output_path)
    
    print(f"Original size: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")
    print(f"Quantized size: {os.path.getsize(output_path) / 1024:.1f} KB")
```

**预期效果**:
- 模型文件大小: ~500KB → ~130KB（-74%）
- 推理延迟: ~15ms → ~9ms（-40%）
- 精度损失: < 0.5pp（Dynamic quantization 通常损失极小）

### 2.4 LRU 结果缓存（P3）

```python
# src/ml/hybrid_classifier.py
from functools import lru_cache
import hashlib

class HybridClassifier:
    def _get_cache_key(self, dxf_bytes: bytes) -> str:
        return hashlib.md5(dxf_bytes).hexdigest()[:16]
    
    @lru_cache(maxsize=1000)
    def _cached_predict(self, cache_key: str, filename: str) -> dict:
        # 实际推理（只在 cache miss 时执行）
        ...
    
    def predict(self, dxf_bytes: bytes, filename: str) -> dict:
        key = self._get_cache_key(dxf_bytes)
        return self._cached_predict(key, filename)
```

---

## 3. 基准测试方案

### 3.1 延迟基准测试脚本

```python
# scripts/benchmark_inference.py

import time, statistics, csv, random
from pathlib import Path

def benchmark(manifest_csv: str, n_files: int = 50, n_runs: int = 10) -> dict:
    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))
    
    samples = random.Random(42).sample(rows, min(n_files, len(rows)))
    
    from src.ml.hybrid_classifier import HybridClassifier
    clf = HybridClassifier()
    
    latencies = []
    for row in samples:
        dxf_bytes = Path(row["file_path"]).read_bytes()
        filename = Path(row["file_path"]).name
        
        for _ in range(n_runs):
            t0 = time.perf_counter()
            clf.predict(dxf_bytes=dxf_bytes, filename=filename)
            latencies.append((time.perf_counter() - t0) * 1000)
    
    return {
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies))],
        "p99_ms": sorted(latencies)[int(0.99 * len(latencies))],
        "mean_ms": statistics.mean(latencies),
        "n_measurements": len(latencies),
    }
```

### 3.2 并发吞吐量测试

```bash
# 使用 locust 进行并发测试
locust -f tests/load/locustfile.py --host=http://localhost:8000 \
    --users=20 --spawn-rate=5 --run-time=60s
```

---

## 4. 验收标准

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| P50 延迟（有缓存） | **< 50ms** | benchmark_inference.py |
| P50 延迟（无缓存） | **< 150ms** | benchmark_inference.py（cold start） |
| P95 延迟 | **< 300ms** | benchmark_inference.py |
| 内存峰值（单进程） | **< 500MB** | `memory_profiler` |
| 并发吞吐量 | **≥ 10 req/s** | locust |
| INT8 精度损失 | **< 0.5pp** | evaluate_graph2d_v2.py |
| 文字+图形单次读取 | 实现 | 代码审查 |

---

## 5. 实施步骤

```
Week 1 (B5.2a): 文字缓存集成
  → 修改 preprocess_dxf_to_graphs.py 同步提取文字
  → 重新运行预处理（仅新增 text 字段，其余缓存不变）
  → 更新 CachedGraphDataset 读取 text 字段
  → 更新 HybridClassifier 从缓存读文字（无需重读 DXF）

Week 2 (B5.2b): 量化 + 基准测试
  → 实现 quantize_graph2d_model.py
  → 量化 v3 模型，验证精度
  → 运行完整基准测试
  → 验证 P50 < 150ms 目标

Week 3 (B5.3): 监控上线
  → 实现 PredictionMonitor
  → 接入生产日志系统
  → 设置低置信度告警阈值（< 0.6）
```

---

## 6. B5.3 监控方案（前瞻）

```python
# src/ml/monitoring.py

class PredictionMonitor:
    """Monitor prediction confidence and detect distribution drift."""
    
    LOW_CONF_THRESHOLD = 0.6
    DRIFT_ALERT_RATE = 0.10  # Alert if >10% low confidence
    
    def __init__(self):
        self.window: list[float] = []
        self.window_size = 1000
    
    def record(self, confidence: float, predicted_class: str):
        self.window.append(confidence)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    @property
    def low_conf_rate(self) -> float:
        if not self.window:
            return 0.0
        return sum(1 for c in self.window if c < self.LOW_CONF_THRESHOLD) / len(self.window)
    
    def check_drift(self) -> bool:
        """Return True if drift detected."""
        rate = self.low_conf_rate
        if rate > self.DRIFT_ALERT_RATE:
            logger.warning("Confidence drift detected: %.1f%% low-conf predictions", rate * 100)
            return True
        return False
```

---

## 7. 里程碑追踪

| 里程碑 | 内容 | 目标 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + 模型 v3 | 92% | ✓ 91.0% |
| B5.1 | 文字融合 | 93% | 进行中 |
| **B5.2a** | **文字缓存集成** | **< 150ms** | 待实施 |
| **B5.2b** | **INT8 量化** | **精度 -0.5pp** | 待实施 |
| B5.3 | 监控上线 | 生产就绪 | 待实施 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
