# B5 验证计划：端到端精度与生产就绪性

**日期**: 2026-04-14  
**范围**: B5.0（数据增强）→ B5.1（文字融合）→ B5.2（性能）→ B5.3（监控）  
**最终目标**: 有名 ≥ 99%，无名 ≥ 93%，延迟 < 100ms，监控上线

---

## 1. 验证框架概览

```
┌─────────────────────────────────────────────────────────┐
│                  验证层次                                │
├─────────────────────────────────────────────────────────┤
│ L1: 模型级      — Graph2D 模型精度（per-class recall）  │
│ L2: 融合级      — HybridClassifier 各场景精度           │
│ L3: 系统级      — 端到端推理流（DXF bytes → label）     │
│ L4: 性能级      — 延迟、内存、吞吐量                   │
│ L5: 回归级      — Superpass gate（现有测试不退步）      │
└─────────────────────────────────────────────────────────┘
```

---

## 2. L1：模型级验证

### 2.1 评估脚本

```bash
# 在原始 manifest 上评估（排除增强数据泄漏）
python3 scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --seed 42

# 对比 B4.4 vs B5.0
python3 scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --baseline models/graph2d_finetuned_24class_v2.pth \
    --manifest data/graph_cache/cache_manifest.csv
```

### 2.2 验收指标（L1）

| 指标 | B4.5 基线 | B5.0 目标 | 测量方法 |
|------|----------|---------|---------|
| Overall Accuracy | 90.5% | **≥ 92%** | val set 914 样本 |
| Top-3 Accuracy | 99.3% | **≥ 99%** | val set |
| Macro F1 | 0.885 | **≥ 0.910** | val set |
| 轴承座 recall | 60% | **≥ 80%** | per-class report |
| 阀门 recall | 33% | **≥ 65%** | per-class report |
| 任意类 recall | min=33% | **min ≥ 50%** | per-class report |

### 2.3 重训练触发条件

若 B5.0 结果未达以下门槛，触发重训练（调整超参数）：

```
if overall_acc < 91.5%: 调高 focal_gamma 到 2.0，增加 epochs 到 80
if 轴承座 recall < 75%: 轴承座增强目标提升到 300，重运行 augment + retrain
if macro_f1 < 0.90: 检查增强是否引入过拟合，降低 aug sigma
```

---

## 3. L2：融合级验证

### 3.1 四场景测试集定义

**测试集构建原则**：
- 从原始验证集（914 样本）中选取
- 场景 B/C/D 使用相同文件，仅改变输入方式

| 场景 | 文件名 | 文字 | 说明 |
|------|--------|------|------|
| **A** | 原始文件名 | 原始文字 | 正常使用 |
| **B** | UUID（无意义） | 原始文字 | 文件更名后 |
| **C** | UUID（无意义） | 空字符串 | 纯图形推理 |
| **D** | 错误类别名 | 原始文字 | 对抗测试 |

### 3.2 测试集构建脚本

```python
# scripts/build_scenario_testsets.py

def build_scenario_b(samples: list) -> list:
    """Replace filename with UUID, keep DXF content."""
    return [(str(uuid.uuid4()) + ".dxf", dxf_bytes, true_label)
            for _, dxf_bytes, true_label in samples]

def build_scenario_c(samples: list) -> list:
    """UUID filename + strip all text entities from DXF."""
    return [(str(uuid.uuid4()) + ".dxf",
             strip_text_entities(dxf_bytes), true_label)
            for _, dxf_bytes, true_label in samples]

def build_scenario_d(samples: list) -> list:
    """Adversarial: wrong label as filename, original text."""
    all_classes = [label for _, _, label in samples]
    return [(f"{random.choice(all_classes)}.dxf", dxf_bytes, true_label)
            for _, dxf_bytes, true_label in samples]
```

### 3.3 融合验收指标（L2）

| 场景 | 最低目标 | B5.1 目标 | 判定 |
|------|---------|---------|------|
| A: 有名+有文字 | **≥ 99%** | ≥ 99.5% | 必须 |
| B: 无名+有文字 | **≥ 93%** | ≥ 94% | B5.1 主目标 |
| C: 无名+无文字 | **≥ 90%** | ≥ 90% | 不退步 |
| D: 错误名+有文字 | **≥ 70%** | ≥ 75% | 鲁棒性 |

---

## 4. L3：系统级端到端验证

### 4.1 端到端测试流程

```python
# tests/integration/test_hybrid_end_to_end.py

def test_no_name_scenario(dxf_path: str, true_label: str):
    """Simulate customer uploading file without meaningful name."""
    clf = HybridClassifier()
    result = clf.predict(
        dxf_bytes=open(dxf_path, "rb").read(),
        filename="20240105_scan_001.dxf",  # 无意义文件名
    )
    assert result["label"] == true_label
    assert result["confidence"] > 0.5

def test_batch_inference(dxf_paths: list[str]):
    """Verify batch prediction works correctly."""
    clf = HybridClassifier()
    results = clf.predict_batch(dxf_paths, batch_size=10)
    assert len(results) == len(dxf_paths)
    correct = sum(r["label"] == label for r, (_, label) in zip(results, dxf_paths))
    assert correct / len(results) >= 0.90
```

### 4.2 临界样本测试

```python
# 必须通过的 5 个典型案例
CRITICAL_CASES = [
    ("data/samples/典型法兰.dxf",     "法兰"),
    ("data/samples/典型轴承座.dxf",   "轴承座"),
    ("data/samples/典型阀门.dxf",     "阀门"),
    ("data/samples/大型箱体.dxf",     "箱体"),
    ("data/samples/细长轴.dxf",       "轴类"),
]
```

### 4.3 边界条件测试

```python
# 空文件 → 返回 "未知" 或最低置信度类别，不崩溃
# 大文件（>10MB DXF）→ 正常推理，延迟 < 500ms
# 损坏文件 → 捕获异常，返回 fallback 结果
# 无节点图（0 实体）→ 不崩溃
```

---

## 5. L4：性能验证

### 5.1 基准测试脚本

```python
# scripts/benchmark_inference.py
import time, statistics

def benchmark_single(clf, dxf_path: str, n_runs: int = 100) -> dict:
    latencies = []
    with open(dxf_path, "rb") as f:
        dxf_bytes = f.read()
    
    for _ in range(n_runs):
        t0 = time.perf_counter()
        clf.predict(dxf_bytes=dxf_bytes, filename="test.dxf")
        latencies.append((time.perf_counter() - t0) * 1000)
    
    return {
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * n_runs)],
        "p99_ms": sorted(latencies)[int(0.99 * n_runs)],
        "mean_ms": statistics.mean(latencies),
    }
```

### 5.2 性能验收标准

| 指标 | 目标 | 测量条件 |
|------|------|---------|
| P50 延迟（缓存命中） | **< 50ms** | 文件已缓存 |
| P50 延迟（无缓存） | **< 150ms** | 冷启动 |
| P99 延迟 | **< 300ms** | 99th percentile |
| 内存峰值 | **< 500MB** | 单进程 |
| 并发吞吐量 | **≥ 10 req/s** | 4 worker 进程 |

### 5.3 性能优化优先级

1. **已完成**：DXF→Graph 缓存（247× 加速，B4.1）
2. **B5.2 候选**：模型 INT8 量化（预期 -40% 延迟）
3. **B5.2 候选**：Graph2D + 文字提取并行化（async gather）
4. **B5.2 候选**：LRU 缓存（复用同一 DXF 的推理结果）

---

## 6. L5：回归验证（Superpass Gate）

### 6.1 现有测试套件

```bash
# 所有现有测试必须全部通过
python3 -m pytest tests/ -v --tb=short

# 关键回归检查
python3 -m pytest tests/unit/ -k "hybrid" -v
python3 -m pytest tests/integration/ -v
```

### 6.2 新增 CI 检查

```yaml
# .github/workflows/ml_regression.yml 新增
- name: Graph2D Model Regression
  run: |
    python3 scripts/evaluate_graph2d_v2.py \
      --model models/graph2d_finetuned_24class_v3.pth \
      --manifest data/graph_cache/cache_manifest.csv \
      --seed 42 | python3 scripts/check_regression_thresholds.py \
      --min-acc 0.91 --min-top3 0.99 --min-f1 0.90
```

### 6.3 回归检查清单

- [ ] 有文件名场景 hybrid acc ≥ 99%
- [ ] 无文件名场景 hybrid acc ≥ 90%
- [ ] 所有 24 类均有非零 recall
- [ ] 推理延迟 P50 < 150ms
- [ ] 内存 < 500MB
- [ ] `pytest tests/` 全绿

---

## 7. 验证执行顺序

```
阶段 1（B5.0 训练完成后）:
  → 运行 L1 评估（model-level）
  → 若达标，更新 hybrid_config.py 模型路径到 v3
  → 运行 L5 回归（Superpass gate）

阶段 2（B5.1 实现后）:
  → 构建场景测试集 A/B/C/D
  → 运行 L2 融合评估
  → 运行 L3 端到端测试

阶段 3（B5.2 优化后）:
  → 运行 L4 性能基准
  → 最终 L5 回归验证
```

---

## 8. 快速检查命令（日常使用）

```bash
# 快速模型健康检查（30s 内完成）
python3 -c "
import os, torch
os.environ['GRAPH2D_MODEL_PATH'] = 'models/graph2d_finetuned_24class_v3.pth'
from src.ml.vision_2d import Graph2DClassifier
clf = Graph2DClassifier()
print(f'Loaded: {clf._loaded}')
print(f'Classes: {len(clf.label_map)}')
print(f'Model type: {clf.model_type}')
"

# 快速精度快照（5min 内完成）
python3 scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --val-split 0.05    # 仅用 5% 做快速估计

# 完整回归（15min）
python3 -m pytest tests/ -v && \
python3 scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache/cache_manifest.csv
```

---

## 9. 里程碑追踪表

| 里程碑 | 验证内容 | 目标值 | 实际值 | 状态 |
|--------|---------|--------|--------|------|
| B4.4 | Graph2D acc | ≥ 85% | **90.5%** | ✓ |
| B4.5 | 无名 hybrid acc | ≥ 70% | **90.5%** | ✓ |
| **B5.0** | 整体 acc（原始集） | ≥ 92% | **91.0%** | ⚠️ 接近目标 |
| **B5.0** | 轴承座 recall | ≥ 80% | **100%** | ✓ 超出 |
| B5.1 | 无名+文字 acc | ≥ 93% | — | 待实现 |
| B5.2 | P50 延迟 | < 100ms | — | 待实现 |
| B5.3 | 监控上线 | 生产就绪 | — | 待实现 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
