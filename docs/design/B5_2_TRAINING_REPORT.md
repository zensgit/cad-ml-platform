# B5.2 实施报告：推理性能优化（文字缓存 + INT8 量化）

**日期**: 2026-04-14  
**阶段**: B5.2 — 推理延迟优化 + 生产就绪  
**基线**: B5.1 — 三路融合 avg=94.1%，推理延迟估计 ~133ms（冷启动）  
**目标**: P50 延迟 < 150ms，INT8 精度损失 < 0.5pp

---

## 1. 实施概要

| 任务 | 状态 | 说明 |
|------|------|------|
| `TextContentConfig` 新增至 `hybrid_config.py` | ✓ 已实现 | B5.1 最优权重 txt_w=0.10，环境变量支持 |
| `Graph2DConfig.fusion_weight` 调整 | ✓ 已实现 | 0.50 → 0.35（三路融合最优配置） |
| `preprocess_dxf_to_graphs.py` 文字缓存（B5.2a） | ✓ 已实现 | 单次 ezdxf 读取同时提取图+文字，节省 ~15ms |
| `scripts/quantize_graph2d_model.py`（B5.2b） | ✓ 已实现 | INT8 动态量化，预期 -40% 延迟，-74% 体积 |

---

## 2. 技术实现

### 2.1 TextContentConfig（hybrid_config.py）

```python
@dataclass
class TextContentConfig:
    """DXF 文字内容分类器配置（B5.1 新增）"""
    enabled: bool = True
    fusion_weight: float = 0.10  # B5.1 网格搜索最优
    min_text_len: int = 4        # 最短有效文字长度
```

**三路融合最优权重**（B5.1 64 组合搜索结果）：

| 分量 | 配置字段 | B4.5 值 | B5.1 值 | 变化原因 |
|------|---------|--------|--------|---------|
| 文件名 | `filename.fusion_weight` | 0.45 | 0.45 | 不变 |
| Graph2D | `graph2d.fusion_weight` | 0.50 | **0.35** | 引入文字分量后重新分配 |
| 文字内容 | `text_content.fusion_weight` | — | **0.10** | 新增（低权重避免稀疏噪声） |

**环境变量支持**：
```bash
export TEXT_CONTENT_ENABLED=true
export TEXT_CONTENT_FUSION_WEIGHT=0.10
export TEXT_CONTENT_MIN_TEXT_LEN=4
```

---

### 2.2 文字内容缓存（preprocess_dxf_to_graphs.py，B5.2a）

**变更前（B5.1）**：
```
推理时序：ezdxf.readfile() → Graph 转换 → [另一次 ezdxf.readfile()] → 文字提取
                                                   ↑ 重复 I/O，浪费 ~15ms
```

**变更后（B5.2a）**：
```python
def convert_dxf(file_path, node_dim=19, edge_dim=7, with_text=True):
    doc = ezdxf.readfile(file_path)          # 单次读取
    x, edge_index, edge_attr = graph_from_doc(doc)   # 图特征
    text_content = _extract_from_doc(doc)    # 文字内容（同一 doc 对象）
    return x, edge_index, edge_attr, text_content

# 缓存结构新增 text 字段
torch.save({
    "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
    "label": label,
    "text": text_content,   # ← B5.2a 新增
}, cache_path)
```

**效果**：
- 单次 ezdxf 读取替代两次，消除重复 I/O
- 预处理时多出约 ~5ms/文件（文字提取），但推理时节省 ~15ms/样本（无需重读 DXF）
- `--no-text` 选项保留向后兼容性

---

### 2.3 INT8 动态量化（quantize_graph2d_model.py，B5.2b）

**量化策略**：Dynamic Quantization（不需要校准数据集）

```python
quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # 仅量化 Linear 层（GNN 的 SAGEConv 内部 Linear）
    dtype=torch.qint8,
)
```

**预期效果**（基于 PyTorch 动态量化典型表现）：

| 指标 | FP32 | INT8 | 变化 |
|------|------|------|------|
| 模型文件大小 | ~500 KB | ~130 KB | -74% |
| 推理延迟（P50） | ~15 ms | ~9 ms | -40% |
| 精度损失 | 基线 | < 0.5 pp | 可接受 |

**量化脚本功能**：
- `--benchmark`：合成图基准测试（50 节点 / 100 边），对比 FP32 vs INT8 延迟
- `--verify-manifest`：在真实缓存数据集上验证量化精度，检查损失是否 < 0.5pp
- 量化 checkpoint 包含 `"quantized": True` 和 `"quantization": "dynamic_int8"` 标记

**使用方式**：
```bash
# 量化
python scripts/quantize_graph2d_model.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --output models/graph2d_finetuned_24class_v3_int8.pth \
    --benchmark

# 验证精度
python scripts/quantize_graph2d_model.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --output models/graph2d_finetuned_24class_v3_int8.pth \
    --verify-manifest data/graph_cache/cache_manifest.csv \
    --verify-limit 500
```

---

## 3. 配置变更汇总

### 3.1 hybrid_config.py 变更

| 字段 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `graph2d.fusion_weight` | 0.50 | **0.35** | B5.1 三路融合权重搜索最优 |
| `text_content`（新）| — | 全新 dataclass | B5.1 文字融合正式接入 |
| `text_content.enabled` | — | `True` | 默认启用 |
| `text_content.fusion_weight` | — | `0.10` | 低权重避免稀疏噪声 |

### 3.2 preprocess_dxf_to_graphs.py 变更

| 变更 | 影响 |
|------|------|
| `convert_dxf()` 返回 4-tuple（含 `text_content`） | 需重新预处理缓存（或追加 text 字段） |
| `.pt` 文件新增 `"text"` 字段 | 向前兼容：`CachedGraphDataset.get()` 应容错读取 |
| `--no-text` 参数 | 无文字需求时可跳过文字提取，加快预处理 |

---

## 4. 下一步：B5.2 待执行操作

### 4.1 重新预处理缓存（可选，增量）
```bash
# 重新预处理以生成含文字字段的缓存（--force 覆盖现有）
python scripts/preprocess_dxf_to_graphs.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output-dir data/graph_cache \
    --force

# 验证新缓存含 text 字段
python -c "
import torch
d = torch.load('data/graph_cache/$(ls data/graph_cache/*.pt | head -1 | xargs basename)')
print('text' in d, repr(d.get('text', '')[:50]))
"
```

### 4.2 执行量化
```bash
python scripts/quantize_graph2d_model.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --output models/graph2d_finetuned_24class_v3_int8.pth \
    --benchmark \
    --verify-manifest data/graph_cache/cache_manifest.csv
```

### 4.3 运行基准测试
```bash
python scripts/benchmark_inference.py \
    --manifest data/graph_cache/cache_manifest.csv \
    --model models/graph2d_finetuned_24class_v3_int8.pth
```

---

## 5. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/ml/hybrid_config.py` | ✓ 已修改 | TextContentConfig 新增，g2d 权重调整为 0.35 |
| `scripts/preprocess_dxf_to_graphs.py` | ✓ 已修改 | 单次 ezdxf 读取同时缓存文字内容 |
| `scripts/quantize_graph2d_model.py` | ✓ 已创建 | INT8 动态量化脚本（含基准测试 + 精度验证） |
| `scripts/benchmark_inference.py` | 待创建 | B5.2 验收基准测试脚本 |
| `src/ml/monitoring.py` | 待创建 | B5.3 PredictionMonitor 置信度漂移检测 |

---

## 6. B5.2 验收标准

| 指标 | 目标 | 状态 |
|------|------|------|
| `TextContentConfig` 上线 | 配置代码完整 | ✓ 完成 |
| `graph2d.fusion_weight` 调整 | 0.35（三路最优） | ✓ 完成 |
| 文字缓存单次 ezdxf 读取 | 代码实现 | ✓ 完成 |
| INT8 量化脚本 | 可执行，含验证 | ✓ 完成 |
| INT8 精度损失 | < 0.5 pp | 待运行验证 |
| P50 推理延迟（有缓存） | < 50ms | 待基准测试 |
| P50 推理延迟（冷启动） | < 150ms | 待基准测试 |

---

## 7. 里程碑追踪

| 里程碑 | 内容 | 目标 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + 模型 v3 | 91% | ✓ 91.0% |
| B5.1 | 文字融合（三路） | 93% avg | ✓ 94.1% avg |
| **B5.2a** | **文字缓存集成** | **单次 I/O** | **✓ 代码完成** |
| **B5.2b** | **INT8 量化** | **< 0.5pp 损失** | **待量化执行** |
| B5.3 | 监控上线 | 生产就绪 | 待实施 |

---

*报告生成: 2026-04-14*
