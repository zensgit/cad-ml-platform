# B5.5 提升计划：综合验收 + 增量训练 v4 + 生产部署

**日期**: 2026-04-14  
**基线**: B5.4 — TextContent 正式接入推理，关键词扩充，主动学习工具就绪  
**目标**:
  1. 无文件名场景精度 ≥ 93%（val set，原始 manifest）
  2. 四场景加权 avg ≥ 95%
  3. INT8 量化 v4 模型生产部署（v3_int8 完成验证后）

---

## 1. 状态盘点

### 1.1 当前精度基线（B5.4 前）

| 场景 | 精度 | 样本量 |
|------|------|--------|
| A: 有名有文字 | ~100% | val set × 87% |
| B: 无名有文字 | 90.9%（评估） | val set × 87% |
| C: 无名无文字 | **91.0%**（v3 原始 manifest） | val set 全量 |
| D: 有名无文字 | ~99% | val set 全量 |
| 综合 avg | 94.1% | 加权 |

### 1.2 B5.4 后预期改善点

| 组件 | 改善预期 | 原因 |
|------|---------|------|
| 法兰关键词命中率 | 5% → **30-40%** | 标准号变体（NB/T 47023 等）大量新增 |
| 箱体关键词命中率 | 25% → **35-45%** | 箱体特有功能孔关键词新增 |
| 轴类关键词命中率 | 15% → **30-40%** | 加工特征关键词新增 |
| 综合文字命中率 | ~47% → **~60%** | 主类命中率全面提升 |
| 场景 B 精度 | 90.9% → **~91.5-92%** | 文字信号更多参与有效融合 |

---

## 2. B5.5a：精度验收测试

### 2.1 执行文字覆盖率再审计

重跑 audit_text_coverage.py，验证 B5.4c 关键词扩充效果：

```bash
python scripts/audit_text_coverage.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output docs/design/B5_5_TEXT_AUDIT.md

# 预期目标
# 法兰：  命中率 5%  → ≥ 30%
# 箱体：  命中率 25% → ≥ 35%
# 轴类：  命中率 15% → ≥ 30%
# 整体：  命中率 47% → ≥ 58%
```

### 2.2 执行四场景精度评估

```bash
# 场景 C：纯 Graph2D（基线确认）
python scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache/cache_manifest.csv

# 场景 B：无名有文字（TextContent 接入后）
# 运行 search_hybrid_weights_v2.py 四场景评估
python scripts/search_hybrid_weights_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output docs/design/B5_5_WEIGHT_SEARCH.md
```

### 2.3 验收决策树

```
四场景 avg ≥ 95%?
  ├── 是 → 进入 B5.5b（v4 增量训练）
  └── 否 →
        场景 B < 91%?
          ├── 是 → 检查文字命中率，优先扩充低命中类别
          └── 否 →
                法兰 prec < 70%?
                  ├── 是 → 检查法兰关键词，分析 FP 来源
                  └── 否 → 考虑提升 txt_w 至 0.15
```

---

## 3. B5.5b：增量训练 v4

### 3.1 数据准备

**来源 1：低置信度审核样本**（主动学习）

```bash
# 查看审核进度
python3 -c "
from src.ml.low_conf_queue import LowConfidenceQueue
q = LowConfidenceQueue()
print(f'总计: {q.size()}，待审核: {q.pending_review()}')
reviewed = q.reviewed_entries()
print(f'已审核: {len(reviewed)}，可用于训练')
"

# 追加到 manifest
python scripts/append_reviewed_to_manifest.py \
    --queue data/review_queue/low_conf.csv \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output data/manifests/unified_manifest_v3.csv \
    --corrections-only \
    --dxf-roots /data/dxf_archive/
```

**来源 2：针对性增强（轴承座 precision 修复）**

轴承座当前 precision=43%（recall 100%，大量假阳性）。针对性策略：

```bash
# 分析轴承座假阳性来源（混淆矩阵中被误预测为轴承座的类别）
python scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --show-confusion
# 预期：箱体被误测为轴承座 → 增加更多箱体/轴承座对比样本
```

### 3.2 v4 训练命令

```bash
# 基于 v3 fine-tune（低 encoder-lr 防过拟合）
python scripts/finetune_graph2d_v2_augmented.py \
    --checkpoint models/graph2d_finetuned_24class_v3.pth \
    --manifest data/manifests/unified_manifest_v3.csv \
    --output models/graph2d_finetuned_24class_v4.pth \
    --epochs 40 \
    --batch-size 32 \
    --encoder-lr 2e-5 \
    --head-lr 2e-4 \
    --focal-gamma 1.5 \
    --patience 10 \
    --device cpu
```

### 3.3 v4 验证标准

| 指标 | 目标 | 说明 |
|------|------|------|
| 整体 val acc（原始 manifest） | ≥ **91.0%** | 不低于 v3 基线 |
| 轴承座 precision | ≥ **60%** | v3=43%，需要改善 |
| 轴承座 recall | ≥ **90%** | 保持高召回 |
| 法兰 recall | ≥ **95%** | v3=100%，需保持 |
| 阀门 recall | ≥ **90%** | v3=100%，需保持 |
| Top-3 acc | ≥ **99%** | 维持现有水平 |

---

## 4. B5.5c：INT8 量化生产化

### 4.1 量化执行

```bash
# v3 量化（B5.2b 延迟任务）
python scripts/quantize_graph2d_model.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --output models/graph2d_finetuned_24class_v3_int8.pth \
    --benchmark \
    --verify-manifest data/graph_cache/cache_manifest.csv \
    --verify-limit 500

# 验证通过后，量化 v4
python scripts/quantize_graph2d_model.py \
    --model models/graph2d_finetuned_24class_v4.pth \
    --output models/graph2d_finetuned_24class_v4_int8.pth \
    --benchmark \
    --verify-manifest data/graph_cache/cache_manifest.csv
```

### 4.2 量化验收标准

| 指标 | 目标 |
|------|------|
| 精度损失 | < **0.5 pp** |
| 模型大小 | < **150 KB**（原始 ~500KB） |
| P50 推理延迟 | < **10 ms**（图结构推理部分） |

### 4.3 生产部署配置

```bash
# 切换到量化 v4 模型
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth

# 启用文字内容融合
export TEXT_CONTENT_ENABLED=true
export TEXT_CONTENT_FUSION_WEIGHT=0.10

# 监控配置
export MONITOR_WINDOW_SIZE=1000
export LOW_CONF_QUEUE_PATH=data/review_queue/low_conf.csv
export LOW_CONF_QUEUE_THRESHOLD=0.50
```

---

## 5. B5.5d：全链路基准测试

### 5.1 推理延迟拆分

B5.4 后，完整推理链路为：

```
DXF bytes 输入
  ├── 文件名分类           ~1ms
  ├── ezdxf 解析           ~20ms（冷启动，无缓存）
  ├── Graph 转换           ~80ms（ezdxf 主要开销）
  ├── 文字提取             ~15ms（B5.2a：与 Graph 同一 doc，但 classify() 中仍需单独解析）
  ├── Graph2D 推理         ~9ms（INT8 量化后）
  ├── TextContent 关键词   ~1ms
  ├── 三路融合             ~1ms
  └── 监控记录             ~0.1ms
总计（冷启动）:            ~127ms
```

**B5.2a 注意**：`classify()` 中仍调用 `extract_text_from_bytes(file_bytes)`，这是一次独立的 ezdxf 解析。生产优化可在 Graph2D 预测阶段解析一次 doc 对象，共享给文字提取。当前先以正确性为优先，性能优化留 B5.6。

### 5.2 基准测试脚本（创建）

```bash
# 创建 scripts/benchmark_inference.py（B5.5d 新建）
python scripts/benchmark_inference.py \
    --manifest data/graph_cache/cache_manifest.csv \
    --model models/graph2d_finetuned_24class_v4_int8.pth \
    --n-files 50 --n-runs 10

# 目标
# P50 < 100ms（有缓存命中时）
# P50 < 150ms（纯冷启动）
# P95 < 300ms
```

---

## 6. 验证计划（B5 总体）

### L1: 单元测试（已通过）

```bash
python3 -m pytest tests/unit/test_monitoring.py tests/unit/test_low_conf_queue.py -v
# 54/54 通过 ✓
```

### L2: 组件测试

```bash
# 关键词扩充效果验证
python3 -c "
from src.ml.text_classifier import TextContentClassifier
clf = TextContentClassifier()
# 法兰：NB/T 标准号命中
assert clf.top_class('NB/T 47023 RF面 PN40') == '法兰'
# 轴类：加工特征命中
assert clf.top_class('轴颈 中心孔 调质处理') == '轴类'
# 箱体：功能孔命中，无轴承座交叉
assert clf.top_class('箱盖 窥视孔 放油孔') == '箱体'
# 换热器：高置信命中
assert clf.top_class('管板 折流板 壳程 管程') == '换热器'
print('ALL OK')
"

# TextContent 推理接入验证
python3 -c "
from src.ml.hybrid_classifier import HybridClassifier, DecisionSource
clf = HybridClassifier()
# DecisionSource 包含 TEXT_CONTENT
assert DecisionSource.TEXT_CONTENT == 'text_content'
# text_content_weight 已从 config 读取
assert clf.text_content_weight == 0.10
# fusion_weights 包含 text_content
from src.ml.hybrid_classifier import ClassificationResult
r = ClassificationResult()
assert r.text_content_prediction is None
print('ALL OK')
"
```

### L3: 集成测试（B5.5a 再审计）

```bash
python scripts/audit_text_coverage.py \
    --manifest data/manifests/unified_manifest_v2.csv
# 验证：法兰命中率 ≥ 30%，箱体 ≥ 35%，轴类 ≥ 30%
```

### L4: 系统测试

```bash
# 四场景精度评估
python scripts/search_hybrid_weights_v2.py
# 验证：场景 B ≥ 91.5%，综合 avg ≥ 95%
```

### L5: 生产验证

```bash
# 监控摘要（100 次推理后）
python3 -c "
clf = HybridClassifier()
# 跑若干样本...
print(clf.monitor.summary())
# 验证：text_hit_rate > 0.10（文字信号实际命中）
"
```

---

## 7. 实施步骤

```
Week 1 (B5.5a): 精度验收
  → 重跑 audit_text_coverage.py（B5.4c 效果验证）
  → 重跑 search_hybrid_weights_v2.py（B5.4a 效果验证）
  → 验收决策：avg ≥ 95% → 推进；否则 → 调参

Week 2-3 (B5.5b): 增量训练
  → 积累审核样本（目标 ≥ 200 条）
  → 运行 append_reviewed_to_manifest.py
  → fine-tune v3 → v4（40 epochs）
  → 评估 v4：acc ≥ 91%，轴承座 prec ≥ 60%

Week 4 (B5.5c): 量化
  → 量化 v3（验证精度损失 < 0.5pp）
  → 量化 v4
  → 基准测试（P50 < 100ms）

Week 5 (B5.5d): 全链路测试 + 生产部署
  → 运行 benchmark_inference.py
  → 切换生产环境到 v4_int8
  → 监控观察 48h（text_hit_rate, low_conf_rate, avg_confidence）
```

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 法兰关键词扩充引入新误判 | 中 | 中 | audit 后检查 FP 来源；PN16/PN25 在过滤器/阀门中也常见 |
| v4 增量训练过拟合审核小样本 | 中 | 高 | encoder-lr=2e-5（极低），patience=10，FocalLoss 正则 |
| INT8 量化 GNN 精度损失 > 0.5pp | 低 | 高 | dynamic quantization 仅量化 Linear，GNN graph conv 保持 fp32 |
| TextContent 并发 extract_text 延迟 | 中 | 低 | B5.6 实现 doc 对象共享（classify 中一次 ezdxf 读取） |
| 监控 text_hit_rate 始终为 0 | 低 | 低 | 检查 file_bytes 是否为 None，TEXT_CONTENT_ENABLED 是否为 true |

---

## 9. 里程碑追踪

| 里程碑 | 内容 | 目标 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + 模型 v3 | 91.0% | ✓ |
| B5.1 | 文字融合评估 | 94.1% avg | ✓ |
| B5.2 | 性能优化（缓存+量化脚本） | 单次 I/O | ✓ |
| B5.3 | 监控上线 + 54/54 测试 | 生产监控 | ✓ |
| B5.4 | TextContent 推理集成 + 关键词扩充 | text_hit 有效 | ✓ |
| **B5.5a** | **精度验收测试（再审计）** | **avg ≥ 95%** | 待执行 |
| **B5.5b** | **v4 增量训练** | **acc ≥ 91%** | 待数据积累 |
| **B5.5c** | **INT8 量化 v4** | **< 0.5pp 损失** | 待 v4 完成 |
| **B5.5d** | **全链路基准测试 + 生产部署** | **P50 < 100ms** | 待验收 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
