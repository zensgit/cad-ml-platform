# CAD ML Platform — 快速上手指南

## 启动服务

```bash
# Docker 部署（推荐）
docker-compose up -d

# 或本地运行
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

健康检查：
```bash
curl http://localhost:8000/health
```

---

## API 使用示例

### 1. 制造成本估算

```bash
curl -X POST http://localhost:8000/v1/cost/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "material": "steel",
    "batch_size": 100,
    "bounding_volume_mm3": 10000,
    "entity_count": 20,
    "tolerance_grade": "IT8",
    "surface_finish": "Ra3.2"
  }'
```

返回：
```json
{
  "estimate": {"material_cost": 0.63, "machining_cost": 155.33, "setup_cost": 2.0, "overhead": 23.69, "total": 181.65},
  "optimistic": {"total": 145.32},
  "pessimistic": {"total": 236.15},
  "process_route": ["cnc_lathe"],
  "complexity_score": 3.2,
  "confidence": 0.6,
  "reasoning": ["材料费基于体积×密度×单价", "加工费基于复杂度和机时"]
}
```

### 2. 图纸差异对比

```bash
curl -X POST http://localhost:8000/v1/diff/compare \
  -F "file_a=@drawing_v1.dxf" \
  -F "file_b=@drawing_v2.dxf"
```

### 3. ECN 工程变更通知

```bash
curl -X POST http://localhost:8000/v1/diff/ecn \
  -F "file_a=@drawing_v1.dxf" \
  -F "file_b=@drawing_v2.dxf" \
  -F "part_number=FL-2024-001" \
  -F "revision=B"
```

### 4. 3D 点云分类

```bash
curl -X POST http://localhost:8000/v1/pointcloud/classify \
  -F "file=@part.stl"
```

支持格式：STL, OBJ, PLY, XYZ

### 5. 知识图谱查询

```python
import requests

# 自然语言查询
r = requests.post("http://localhost:8000/v1/assistant/ask", json={
    "question": "SUS304适合什么加工工艺？"
})
print(r.json()["answer"])
# → "SUS304不锈钢适合的加工工艺有：CNC车削、CNC铣削、5轴加工、线切割..."

# 最优工艺推荐
r = requests.post("http://localhost:8000/v1/assistant/ask", json={
    "question": "法兰盘用SUS304做，推荐什么工艺？"
})
```

### 6. AI Copilot 对话

```python
import requests

r = requests.post("http://localhost:8000/v1/assistant/ask", json={
    "question": "帮我分析这个零件的成本，材料是铝合金6061，批量500件，精度IT7"
})
# Copilot 会自动调用 classify → recommend_process → estimate_cost 工具
# 返回综合分析结果
```

### 7. 可用材料列表

```bash
curl http://localhost:8000/v1/cost/materials
```

### 8. 支持的 3D 格式

```bash
curl http://localhost:8000/v1/pointcloud/formats
# → [".stl", ".obj", ".ply", ".xyz"]
```

---

## 训练模型

```bash
# 生成 3D 合成训练数据
python scripts/generate_3d_training_data.py --samples 100

# 挖掘度量学习对
python scripts/mine_metric_pairs.py --data-dir data/training_v8

# 训练领域嵌入
python scripts/train_domain_embeddings.py --epochs 3

# 知识蒸馏
python scripts/train_knowledge_distillation.py --teacher models/cad_classifier_v15_ensemble.pt

# 训练 Graph2D（最优配置）
python scripts/train_2d_graph.py \
  --manifest data/training_v8/manifest_5class.csv \
  --dxf-dir data/training_v8 \
  --model edge_sage --hidden-dim 128 --epochs 100 \
  --early-stop-patience 20 --save-best --scheduler cosine --warmup-epochs 10 --augment
```

---

## 测试

```bash
# 全部新模块测试（236项）
make test-new-modules

# 按模块测试
make test-cost           # 成本估算
make test-diff           # 图纸对比
make test-pointcloud     # 3D点云
make test-knowledge      # 知识图谱
make test-copilot        # AI Copilot
make test-ai-intelligence # 反馈闭环+智能融合
make test-embeddings     # 领域嵌入

# 冒烟测试（验证所有模块可导入）
make smoke-new-modules

# 性能基准
python scripts/run_performance_baseline.py --output reports/performance_baseline.md
```

---

## 已训练模型

| 模型 | 文件 | 精度 |
|------|------|------|
| PointNet 3D 分类 | `models/pointnet_synthetic_v1.pth` | 98.3% val_acc |
| Graph2D 5类分类 | `models/graph2d_5class_sage_best.pth` | 39.5% val_acc |
| 度量学习嵌入 | `models/metric_learning_v2.pth` | loss 0.024 |
| 蒸馏分类器 | `models/cad_classifier_distilled.pt` | 10.6x 压缩 |
| 领域嵌入 | `models/embeddings/manufacturing_v2/` | Spearman 0.50 |
