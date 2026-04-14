# Training Data Governance 验证报告

**日期**: 2026-04-14  
**实施范围**: Phase 1（Provenance Schema + Fail-Closed）+ Phase 2（Golden Validation Set）  
**验证结果**: **15/15 检查通过，54/54 回归测试通过**

---

## 1. Batch 1 — Provenance Schema 验证

### 1.1 ActiveLearningSample 新增字段

| 字段 | 默认值 | 验证 |
|------|--------|------|
| `sample_source` | `"unknown"` | ✓ |
| `label_source` | `"unknown"` | ✓ |
| `review_source` | `None` | ✓ |
| `human_verified` | `False` | ✓ |
| `verified_by` | `None` | ✓ |
| `verified_at` | `None` | ✓ |
| `eligible_for_training` | `False` | ✓ |
| `training_block_reason` | `"missing_provenance"` | ✓ |

**关键行为验证**：
- `flag_for_review()` → `sample_source="analysis_review_queue"`, `label_source="model_auto"`
- `submit_feedback(label_source="human_feedback")` → `human_verified=True`, `eligible_for_training=True`
- `export_training_data()` → 默认只导出 `eligible_for_training=True`

### 1.2 LowConfidenceQueue 扩展

| 新增列 | 默认值 | 验证 |
|--------|--------|------|
| `sample_source` | `"legacy_low_conf_queue"` | ✓ |
| `label_source` | `""` | ✓ |
| `human_verified` | `""` | ✓ |
| `eligible_for_training` | `""` | ✓ |

新增 `human_verified_entries()` 方法：仅返回 `human_verified` 为 truthy 的行。

### 1.3 Feedback API 扩展

`FeedbackRequest` 新增可选字段：`label_source`, `review_source`, `verified_by`。
POST 端点默认 `label_source="human_feedback"`。

### 1.4 Active Learning API 扩展

`FeedbackRequest` 新增可选字段，透传至 `submit_feedback()`。

---

## 2. Batch 2 — Fail-Closed Training 验证

### 2.1 finetune_from_feedback.py

| 行为 | 旧 | 新 | 验证 |
|------|-----|-----|------|
| 无向量时 | 退回 mock 随机数据 | **SystemExit(1)** | ✓ |
| 开发/测试 | — | `--allow-mock` 显式开关 | ✓ |

### 2.2 train_knowledge_distillation.py

| 行为 | 旧 | 新 | 验证 |
|------|-----|-----|------|
| 无教师模型 | 静默用随机权重 | **SystemExit(1)** | ✓ |
| synthetic 数据 | 默认生成 | 仅 `--demo` 时允许 | ✓ |

### 2.3 append_reviewed_to_manifest.py

| 行为 | 旧 | 新 | 验证 |
|------|-----|-----|------|
| 导出筛选 | reviewed_label 非空即可 | **human_verified=true 必须** | ✓ |
| 迁移模式 | — | `--include-unverified` 显式开关 | ✓ |
| 日志 | 无统计 | "Exported X verified, blocked Y unverified" | ✓ |

### 2.4 auto_retrain.sh

| 行为 | 旧 | 新 | 验证 |
|------|-----|-----|------|
| 门控依据 | reviewed_entries 总数 | **human_verified 总数** | ✓ |
| Step 1b | 不存在 | 新增 provenance 计数检查 | ✓ |
| 日志 | 仅"Reviewed: N" | "N total, M human-verified, K eligible" | ✓ |

---

## 3. Batch 3 — Golden Validation Set 验证

### 3.1 数据集

| 文件 | 样本数 | 验证 |
|------|--------|------|
| `data/manifests/golden_val_set.csv` | **914** | ✓ |
| `data/manifests/golden_train_set.csv` | **3660** | ✓ |
| 重叠检查 | **0** | ✓ |
| 总计 | 4574（= 原始 manifest） | ✓ |

### 3.2 脚本支持

| 脚本 | 新参数 | 验证 |
|------|--------|------|
| `evaluate_graph2d_v2.py` | `--golden-val-manifest` | ✓ 编译通过 |
| `finetune_graph2d_v2_augmented.py` | `--val-manifest` | ✓ 编译通过 |
| `auto_retrain.sh` | `GOLDEN_VAL` 环境变量 | ✓ |

### 3.3 auto_retrain.sh 评估路径修复

| 行为 | 旧 | 新 |
|------|-----|-----|
| 评估数据集 | `data/graph_cache/cache_manifest.csv`（与训练重叠） | **`data/manifests/golden_val_set.csv`**（独立集合） |
| 训练时 val | random_split | `--val-manifest golden_val_set.csv` |

---

## 4. 回归测试

```
tests/unit/test_monitoring.py:    28 passed
tests/unit/test_low_conf_queue.py: 26 passed
总计: 54/54 passed ✓
```

---

## 5. 变更文件清单

### Batch 1（Provenance Schema）

| 文件 | 变更 |
|------|------|
| `src/core/active_learning.py` | ActiveLearningSample +8 字段, flag_for_review/submit_feedback/export_training_data 更新 |
| `src/api/v1/active_learning.py` | FeedbackRequest +3 字段, 透传 submit_feedback |
| `src/api/v1/feedback.py` | FeedbackRequest +3 字段, POST 端点写入 JSONL |
| `src/ml/low_conf_queue.py` | _FIELDNAMES +4 列, maybe_enqueue/human_verified_entries |

### Batch 2（Fail-Closed Training）

| 文件 | 变更 |
|------|------|
| `scripts/finetune_from_feedback.py` | load_training_data fail-closed + --allow-mock |
| `scripts/train_knowledge_distillation.py` | --demo 显式开关, 生产路径 sys.exit(1) |
| `scripts/append_reviewed_to_manifest.py` | human_verified 筛选 + --include-unverified + 日志统计 |
| `scripts/auto_retrain.sh` | Step 1b provenance 检查, human_verified 门控 |

### Batch 3（Golden Validation Set）

| 文件 | 变更 |
|------|------|
| `data/manifests/golden_val_set.csv` | 新建（914 样本） |
| `data/manifests/golden_train_set.csv` | 新建（3660 样本） |
| `scripts/evaluate_graph2d_v2.py` | --golden-val-manifest 参数 |
| `scripts/finetune_graph2d_v2_augmented.py` | --val-manifest 参数 |
| `scripts/auto_retrain.sh` | GOLDEN_VAL 环境变量, 评估改用 golden set |

---

## 6. 使用示例

### 生产训练（fail-closed）

```bash
# 自动重训（使用 golden val 评估 + provenance 门控）
GOLDEN_VAL=data/manifests/golden_val_set.csv bash scripts/auto_retrain.sh

# 手动评估（golden val）
python scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v4.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --golden-val-manifest data/manifests/golden_val_set.csv

# 手动训练（golden val）
python scripts/finetune_graph2d_v2_augmented.py \
    --manifest data/graph_cache/cache_manifest.csv \
    --val-manifest data/manifests/golden_val_set.csv \
    --output models/graph2d_v_new.pth
```

### 开发/测试（显式开关）

```bash
# 允许 mock 数据（仅测试）
python scripts/finetune_from_feedback.py --allow-mock

# demo 模式蒸馏
python scripts/train_knowledge_distillation.py --demo

# 迁移模式（含未验证样本）
python scripts/append_reviewed_to_manifest.py --include-unverified ...
```

---

## 7. 总结

| 检查项 | 结果 |
|--------|------|
| Provenance 字段完整 | ✓ 8 个字段，安全默认值 |
| Fail-closed 生效 | ✓ 3 个脚本已改为默认失败 |
| 显式开关存在 | ✓ --allow-mock / --demo / --include-unverified |
| Golden val 建立 | ✓ 914 样本，0 重叠 |
| 脚本接口支持 | ✓ --golden-val-manifest / --val-manifest |
| auto_retrain 路径修复 | ✓ 评估改用 golden set |
| 回归测试 | ✓ 54/54 通过 |

**Phase 1 + Phase 2 全部落地完成。**

---

*验证报告生成: 2026-04-14*
