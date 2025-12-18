# 2D 查重端到端验证（训练图纸集）- 2025-12-15

本次验证目标：用真实 DWG 数据集（已转为 `PNG + v2 JSON`）验证 **vision 召回 + L4 JSON 精查** 的端到端批量查重效果，并确认阈值可调可复现。

> 说明：当前 macOS 开发环境存在“无法监听端口”的限制（启动 `uvicorn/FastAPI` 会 `PermissionError: [Errno 1]`），因此本文同时给出：
>
> - **API 方式**：当服务可启动/可访问时使用（推荐用于真实端到端）。
> - **local_l4 方式**：无需启动服务，直接对 `v2 JSON` 进行全量矩阵验证（用于算法验证/阈值校准）。

## 1. 环境与数据

- `dedupcad-vision`：`http://127.0.0.1:58001`
- `cad-ml-platform`：`http://127.0.0.1:18000`
- 数据集（已生成工件）：`data/train_artifacts/`（109 张图：`*.png + *.v2.json`）
- 弱标签（按文件名版本组）：`data/train_drawings_manifest/expected_groups.json`

## 2. 执行步骤（可复现）

### 2.1 批量索引（入库 PNG + v2 JSON）

```bash
python3 scripts/dedup_2d_batch_index.py data/train_artifacts \
  --base-url http://127.0.0.1:18000 \
  --api-key test \
  --user-name batch \
  --require-json \
  --rebuild-index
```

### 2.2 严格重复（强一致）报告

用途：只把“几乎相同”的图纸判为 `duplicate`，偏向 **低误报**。

```bash
python3 scripts/dedup_2d_batch_search_report.py data/train_artifacts \
  --base-url http://127.0.0.1:18000 \
  --api-key test \
  --engine api \
  --no-index \
  --mode balanced \
  --max-results 200 \
  --top-k 50 \
  --precision-top-n 20 \
  --precision-visual-weight 0.3 \
  --precision-geom-weight 0.7 \
  --duplicate-threshold 0.95 \
  --similar-threshold 0.80 \
  --group-rule verdict \
  --output-dir data/dedup_report_train_api_18000_strict_095
```

输出：

- `data/dedup_report_train_api_18000_strict_095/matches.csv`
- `data/dedup_report_train_api_18000_strict_095/groups.json`
- `data/dedup_report_train_api_18000_strict_095/summary.json`

### 2.3 版本查重（同图不同版）报告

用途：把“同一图纸的不同版本”尽量聚到同一簇，偏向 **高召回**。

```bash
python3 scripts/dedup_2d_batch_search_report.py data/train_artifacts \
  --base-url http://127.0.0.1:18000 \
  --api-key test \
  --engine api \
  --no-index \
  --mode balanced \
  --max-results 200 \
  --top-k 50 \
  --precision-top-n 50 \
  --precision-visual-weight 0.3 \
  --precision-geom-weight 0.7 \
  --duplicate-threshold 0.95 \
  --similar-threshold 0.70 \
  --group-rule threshold \
  --group-threshold 0.70 \
  --output-dir data/dedup_report_train_api_18000_version_070
```

输出：

- `data/dedup_report_train_api_18000_version_070/matches.csv`
- `data/dedup_report_train_api_18000_version_070/groups.json`
- `data/dedup_report_train_api_18000_version_070/summary.json`

### 2.4 （离线）全量矩阵：local_l4 + version(profile=spatial)

用途：在不依赖 `dedupcad-vision/cad-ml-platform` 服务启动的情况下，对训练集做 **O(N²)** 的全量相似度矩阵计算，便于做阈值扫描与负样本分析。

```bash
python3 scripts/dedup_2d_batch_search_report.py data/train_artifacts \
  --engine local_l4 \
  --require-json \
  --precision-profile version \
  --top-k 108 \
  --min-similarity 0.0 \
  --group-rule verdict \
  --duplicate-threshold 0.95 \
  --similar-threshold 0.80 \
  --output-dir data/dedup_report_train_local_version_profile_spatial_full
```

生成可离线打开的 HTML 报告：

```bash
python3 scripts/dedup_2d_generate_html_report.py data/dedup_report_train_local_version_profile_spatial_full \
  --max-matches-rows 500
```

并打包为 ZIP（便于交付/转发）：

```bash
python3 scripts/dedup_2d_package_report.py data/dedup_report_train_local_version_profile_spatial_full --overwrite
```

## 3. 结果摘要（弱标签评估）

说明：弱标签来自文件名“版本组”（并非人工精标），因此指标仅用于 **阈值校准与趋势对比**。

### 3.1 严格重复（`duplicate_threshold=0.95`）

- 预测重复边（edges）：46
- 预测重复簇（groups）：37
- 对弱标签“版本对”的 pairwise 指标：
  - `precision = 1.000`
  - `recall ≈ 0.639`（严格口径不追求覆盖所有版本）

### 3.2 版本查重（`similar_threshold=0.70`，按 `similarity>=0.70` 成簇）

- 预测重复边（edges）：69
- 预测重复簇（groups）：46
- 对弱标签“版本对”的 pairwise 指标：
  - `precision = 1.000`
  - `recall ≈ 0.958`（FN=3）

未召回的 3 对（弱标签认为同版，但相似度不足）：

- `J0225054-15-01支承座v1 <-> J0225054-15-01支承座v2`
- `LTJ012306102-0084调节螺栓v1 <-> LTJ012306102-0084调节螺栓v2`
- `J1424042-51-01-08对接法兰v1 <-> J1424042-51-01-08对接法兰v2`

对应的 PNG 对比图（左=版本1，中=版本2，右=像素差异热力图）已生成：

- `data/dedup_debug_diffs_20251215/J0225054-15-01支承座v1__J0225054-15-01支承座v2__compare.png`
- `data/dedup_debug_diffs_20251215/LTJ012306102-0084调节螺栓v1__LTJ012306102-0084调节螺栓v2__compare.png`
- `data/dedup_debug_diffs_20251215/J1424042-51-01-08对接法兰v1__J1424042-51-01-08对接法兰v2__compare.png`

### 3.3 版本 profile（spatial 增强）阈值扫描（local_l4 全矩阵）

核心结论：**“版本聚类”仅靠几何相似度阈值很难做到同时高召回+低误报**，需要引入标题栏/图号/版本号等元数据做 gate（这是主流商业查重的关键差异点之一）。

运行阈值扫描：

```bash
python3 scripts/dedup_2d_threshold_scan_manifest.py \
  --matches-csv data/dedup_report_train_local_version_profile_spatial_full/matches.csv \
  --expected-groups-json data/train_drawings_manifest/expected_groups.json \
  --top-negatives 20
```

本次训练集（109 张图）统计（弱标签）：

- positives（同版本组）：72 对；negatives：5814 对
- `pos_min ≈ 0.69845`（最弱“同版”对）
- `neg_max ≈ 0.85211`（最强“不同图”对）
- 在该数据上：
  - 阈值 `0.86`：`precision=1.0`，`recall≈0.931`（0 FP，5 FN）
  - 阈值 `0.82`：`precision≈0.92`，`recall≈0.958`（6 FP，3 FN）

离线报告产物：

- HTML：`data/dedup_report_train_local_version_profile_spatial_full/index.html`
- ZIP：`data/dedup_report_train_local_version_profile_spatial_full_package.zip`

## 4. 建议

- 上线默认可先给两档预设：
  - **严格重复**：`duplicate_threshold=0.95`，`similar_threshold=0.80`
  - **版本查重**：`duplicate_threshold=0.95`，`similar_threshold=0.70`
- 对 FN 的 3 对建议人工抽检（它们的 v2 JSON 精查分数很低，可能是“版本差异极大”而非算法漏检）。

补充建议（落地对标主流）：

- “版本查重/同图不同版”建议叠加 **元数据 gate**（同图号/同标题栏主键）再用较低阈值（例如 `0.70`）聚类，避免跨零件误聚类。
