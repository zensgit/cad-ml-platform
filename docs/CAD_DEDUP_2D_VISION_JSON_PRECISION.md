# CAD 2D 查重：视觉召回 + JSON 精查（对标主流路线）

> 目标：在保持“秒级召回”的同时，引入 **v2 JSON 几何精查**，把 2D 查重能力拉齐到主流商业查重软件的可用水平；3D 为后续阶段。

## 1. 背景与目标

- **主要输入格式**：DWG（用户侧为主），服务侧优先处理插件上传的 `PNG + v2 JSON`。
- **系统现状**：
  - `dedupcad-vision`：以图像为中心的 2D 查重（pHash + 视觉特征 + FAISS），具备渐进式召回与可选 diff。
  - `dedupcad`：以 **CAD 几何/语义 JSON（v2）** 为中心的精确相似度/差异计算（`weighted_similarity`、JSON diff、DXF 提取）。
  - `cad-ml-platform`：统一 API/鉴权/编排入口，可作为对外统一网关并代理 `dedupcad-vision`。

## 2. 与主流 CAD 查重的差距（现状评估）

### 2.1 现有 `dedupcad-vision` 的强项

- L1 pHash 快速过滤：适合大规模库的粗筛。
- L2 视觉特征 + FAISS：召回速度快，工程实现简单稳定。
- 渐进式 pipeline：天然适合“先快后准”的多层查重框架。

### 2.2 主要差距（为什么还无法对标主流）

主流商业查重通常不是纯视觉，而是“**视觉/栅格 + 结构/几何 + 语义（文字/标注/块）**”的融合体系。当前差距集中在：

- **几何语义缺失**：图像相似 ≠ CAD 相似；块、图层、标注、文字、尺寸等在主流产品中权重很高。
- **鲁棒性不足**：打印样式、线宽、比例尺、旋转、局部裁剪、视口差异等会导致纯视觉误判/漏判。
- **可解释性差**：主流产品往往能输出结构化差异（哪些层/块/标注不同），用于工程复核。
- **精排缺位**：缺少“对 Top-N 候选做几何级验证”的精排层，导致最终 Top-10 仍可能混入“看起来像但几何不同”的结果。

**结论**：要对标主流，必须增加 **L4 Precision（几何精查层）**，并逐步引入更强的 L3 表征（深度 embedding/结构特征）。

## 3. 总体架构（2D 对标版）

```mermaid
graph TD
    User[客户端/插件] -->|上传 PNG + v2 JSON| Platform[cad-ml-platform]
    User -->|批量上传 DXF/DWG(可选)| Platform

    Platform -->|视觉召回| VisionAPI[DedupCAD-Vision API]
    VisionAPI -->|Top-K 候选| Platform

    subgraph "cad-ml-platform (Orchestrator)"
        Platform --> L4[L4: Precision Layer (v2 JSON/几何精查)]
        L4 --> Final[最终 Top-10 + 差异报告]

        subgraph "Precision Core (vendored from dedupcad)"
            VScoring[scoring.py / weighted_similarity]
            VMatch[entities_match.py]
            VNorm[v2_normalize.py]
            VDiff[json_diff.py]
        end

        L4 --> VScoring
    end

    subgraph "Storage"
        DB[(Metadata DB)]
        OBJ[(Object Store)]
        VisionAPI -->|存 PNG/特征| OBJ
        Platform -->|存 v2 JSON| OBJ
        DB -->|读取 JSON 路径| L4
    end
```

## 4. API 约定（对插件最友好）

> MVP 优先支持插件场景：**PNG + v2 JSON**。DXF/DWG 的服务端兜底可分期实现。

### 4.1 索引（Index Add）

- `POST /api/v1/dedup/2d/index/add`（`cad-ml-platform` 对外接口；内部调用 `dedupcad-vision` 建立视觉索引）
- `multipart/form-data`
  - `file`: `PNG/JPG/PDF`（用于 L1/L2/L3）
  - `geom_json`（可选，推荐）：`application/json`（插件导出的 v2 JSON，用于 L4）
  - query/params：
    - `user_name`
    - `upload_to_s3`（或对象存储）

服务端行为（MVP）：

- 视觉侧照常建立索引。
- 若 `geom_json` 存在（由 `cad-ml-platform` 落库/落盘）：
  - 存储：`geoms/{file_hash}.v2.json`（或按对象存储规范分区）
  - 记录 `geom_hash`（建议：canonical json 的 sha256，用于变更检测）

cad-ml-platform 本地落盘默认路径：

- `DEDUPCAD_GEOM_STORE_DIR`（默认 `data/dedup_geom`）

### 4.2 搜索（Search）

- `POST /api/v1/dedup/2d/search`
- `multipart/form-data`
  - `file`: `PNG/JPG/PDF`
  - `geom_json`（可选，推荐）：v2 JSON
  - query params：
    - `mode`: `fast|balanced|precise`
    - `max_results`: 召回数量上限（默认 50）
    - `enable_precision`: 是否启用 L4（默认 true；当 query 有 geom_json 时生效）
    - `preset`（可选）：`strict|version|loose`（只在调用方未显式传入相关参数时，自动填充 `mode/precision_top_n/阈值/权重` 的默认值）
    - `precision_profile`（可选）：`strict|version`（L4 几何精查 profile；不传时由 `preset` 或租户默认配置决定）
    - `version_gate`（可选）：`off|auto|file_name|meta`（仅对 `precision_profile=version` 生效；用于“版本查重”时优先匹配同图号/同文件基名，降低跨零件误报）
    - `precision_top_n`: 精查候选数量（默认 20）
    - `precision_visual_weight`: 融合权重（默认 0.3）
    - `precision_geom_weight`: 融合权重（默认 0.7）
    - `precision_compute_diff`（可选）：是否输出 L4 JSON 差异（默认 false；建议只对 Top 命中开启）
    - `precision_diff_top_n`（可选）：最多输出多少个候选的 JSON 差异（默认 5）
    - `precision_diff_max_paths`（可选）：差异路径截断上限（默认 200）
    - `duplicate_threshold`: “重复/duplicate”阈值（默认 0.95）
    - `similar_threshold`: “相似/similar”阈值（默认 0.80；要求 `similar_threshold <= duplicate_threshold`）

返回建议（兼容现有结构）：

- 在每个候选上补充：
  - `precision_score`（0~1）
  - `precision_breakdown`（可选 breakdown）
  - `visual_similarity`（可选，融合前视觉分数）

### 4.3 调试接口（可选）

- `POST /api/v1/dedup/2d/precision/compare`：上传两份 `geom_json`，返回 `score + breakdown`
- `GET /api/v1/dedup/2d/geom/{file_hash}/exists`：检查候选 `file_hash` 是否已有落盘 JSON

### 4.4 租户默认阈值/预设（可选，推荐产品化）

为避免插件/网页每次请求都携带一堆阈值参数，`cad-ml-platform` 支持按 `X-API-Key` 保存默认配置：

- `GET /api/v1/dedup/2d/presets`：查看内置预设（`strict|version|loose`）
- `GET /api/v1/dedup/2d/config`：读取当前租户默认配置
- `PUT /api/v1/dedup/2d/config`：写入当前租户默认配置（需要 `X-Admin-Token`）
- `DELETE /api/v1/dedup/2d/config`：清空当前租户默认配置（需要 `X-Admin-Token`）

应用优先级（从高到低）：

1. 请求显式 query params
2. 请求显式 `preset=...` 的预设默认值
3. 租户默认配置（含默认 `preset` 与阈值/权重覆盖）
4. 服务端参数默认值

## 5. 存储与一致性设计

### 5.1 主键与哈希

- `file_hash`：对上传 `file` 原始 bytes 做 sha256（便于去重与幂等）。
- `geom_hash`：对 v2 JSON 做 canonical 化后 sha256（便于判断几何是否变更）。
- `drawing_id`：索引库内部自增/UUID，返回给调用方用于追溯。

### 5.2 JSON 落库策略（必须做）

- 索引时把 v2 JSON 落盘/落对象存储，**避免每次精查都重新解析 DWG/DXF**。
- 候选精查时只做“加载 JSON + 计算相似度”，保证延迟可控。

## 6. 关键代码设计（Precision Layer）

### 6.1 Vendoring 结构（落地在 `cad-ml-platform`）

在 `cad-ml-platform/src/core/dedupcad_precision/` 新增：

```
dedupcad_precision/
  __init__.py
  store.py                  # v2 JSON 落盘（按 file_hash）
  verifier.py               # weighted_similarity 封装
  vendor/                   # 从 dedupcad vendoring 的核心
    __init__.py
    config.py
    scoring.py
    entities_match.py
    v2_normalize.py
    json_diff.py
    neighbor_index.py
```

**原则**：尽量不改 vendored 文件（便于后续同步）；通过 wrapper/adapter 解决配置与依赖。

### 6.2 L4 执行策略

- 输入：`query_json` + `candidate_json`
- 输出：`precision_score` + `breakdown`（可选）+ `diff_summary`（可选）
- 仅对 L3 输出的 **Top-N** 执行（默认 20），避免几何匹配的 O(N²) 成本放大。

### 6.3 分数融合（推荐默认策略）

- `final = visual_w * visual + geom_w * precision`
- 推荐预设默认权重（可按业务再标定）：
  - `strict`：`visual_w=0.3`，`geom_w=0.7`（更偏低误报）
  - `version`：`visual_w=0.5`，`geom_w=0.5`（更偏版本归并）
  - `loose`：`visual_w=0.6`，`geom_w=0.4`（更偏召回）
- 若候选缺少 JSON：
  - 默认不参与 L4（保留视觉分数，但降低置信度/标记 `precision_missing`）
- 可选 gated 策略：
  - `precision < th => 直接剔除`（用于“强一致”查重场景）

### 6.4 精查默认策略（重要）

- `precision_profile=strict`：
  - 更偏“精确”而非“召回”
  - `CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH` 默认关闭（可通过 env 设置为 `1` 开启）
- `precision_profile=version`：
  - 更偏“版本相似/召回”
  - 强制开启 `entities_geom_hash`（对导出差异/轻微偏移更鲁棒）
  - 启用 `entities_spatial_enable`：在“几何 bag-of-features”之外加入 **空间分布签名**，减少“不同零件但局部特征分布相似”导致的误报
  - 关闭 Hungarian / Frechet / Procrustes 等重匹配开关以提升吞吐
  - 下调 `dimensions/blocks` 权重，避免版本差异导致“一票否决”
- 当双方缺少 `dimensions` / `text_content` / `hatches` 等可选段时，L4 会自动把对应权重置 `0`，避免“缺失即满分”导致的分数虚高（仅在两边都有该段时才参与打分）。

### 6.5 阈值标定（训练集弱标签）

> 结论：如果不引入标题栏/图号/版本号等元数据 gate，仅靠相似度阈值很难同时做到“版本高召回”和“低误报”（主流商业软件通常会把元数据作为强约束）。

建议用脚本对你的数据集做阈值扫描（弱标签校准）：

```bash
python3 scripts/dedup_2d_threshold_scan_manifest.py \
  --matches-csv data/dedup_report_train_local_version_profile_spatial_full/matches.csv \
  --expected-groups-json data/train_drawings_manifest/expected_groups.json
```

基于当前训练集（109 张）扫描结果：

- `pos_min ≈ 0.698`，`neg_max ≈ 0.852`（存在明显重叠）
- 若只想“自动聚类”且尽量 0 误报（对该训练集）：阈值可取 `>=0.86`（但会漏掉变化较大的版本）
- 若想“召回更多版本候选”：可先用 `0.80~0.82` 做候选过滤，再叠加元数据 gate（同图号）做聚类

## 7. DWG 批量查重（Windows 优先）的更优路径

你提出的“DWG→DWF”更偏展示格式，不利于提取高保真几何语义；建议：

- **最佳**：插件侧直接产出 v2 JSON（无需服务端装 ODA/LibreDWG）。
- **批量/离线（Windows）**：
  - DWG → DXF：`ODAFileConverter` / `dwg2dxf` / `accoreconsole`
  - DXF → v2 JSON：使用 `dedupcad` 的 extractor（本地 worker 执行）
  - 可选 DXF/DWG → PNG：用 `accoreconsole` Plot 或已有渲染链路生成缩略图
  - 然后调用 `/api/v1/dedup/2d/index/add` 批量入库

Linux 视客户环境：若无可靠 DWG 转换能力，可降级“仅视觉查重”或强制走插件。

### 7.1 批量入库脚本（PNG + v2 JSON）

当你已经能批量得到 `*.png + *.json`（同名）时，可直接用平台脚本入库：

```bash
python3 scripts/dedup_2d_batch_index.py <input_dir> \
  --base-url http://localhost:8000 \
  --api-key test \
  --user-name batch
```

说明：

- 脚本默认会在结束后调用 `POST /api/v1/dedup/2d/index/rebuild`，让 `dedupcad-vision` 的 L1/L2 索引进入 `ready` 状态（可用 `--no-rebuild-index` 禁用）。
- 索引状态可通过 `GET /api/v1/dedup/2d/health` 查看（会透传 `dedupcad-vision /health` 的 `indexes`）。

### 7.2 批量生成（DWG/DXF → PNG + v2 JSON → 入库）

当输入是 `DWG/DXF` 时，可用脚本自动完成：

1) DXF → PNG（ezdxf + matplotlib 渲染，跨平台）  
2) DXF → v2 JSON（ezdxf 解析；并额外提取 `DIMENSION/HATCH` 写入 `dimensions/hatches`，用于后续精查增强）  
3) 生成后的 `PNG + v2 JSON` 调用 `/api/v1/dedup/2d/index/add` 入库

命令示例（DXF 目录）：

```bash
python3 scripts/dedup_2d_batch_ingest_cad.py <input_dir> \
  --base-url http://localhost:8000 \
  --api-key test \
  --user-name batch \
  --work-dir data/dedupcad_batch
```

说明：

- 脚本默认会在结束后调用 `POST /api/v1/dedup/2d/index/rebuild`（可用 `--no-rebuild-index` 禁用）。
- 做 smoke test 可加 `--max-files 10` 先验证小样本。
- 对 DWG：建议在 Windows 使用“无 UI 批处理”导出 `PNG + v2 JSON`，再走 7.1 的入库脚本（见 7.3）。

DWG 转 DXF（Windows 推荐 ODA File Converter）：

- 环境变量方式：设置 `ODA_FILE_CONVERTER_EXE` 指向 `ODAFileConverter.exe`
- 或命令行参数：`--oda-exe "C:\\Path\\To\\ODAFileConverter.exe"`

也可用自定义命令模板（例如已有 `dwg2dxf.exe`）：

- 设置环境变量 `DWG_TO_DXF_CMD='dwg2dxf \"{input}\" \"{output}\"'`
- 或使用 `--dwg-to-dxf cmd --dwg-to-dxf-cmd 'dwg2dxf \"{input}\" \"{output}\"'`

### 7.3 Windows 无 UI 批处理（推荐：accoreconsole + 插件导出 JSON）

目标：在“用户不打开 DWG”的前提下，尽量保留 DWG 级语义（层/块/标注/文字），并输出：

- `*.png`：用于视觉召回（L1/L2）
- `*.v2.json`：用于几何精查（L4）

推荐路径（Windows）：

1) 在装有 AutoCAD 的机器上，用 `accoreconsole.exe` 批量跑脚本（.scr/.lsp/.net），调用你们已有插件命令导出 `PNG + v2 JSON`。  
2) 把导出的目录（同名 `png + v2.json`）交给 `scripts/dedup_2d_batch_index.py` 批量入库。
3) 在服务端生成“批量查重报告”（匹配明细 + 重复分组）：

```bash
python3 scripts/dedup_2d_batch_search_report.py <export_output_dir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --within-input-only \
  --duplicate-threshold 0.95 \
  --similar-threshold 0.80 \
  --output-dir data/dedup_report
```

也可以用预设（更推荐给非算法用户）：

```bash
python3 scripts/dedup_2d_batch_search_report.py <export_output_dir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --within-input-only \
  --preset strict \
  --output-dir data/dedup_report
```

如需“可解释差异”（L4 JSON diff）可加：

```bash
python3 scripts/dedup_2d_batch_search_report.py <export_output_dir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --within-input-only \
  --preset strict \
  --precision-compute-diff \
  --save-precision-diffs \
  --output-dir data/dedup_report
```

输出：

- `data/dedup_report/matches.csv`：逐张图纸的 Top-K 匹配明细
- `data/dedup_report/groups.json` / `groups.csv`：重复分组（同一组视为重复簇）
- `data/dedup_report/summary.json`：总体统计
- `data/dedup_report/precision_diffs/`（可选）：逐对匹配的 L4 JSON diff（用于人工复核）

生成静态 HTML 报告（推荐交付给业务/工程复核）：

```bash
python3 scripts/dedup_2d_generate_html_report.py data/dedup_report \
  --max-matches-rows 300
```

会输出 `data/dedup_report/index.html`，可直接双击打开。

如果 Windows 机器也有 Python 环境，并且可访问服务端，可直接用一键脚本跑通“导出→入库→报告→HTML→ZIP”：

- `scripts/windows/dedup_end_to_end.ps1:1`

如需把报告打包给客户/同事（不依赖原始 `data/train_artifacts` 目录），可一键生成“自包含”目录 + zip：

```bash
python3 scripts/dedup_2d_package_report.py data/dedup_report \
  --overwrite
```

输出：

- `data/dedup_report_package/index.html`（自包含，可离线打开）
- `data/dedup_report_package.zip`（可直接发送）

PowerShell 示例（需将命令名替换为你们插件实际命令）：

```powershell
$accore = "C:\\Program Files\\Autodesk\\AutoCAD 2024\\accoreconsole.exe"
$scr = "C:\\dedup\\export.scr"
$inDir = "D:\\dwg"

Get-ChildItem -Path $inDir -Recurse -Filter *.dwg | ForEach-Object {
  & $accore /i $_.FullName /s $scr /l en-US
}
```

`export.scr` 示例（示意，按你们插件实现调整参数）：

```text
NETLOAD
C:\\dedup\\DedupPlugin.dll
DEDUP_EXPORT_V2JSON
C:\\dedup\\out\\$(DWGNAME).v2.json
DEDUP_EXPORT_PNG
C:\\dedup\\out\\$(DWGNAME).png
QUIT
```

落地建议：

- 推荐直接使用仓库脚本模板：`scripts/windows/accoreconsole_batch_export.ps1:1` 与 `scripts/windows/README.md:1`

关键点：

- 插件导出命令最好支持“输入 DWG → 输出路径”，且不弹 UI 对话框（便于自动化）。
- 输出文件名建议稳定：同 stem 的 `png + v2.json`，便于平台脚本配对入库。
- 若必须服务端处理 DWG：建议只做“视觉降级”（提取 DWG thumbnail 或转图），精查依赖插件 v2 JSON。

## 8. 实施路线图（建议分期）

### Phase 1（MVP，对标主流的关键一步）

- `dedupcad-vision`：
  - 保持“视觉召回”为主：`/api/index/add` 建视觉索引、`/api/v2/search` 输出 Top-K 候选
- `cad-ml-platform`：
  - 对外统一入口：`/api/v1/dedup/2d/index/add` 接收 `PNG + v2 geom_json`，并落盘 JSON
  - 精查与融合：`/api/v1/dedup/2d/search` 在 Top-N 上执行 `weighted_similarity`，输出 `precision_score + final_score`
  - 批处理友好：`POST /api/v1/dedup/2d/index/rebuild` 触发 vision 侧索引 ready
  - 统一鉴权与观测（metrics/logs）

### Phase 2（进一步缩小差距）

- 启用/接入 ONNX embedding（L3 真正的深度表征），提升召回质量与抗干扰能力。
- 精查输出更可解释的差异报告（层/块/标注级摘要）。

### Phase 3（服务端兜底）

- DXF 上传兜底：服务端 `DXF -> v2 JSON`（`ezdxf`）+（可选）渲染 PNG。

### Phase 4（3D）

- 3D：STEP/IGES/Parasolid 等，采用 3D embedding + 局部特征 + 拓扑/装配结构精查（独立 pipeline）。

## 9. 验收指标（对标主流必须有）

- 数据集：真实客户图纸 + 合成扰动（旋转/缩放/线宽/黑白反转/局部裁剪）。
- 指标：
  - `Recall@K`（K=10/50）
  - `Precision@K`
  - 误报率（False Positive）在“强一致查重”场景可控
- 性能：
  - 召回阶段：< 300ms（库规模相关）
  - Top-20 精查：< 1~2s（视 JSON 复杂度与匹配算法开关）

### 9.1 阈值调参与校准（建议必做）

查重实际会用到 **三类阈值**：

- `duplicate_threshold`：最终判定 “duplicate” 的阈值（更严格）
- `similar_threshold`：最终判定 “similar” 的阈值（更宽松）
- `group_threshold`：批量报告里“分组/成簇”的阈值（用于把相似图纸连成重复簇；常与 `similar_threshold` 或 `duplicate_threshold` 对齐）

建议做法：

1) 先用本地 L4（纯几何/JSON）在你的数据集上跑出“正负样本分布”，得到一个合理区间  
2) 再把该阈值应用到 API（vision+L4 融合）上，验证是否达到目标（误报可控 + 召回可用）

弱标签（例如“文件名版本组”）场景下，可直接用：

```bash
python3 scripts/dedup_2d_threshold_scan_manifest.py \
  --expected-groups-json data/train_drawings_manifest/expected_groups.json \
  --matches-csv data/dedup_report_train_local_070/matches.csv
```

### 9.2 推荐预设（MVP）

> 预设只是起点；最终以客户数据集校准结果为准。

- **严格重复（强一致）**：`duplicate_threshold=0.95`，`similar_threshold=0.80`，`group_threshold=0.95`
- **版本查重（同图不同版）**：`duplicate_threshold=0.95`，`similar_threshold=0.70`，`group_threshold=0.70`
- **宽松相似（召回优先）**：`duplicate_threshold=0.90`，`similar_threshold=0.50`，`group_threshold=0.50`

也可以直接用 `preset`（等价快捷方式）：

- `preset=strict` ≈ 严格重复
- `preset=version` ≈ 版本查重
- `preset=loose` ≈ 宽松相似

建议直接使用评测脚本落地（按“分组目录”组织数据集）：

```bash
python3 scripts/dedup_2d_eval_groups.py <dataset_root> \
  --base-url http://localhost:8000 \
  --api-key test \
  --top-k 10 \
  --enable-precision
```

## 10. 可参考的开源组件（工程落地）

- 视觉近重复：`imagehash` / `imagededup`、`faiss`
- CAD 解析：`ezdxf`（DXF）、（DWG 需商业/客户端能力）
- 3D：`open3d`、`trimesh`（后续阶段）
- 本项目可复用：`dedupcad`（v2 JSON + weighted_similarity）
