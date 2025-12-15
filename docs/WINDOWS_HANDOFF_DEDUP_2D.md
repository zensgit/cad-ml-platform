# Windows 切换开发/验收交接文档（2D 查重）

> 用途：把当前对话与实现进度固化到仓库里，方便你切换到 Windows 电脑后继续推进“批量 DWG 查重”验收与联调。

## 1. 当前完成情况（macOS 已完成）

- 已完成 2D 查重的 **L4 v2 JSON 精查** 接入（`precision_profile=strict|version`），并支持阈值/权重可配。
- 已新增 `version_gate`（`off|auto|file_name|meta`）：用于“版本查重”时先按同图号/同基名门控，再比几何，降低跨零件误聚类。
- 已完成训练集（109 张 `PNG + *.v2.json`）离线全矩阵验证（`local_l4`）与报告输出：
  - 报告 ZIP：`data/dedup_report_train_local_version_profile_spatial_full_package.zip`
  - 阈值扫描结论：`pos_min≈0.698`、`neg_max≈0.852`（仅靠阈值难以稳定“版本聚类”，需要叠加 gate）
- 说明/方案文档：
  - `docs/CAD_DEDUP_2D_VISION_JSON_PRECISION.md`
  - `reports/dedup_2d_validation_train_20251215.md`

## 2. Windows 上的目标（验收口径）

同一套 DWG 数据，分别跑两档报告并产出可交付的 `index.html + zip`：

- **strict（强一致重复）**：只把几乎一致的图纸判为 duplicate（低误报）
- **version（同图不同版）**：尽量把同一图纸的不同版本召回/聚类（高召回），建议打开 `VersionGate`

最终以 **Plugin/accoreconsole 批量导出（含标题栏 meta）** 作为生产主链路；ODA 作为兜底链路。

## 3. Windows 同机部署约定（必须）

同一台 Windows 机器上，至少需要：

- `dedupcad-vision`：建议固定 `http://127.0.0.1:58001`
- `cad-ml-platform`：对外一个端口（例如 `http://127.0.0.1:8000` 或 `http://127.0.0.1:18000`）
- `cad-ml-platform` 启动时必须设置：
  - `DEDUPCAD_VISION_URL=http://127.0.0.1:58001`

健康检查（任意方式）：

- `GET <BaseUrl>/health`
- `GET <BaseUrl>/api/v1/dedup/2d/health`（带 `X-API-Key`）

## 4. Windows 一键端到端脚本（推荐）

脚本位置：`scripts/windows/dedup_end_to_end.ps1`

它会自动完成：

1) 导出/生成工件（`artifacts/`：`PNG + *.v2.json`）
2) 入库索引（必要时）
3) 批量查重报告（CSV/JSON）
4) 生成 HTML
5) 打包 ZIP（自包含，可直接发客户/同事）

### 4.1 ODA 模式（兜底链路 / smoke test 推荐）

strict：

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode ODA `
  -InputDir "D:\dwg_test" `
  -OutputDir "D:\dedup_out_oda" `
  -OdaExe "C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe" `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "strict" `
  -MaxFiles 20
```

version（建议启用 gate）：

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode ODA `
  -InputDir "D:\dwg_test" `
  -OutputDir "D:\dedup_out_oda" `
  -OdaExe "C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe" `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "version" `
  -VersionGate "auto" `
  -MaxFiles 20
```

### 4.2 Plugin 模式（生产主链路 / 对标主流）

strict：

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode Plugin `
  -InputDir "D:\dwg_test" `
  -OutputDir "D:\dedup_out_plugin" `
  -AccoreConsoleExe "C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe" `
  -PluginDll "C:\dedup\DedupPlugin.dll" `
  -ExportCommandTemplate 'DEDUP_EXPORT "{json}" "{png}"' `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "strict" `
  -MaxFiles 20
```

version（建议启用 gate）：

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode Plugin `
  -InputDir "D:\dwg_test" `
  -OutputDir "D:\dedup_out_plugin" `
  -AccoreConsoleExe "C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe" `
  -PluginDll "C:\dedup\DedupPlugin.dll" `
  -ExportCommandTemplate 'DEDUP_EXPORT "{json}" "{png}"' `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "version" `
  -VersionGate "auto" `
  -MaxFiles 20
```

> `VersionGate` 说明：
>
> - `auto`：优先 `meta`，缺失则回退到 `file_name`（推荐默认）
> - `meta`：要求插件 JSON 中有 `meta.drawing_number` 等字段（推荐最终形态）
> - `file_name`：按文件名 `xxx_v1/xxx_v2` 兜底门控

## 5. 验收产物（跑完给我复核用）

每次跑完脚本会输出 2 个关键文件（建议都保留）：

- `D:\dedup_out_*\report_<preset>\index.html`
- `D:\dedup_out_*\report_<preset>_package.zip`

如果你需要我远程复核/调参，至少把 `report_<preset>_package.zip` 发我即可。

## 6. 常见问题排查

- `dedup_end_to_end.ps1` 报 “dedupcad-vision not reachable”：
  - 检查 `cad-ml-platform` 的 `DEDUPCAD_VISION_URL` 是否正确指向 `http://127.0.0.1:58001`
  - 检查 `dedupcad-vision` 是否监听了该端口
- Plugin 导出失败：
  - 看 `artifacts\logs\*.log`
  - 中文路径问题：建议保持 `ScriptEncoding Default`（脚本默认已设置）
  - 插件导出命令是否支持无 UI（建议单独手动跑 1 个 DWG 验证命令）

