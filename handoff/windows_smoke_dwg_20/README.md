# Windows Smoke 数据集（20 个 DWG）

用于在 Windows 上做端到端 smoke test（不需要逐个打开 DWG）：

- 输入：本目录下 20 个 `*.dwg`
- 输出：`artifacts/`（`PNG + *.v2.json`）+ `report_*/index.html` + `report_*_package.zip`

## 推荐执行（同机部署）

假设：

- `dedupcad-vision`：`http://127.0.0.1:58001`
- `cad-ml-platform`：`http://127.0.0.1:18000`（按实际端口修改）
- `cad-ml-platform` 启动时已设置：`DEDUPCAD_VISION_URL=http://127.0.0.1:58001`

### 1) ODA 模式（兜底链路）

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode ODA `
  -InputDir "D:\dwg_smoke_20" `
  -OutputDir "D:\dedup_out_oda" `
  -OdaExe "C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe" `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "strict"
```

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode ODA `
  -InputDir "D:\dwg_smoke_20" `
  -OutputDir "D:\dedup_out_oda" `
  -OdaExe "C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe" `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "version" `
  -VersionGate "auto"
```

### 2) Plugin 模式（主链路）

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode Plugin `
  -InputDir "D:\dwg_smoke_20" `
  -OutputDir "D:\dedup_out_plugin" `
  -AccoreConsoleExe "C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe" `
  -PluginDll "C:\dedup\DedupPlugin.dll" `
  -ExportCommandTemplate 'DEDUP_EXPORT "{json}" "{png}"' `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "strict"
```

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode Plugin `
  -InputDir "D:\dwg_smoke_20" `
  -OutputDir "D:\dedup_out_plugin" `
  -AccoreConsoleExe "C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe" `
  -PluginDll "C:\dedup\DedupPlugin.dll" `
  -ExportCommandTemplate 'DEDUP_EXPORT "{json}" "{png}"' `
  -BaseUrl "http://127.0.0.1:18000" `
  -ApiKey "test" `
  -Preset "version" `
  -VersionGate "auto"
```

