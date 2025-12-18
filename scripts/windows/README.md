# Windows 批处理导出（DWG → PNG + v2 JSON）

目标：在“用户不打开 DWG”的情况下，通过 `accoreconsole.exe` 批量调用 CAD 插件命令，导出：

- `*.png`：用于视觉召回（L1/L2）
- `*.v2.json`：用于几何精查（L4）

## 1) 前置条件

- 安装 AutoCAD（`accoreconsole.exe` 可用）
- 插件 DLL 可通过 `NETLOAD` 加载，并提供“无 UI”导出命令
- `cad-ml-platform` 已运行，且其环境变量 `DEDUPCAD_VISION_URL` 指向同机的 `dedupcad-vision`（例如 `http://127.0.0.1:58001`）

## 2) 使用

### 推荐：一键端到端（导出 → 入库 → 报告 → HTML → ZIP）

如果这台 Windows 机器也装了 Python，并且能访问 `cad-ml-platform`（以及其依赖的 `dedupcad-vision`），可以直接跑端到端 smoke test：

#### Mode A：ODA（不依赖 CAD UI）

适合快速验证“Windows 批处理”链路（只需要安装 ODA File Converter）：

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode ODA `
  -InputDir "D:\dwg" `
  -OutputDir "D:\dedup_out" `
  -OdaExe "C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe" `
  -BaseUrl "http://<cad-ml-platform-host>:8000" `
  -ApiKey "<X-API-Key>" `
  -Preset "strict" `
  -EnablePrecisionDiff
```

#### Mode B：Plugin（accoreconsole + 插件无 UI 导出）

适合最终交付形态（插件直接产出 `PNG + v2 JSON`）：

```powershell
.\scripts\windows\dedup_end_to_end.ps1 `
  -Mode Plugin `
  -InputDir "D:\dwg" `
  -OutputDir "D:\dedup_out" `
  -AccoreConsoleExe "C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe" `
  -PluginDll "C:\dedup\DedupPlugin.dll" `
  -ExportCommandTemplate 'DEDUP_EXPORT "{json}" "{png}"' `
  -BaseUrl "http://<cad-ml-platform-host>:8000" `
  -ApiKey "<X-API-Key>" `
  -Preset "strict" `
  -EnablePrecisionDiff
```

可选：当你在做“版本查重/同图不同版”（`-Preset version`）且想强制使用候选 gate（更贴近主流商业软件的“同图号/同基名再比几何”逻辑），可加：

- `-VersionGate meta`：要求插件 JSON 里带 `meta.drawing_number` 等字段（推荐最终形态）
- `-VersionGate file_name`：按文件名 `xxx_v1/xxx_v2` 自动门控（兜底）
- `-VersionGate auto`：优先 meta，缺失则回退 file_name（默认推荐）

输出：

- `D:\dedup_out\report_<preset>\index.html`
- `D:\dedup_out\report_<preset>_package.zip`（可直接发客户/同事）

---

编辑/确认参数后运行：

```powershell
.\scripts\windows\accoreconsole_batch_export.ps1 `
  -InputDir "D:\dwg" `
  -OutputDir "D:\dedup_out" `
  -AccoreConsoleExe "C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe" `
  -PluginDll "C:\dedup\DedupPlugin.dll" `
  -ExportCommandTemplate 'DEDUP_EXPORT "{json}" "{png}"'
```

说明：

- `ExportCommandTemplate` 支持占位符：
  - `{dwg}`：当前 DWG 的绝对路径
  - `{json}`：输出 JSON 绝对路径（默认 `<OutputDir>\<stem>.v2.json`）
  - `{png}`：输出 PNG 绝对路径（默认 `<OutputDir>\<stem>.png`）
- 脚本会为每个 DWG 写入日志：`<OutputDir>\logs\<stem>.log`
- 推荐参数：
  - `-PreserveSubdirs $true`：保持输入目录层级，避免同名 DWG 输出冲突
  - `-ScriptEncoding Default`：更适合包含中文路径的环境（可按实际 CAD/OS 调整）
  - `-MaxFiles 10`：做 smoke test 时先跑 10 个文件

## 3) 导出后入库

导出完成（同名 `png + v2.json`）后，在服务端/网关执行：

```bash
python3 scripts/dedup_2d_batch_index.py <OutputDir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --user-name batch
```

生成“批量查重报告”（匹配明细 + 重复分组）：

```bash
python3 scripts/dedup_2d_batch_search_report.py <OutputDir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --duplicate-threshold 0.95 \
  --similar-threshold 0.80 \
  --within-input-only \
  --output-dir data/dedup_report
```

也可以用预设（更推荐给非算法用户）：

```bash
python3 scripts/dedup_2d_batch_search_report.py <OutputDir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --preset strict \
  --within-input-only \
  --output-dir data/dedup_report
```

如需“可解释差异”（L4 JSON diff）可加：

```bash
python3 scripts/dedup_2d_batch_search_report.py <OutputDir> \
  --base-url http://<cad-ml-platform-host>:8000 \
  --api-key <X-API-Key> \
  --preset strict \
  --precision-compute-diff \
  --save-precision-diffs \
  --within-input-only \
  --output-dir data/dedup_report
```

生成静态 HTML 报告（推荐交付给业务/工程复核）：

```bash
python3 scripts/dedup_2d_generate_html_report.py data/dedup_report \
  --max-matches-rows 300
```

如需把报告打包给客户/同事（不依赖原始导出目录），可一键生成“自包含”目录 + zip：

```bash
python3 scripts/dedup_2d_package_report.py data/dedup_report \
  --overwrite
```
