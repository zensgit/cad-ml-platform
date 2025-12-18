Param(
  # Input DWG directory (recursively searched)
  [Parameter(Mandatory = $true)]
  [string]$InputDir,

  # Working output directory (artifacts + reports)
  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  # Pipeline mode:
  #  - ODA: uses ODAFileConverter.exe to convert DWG->DXF, then server-side extracts v2 JSON + renders PNG.
  #  - Plugin: uses accoreconsole + your plugin export command to produce PNG + v2 JSON directly.
  [ValidateSet("ODA", "Plugin")]
  [string]$Mode = "ODA",

  # cad-ml-platform API
  [string]$BaseUrl = "http://127.0.0.1:18000",
  [string]$ApiKey = "test",
  [string]$UserName = "batch",

  # Report tuning
  [ValidateSet("strict", "version", "loose")]
  [string]$Preset = "strict",
  # Optional: when Preset=version and you want to force candidate gating.
  # off|auto|file_name|meta (auto prefers meta, falls back to file name).
  [ValidateSet("", "off", "auto", "file_name", "meta")]
  [string]$VersionGate = "",
  [switch]$EnablePrecisionDiff,
  [int]$MaxFiles = 0,

  # ODA mode: path to ODAFileConverter.exe
  [string]$OdaExe = "",
  [string]$OdaOutputVersion = "ACAD2018",

  # Plugin mode: accoreconsole.exe + plugin dll + export command template
  [string]$AccoreConsoleExe = "",
  [string]$PluginDll = "",
  [string]$ExportCommandTemplate = "",

  # Python runner (defaults to python on PATH)
  [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Run([string]$Cmd, [string[]]$Args) {
  Write-Host ""
  Write-Host ("[run] " + $Cmd + " " + ($Args -join " "))
  & $Cmd @Args
  if ($LASTEXITCODE -ne 0) {
    throw "command failed: $Cmd (exit=$LASTEXITCODE)"
  }
}

$baseUrlTrim = $BaseUrl.TrimEnd("/")

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\\..")).Path
$InputDirAbs = (Resolve-Path -LiteralPath $InputDir).Path
Ensure-Dir $OutputDir
$OutputDirAbs = (Resolve-Path -LiteralPath $OutputDir).Path

$ArtifactsDir = Join-Path $OutputDirAbs "artifacts"
$ReportDir = Join-Path $OutputDirAbs ("report_" + $Preset.ToLower())
Ensure-Dir $ArtifactsDir
Ensure-Dir $ReportDir

Write-Host "=== Dedup 2D End-to-End (Windows) ==="
Write-Host ("RepoRoot: " + $RepoRoot)
Write-Host ("Mode: " + $Mode)
Write-Host ("InputDir: " + $InputDirAbs)
Write-Host ("OutputDir: " + $OutputDirAbs)
Write-Host ("ArtifactsDir: " + $ArtifactsDir)
Write-Host ("ReportDir: " + $ReportDir)
Write-Host ("BaseUrl: " + $BaseUrl)
Write-Host ("Preset: " + $Preset)
Write-Host ("MaxFiles: " + $MaxFiles)

Write-Host ""
Write-Host "=== Health Checks ==="
try {
  $health = Invoke-RestMethod -Uri ($baseUrlTrim + "/health") -Method Get -TimeoutSec 10
  Write-Host ("[ok] cad-ml-platform /health: " + ($health.status))
} catch {
  throw ("cad-ml-platform not reachable: " + $baseUrlTrim + " (check service is running)")
}
try {
  $headers = @{ "X-API-Key" = $ApiKey }
  $d2 = Invoke-RestMethod -Uri ($baseUrlTrim + "/api/v1/dedup/2d/health") -Method Get -Headers $headers -TimeoutSec 10
  Write-Host ("[ok] dedup 2d proxy: " + ($d2.status))
} catch {
  throw "dedupcad-vision not reachable via cad-ml-platform (/api/v1/dedup/2d/health). Check DEDUPCAD_VISION_URL."
}

if ($Mode -eq "Plugin") {
  if (-not $AccoreConsoleExe) { throw "Mode=Plugin requires -AccoreConsoleExe" }
  if (-not $PluginDll) { throw "Mode=Plugin requires -PluginDll" }
  if (-not $ExportCommandTemplate) { throw "Mode=Plugin requires -ExportCommandTemplate" }

  $exportScript = Join-Path $RepoRoot "scripts\\windows\\accoreconsole_batch_export.ps1"
  $args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $exportScript,
    "-InputDir", $InputDirAbs,
    "-OutputDir", $ArtifactsDir,
    "-AccoreConsoleExe", $AccoreConsoleExe,
    "-PluginDll", $PluginDll,
    "-ExportCommandTemplate", $ExportCommandTemplate,
    "-PreserveSubdirs", "true",
    "-ScriptEncoding", "Default",
    "-MaxFiles", "$MaxFiles"
  )
  Run "powershell.exe" $args
} else {
  if (-not $OdaExe) { throw "Mode=ODA requires -OdaExe (path to ODAFileConverter.exe)" }

  $ingest = Join-Path $RepoRoot "scripts\\dedup_2d_batch_ingest_cad.py"
  $args = @(
    $ingest,
    $InputDirAbs,
    "--work-dir", $ArtifactsDir,
    "--base-url", $BaseUrl,
    "--api-key", $ApiKey,
    "--user-name", $UserName,
    "--dwg-to-dxf", "oda",
    "--oda-exe", $OdaExe,
    "--oda-output-version", $OdaOutputVersion
  )
  if ($MaxFiles -gt 0) { $args += @("--max-files", "$MaxFiles") }
  Run $PythonExe $args
}

# Index exported PNG + v2 JSON (if you exported via Plugin mode without indexing).
if ($Mode -eq "Plugin") {
  $index = Join-Path $RepoRoot "scripts\\dedup_2d_batch_index.py"
  Run $PythonExe @(
    $index,
    $ArtifactsDir,
    "--base-url", $BaseUrl,
    "--api-key", $ApiKey,
    "--user-name", $UserName,
    "--require-json",
    "--rebuild-index"
  )
}

# Batch search report
$report = Join-Path $RepoRoot "scripts\\dedup_2d_batch_search_report.py"
$reportArgs = @(
  $report,
  $ArtifactsDir,
  "--base-url", $BaseUrl,
  "--api-key", $ApiKey,
  "--engine", "api",
  "--no-index",
  "--preset", $Preset,
  "--within-input-only",
  "--output-dir", $ReportDir
)
if ($VersionGate) {
  $reportArgs += @("--version-gate", $VersionGate)
}
if ($EnablePrecisionDiff) {
  $reportArgs += @("--precision-compute-diff", "--save-precision-diffs")
}
Run $PythonExe $reportArgs

# HTML report
$html = Join-Path $RepoRoot "scripts\\dedup_2d_generate_html_report.py"
Run $PythonExe @($html, $ReportDir, "--max-matches-rows", "300")

# Package report as ZIP (self-contained)
$pkg = Join-Path $RepoRoot "scripts\\dedup_2d_package_report.py"
Run $PythonExe @($pkg, $ReportDir, "--overwrite")

Write-Host ""
Write-Host ("[ok] report html: " + (Join-Path $ReportDir "index.html"))
Write-Host ("[ok] packaged zip: " + ($ReportDir + "_package.zip"))
