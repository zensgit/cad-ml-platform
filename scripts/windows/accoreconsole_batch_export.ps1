Param(
  [Parameter(Mandatory = $true)]
  [string]$InputDir,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [Parameter(Mandatory = $true)]
  [string]$AccoreConsoleExe,

  [Parameter(Mandatory = $true)]
  [string]$PluginDll,

  [Parameter(Mandatory = $true)]
  [string]$ExportCommandTemplate,

  [string]$Language = "en-US",

  # Where to place outputs:
  #   - $true  => preserve subdirectory tree under OutputDir (recommended to avoid name collisions)
  #   - $false => flat output dir (legacy behavior)
  [bool]$PreserveSubdirs = $true,

  # Script encoding for .scr files. "Default" uses system ANSI code page (often best for Chinese paths).
  [ValidateSet("Default", "ASCII", "OEM", "UTF8", "Unicode")]
  [string]$ScriptEncoding = "Default",

  # Process at most N files (0 = no limit)
  [int]$MaxFiles = 0,

  [switch]$KeepScripts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Get-RelativePath([string]$Base, [string]$FullPath) {
  $baseFull = (Resolve-Path -LiteralPath $Base).Path.TrimEnd("\\")
  $full = (Resolve-Path -LiteralPath $FullPath).Path
  if ($full.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $full.Substring($baseFull.Length).TrimStart("\\")
  }
  return [System.IO.Path]::GetFileName($full)
}

if (-not (Test-Path -LiteralPath $InputDir)) {
  throw "InputDir not found: $InputDir"
}
if (-not (Test-Path -LiteralPath $AccoreConsoleExe)) {
  throw "AccoreConsoleExe not found: $AccoreConsoleExe"
}
if (-not (Test-Path -LiteralPath $PluginDll)) {
  throw "PluginDll not found: $PluginDll"
}

Ensure-Dir $OutputDir
$scrDir = Join-Path $OutputDir "scr"
$logDir = Join-Path $OutputDir "logs"
Ensure-Dir $scrDir
Ensure-Dir $logDir

$dwgs = Get-ChildItem -Path $InputDir -Recurse -Filter *.dwg -File
if ($dwgs.Count -eq 0) {
  Write-Host "No DWG files found under: $InputDir"
  exit 0
}

$ok = 0
$failed = 0
$processed = 0

foreach ($dwg in $dwgs) {
  if ($MaxFiles -gt 0 -and $processed -ge $MaxFiles) { break }

  $rel = Get-RelativePath $InputDir $dwg.FullName
  $relNoExt = [System.IO.Path]::ChangeExtension($rel, $null)

  if ($PreserveSubdirs) {
    $outJson = Join-Path $OutputDir "$relNoExt.v2.json"
    $outPng = Join-Path $OutputDir "$relNoExt.png"
    $scrPath = Join-Path $scrDir "$relNoExt.scr"
    $logPath = Join-Path $logDir "$relNoExt.log"
  } else {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($dwg.Name)
    $outJson = Join-Path $OutputDir "$stem.v2.json"
    $outPng = Join-Path $OutputDir "$stem.png"
    $scrPath = Join-Path $scrDir "$stem.scr"
    $logPath = Join-Path $logDir "$stem.log"
  }

  Ensure-Dir ([System.IO.Path]::GetDirectoryName($outJson))
  Ensure-Dir ([System.IO.Path]::GetDirectoryName($outPng))
  Ensure-Dir ([System.IO.Path]::GetDirectoryName($scrPath))
  Ensure-Dir ([System.IO.Path]::GetDirectoryName($logPath))

  $cmd = $ExportCommandTemplate.
    Replace("{dwg}", $dwg.FullName).
    Replace("{json}", $outJson).
    Replace("{png}", $outPng)

  $scrLines = @(
    "FILEDIA",
    "0",
    "CMDDIA",
    "0",
    "NETLOAD",
    "`"$PluginDll`"",
    $cmd,
    "QUIT",
    "N"
  )
  $scrLines | Out-File -FilePath $scrPath -Encoding $ScriptEncoding

  Write-Host "[run] $($dwg.FullName)"
  Write-Host "      -> $outPng"
  Write-Host "      -> $outJson"

  try {
    & $AccoreConsoleExe /i $dwg.FullName /s $scrPath /l $Language *>> $logPath
    if ($LASTEXITCODE -ne 0) {
      throw "accoreconsole exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $outJson)) {
      throw "missing output json: $outJson"
    }
    if (-not (Test-Path -LiteralPath $outPng)) {
      throw "missing output png: $outPng"
    }

    Write-Host "[ok]  $rel"
    $ok += 1
  }
  catch {
    Write-Host "[fail] $rel -> $($_.Exception.Message)"
    Write-Host "       log: $logPath"
    $failed += 1
  }
  finally {
    if (-not $KeepScripts) {
      Remove-Item -LiteralPath $scrPath -Force -ErrorAction SilentlyContinue
    }
  }

  $processed += 1
}

Write-Host "done: total=$($dwgs.Count) ok=$ok failed=$failed out=$OutputDir"
if ($failed -gt 0) { exit 2 }
exit 0
