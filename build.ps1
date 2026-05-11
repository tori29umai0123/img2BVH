<#
.SYNOPSIS
    Build script for the image2BVH portable Windows distribution.

.DESCRIPTION
    Produces a single self-extracting EXE (7-Zip SFX) that:
      * Writes ZERO registry entries
      * Defaults its extraction destination to the directory the EXE itself
        sits in (i.e. ".\image2bvh\") — user can still browse to change
      * Is uninstalled by deleting the extracted folder; no separate
        uninstaller is needed (there is no installer in the registry sense)

    Pipeline:
      1. uv sync (project deps) + install PyInstaller into the venv
      2. PyInstaller --noconfirm packaging\image2bvh.spec → dist\image2bvh\
      3. Assemble LICENSE_BUNDLE.txt + Portable_README.txt under dist\image2bvh\
      4. Download portable 7-Zip CLI / SFX module into .build-tools\7zip\
         (cached across builds; ~1.5 MB one-time fetch)
      5. Compress dist\image2bvh\ to dist\image2bvh-0.1.0.7z (LZMA2 ultra)
      6. Prepend 7zSD.sfx + SFX config → dist\image2bvh-0.1.0-portable.exe

    First run takes ~25-50 minutes (PyInstaller ~10 min + 7-Zip LZMA2
    compression ~15-30 min on a 9 GB tree). Subsequent runs reuse the
    PyInstaller build cache and the .build-tools\7zip\ download cache.

.PARAMETER Clean
    Wipe dist\ and build\ before starting. Use after editing the .spec or
    changing model files; otherwise PyInstaller's cache may surface stale
    state.

.PARAMETER SkipPyInstaller
    Skip steps (1) and (2). Use to re-package an existing dist\image2bvh\
    tree without rebuilding the EXE bundle.

.PARAMETER SkipPackage
    Skip steps (4)-(6). Use to test the PyInstaller output locally without
    waiting for LZMA2 compression.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Clean
    .\build.ps1 -SkipPyInstaller
#>
#requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$SkipPyInstaller,
    [switch]$SkipPackage
)

# Windows PowerShell 5.1 wraps every line a native exe writes to stderr in
# an ErrorRecord (NativeCommandError) the moment any stream-capturing wrapper
# is in play (e.g. running the script via ``*> build.log``). With Stop, uv's
# "Resolved 117 packages" status — written to stderr — would halt the script
# before $LASTEXITCODE is even checked. Use Continue and rely on explicit
# ``throw`` for every check below; that way native-stderr-as-status doesn't
# kill the build, but a real PyInstaller/7-Zip failure still surfaces.
$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSCommandPath
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"
$ToolsDir = Join-Path $ProjectRoot ".build-tools\7zip"
$SpecFile = Join-Path $ProjectRoot "packaging\image2bvh.spec"

$AppVersion = "0.1.0"
$AppName = "image2bvh"

Write-Host "[build] Project root: $ProjectRoot"

# --- 0) optional clean -----------------------------------------------------
if ($Clean) {
    Write-Host "[build] Cleaning dist\ and build\"
    if (Test-Path $DistDir)  { Remove-Item -Recurse -Force $DistDir }
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
}

# --- 1) ensure uv is available --------------------------------------------
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    throw "uv not found on PATH. Install from https://docs.astral.sh/uv/"
}

Push-Location $ProjectRoot
try {
    # --- 2) sync deps + install PyInstaller transiently --------------------
    if (-not $SkipPyInstaller) {
        Write-Host "[build] uv sync (project deps)"
        & uv sync --no-dev
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit $LASTEXITCODE)" }

        Write-Host "[build] Installing PyInstaller into the venv (transient build dep)"
        & uv pip install "pyinstaller>=6.10"
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller install failed (exit $LASTEXITCODE)" }

        # --- 3) PyInstaller -----------------------------------------------
        Write-Host "[build] Running PyInstaller against $SpecFile"
        Write-Host "[build]   (this can take several minutes; ~9 GB output tree)"
        & uv run --no-dev pyinstaller --noconfirm $SpecFile
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }
    }
    else {
        Write-Host "[build] Skipping PyInstaller (-SkipPyInstaller)"
    }
}
finally {
    Pop-Location
}

$expectedExe = Join-Path $DistDir "image2bvh\image2bvh.exe"
if (-not (Test-Path $expectedExe)) {
    throw "Expected $expectedExe missing — PyInstaller output incomplete"
}

# --- 4) LICENSE_BUNDLE.txt + Portable_README.txt into the bundle root ------
# Both files end up at dist\image2bvh\<file>, so they show up at the root of
# whatever folder the user extracts the SFX to. SAM License §1.b.i / DINOv3
# License §1.b.i require redistributing parties to "provide a copy of this
# Agreement with any such [Materials]" — shipping the bundled license file
# alongside the binary satisfies that.
Write-Host "[build] Assembling LICENSE_BUNDLE.txt"

$bundleRoot = Join-Path $DistDir "image2bvh"

$samLicPath  = Join-Path $ProjectRoot "LICENSES\SAM_LICENSE.txt"
$dinoLicPath = Join-Path $ProjectRoot "LICENSES\DINOv3_LICENSE.md"
$ownLicPath  = Join-Path $ProjectRoot "LICENSE"
foreach ($p in @($samLicPath, $dinoLicPath, $ownLicPath)) {
    if (-not (Test-Path $p)) { throw "Missing license file: $p" }
}

$header = @'
================================================================================
image2BVH redistributes Meta's SAM 3, SAM 3D Body, and DINOv3 models. By
using this software you agree to the following licenses:

  1. SAM License           (Meta's Segment Anything Model 3 / SAM 3D Body)
  2. DINOv3 License        (Meta's DINOv3 vision backbone)
  3. image2BVH LICENSE     (this project's own license)

Each license is reproduced in full below.
================================================================================

'@

$samLic  = Get-Content -Raw -Path $samLicPath
$dinoLic = Get-Content -Raw -Path $dinoLicPath
$ownLic  = Get-Content -Raw -Path $ownLicPath

$sep1 = "`r`n`r`n================================================================================`r`n 1. SAM License (Meta)`r`n================================================================================`r`n`r`n"
$sep2 = "`r`n`r`n================================================================================`r`n 2. DINOv3 License (Meta)`r`n================================================================================`r`n`r`n"
$sep3 = "`r`n`r`n================================================================================`r`n 3. image2BVH LICENSE`r`n================================================================================`r`n`r`n"

$bundle = $header + $sep1 + $samLic + $sep2 + $dinoLic + $sep3 + $ownLic
Set-Content -Path (Join-Path $bundleRoot "LICENSE_BUNDLE.txt") -Value $bundle -Encoding utf8

Write-Host "[build] Copying README.txt (JA) + README.en.txt (EN) from templates"

# Bilingual end-user READMEs ship as plain-text .txt files at the bundle
# root so they appear right next to image2bvh.exe once the SFX is
# extracted. We deliberately source them from external template files
# (packaging\templates\) instead of embedding them as PowerShell
# here-strings: PS 5.1's here-string parsing is fragile around non-ASCII
# characters when the script file lacks a UTF-8 BOM, and a malformed
# here-string can produce a SILENTLY EMPTY string with no error message
# (we hit this when the EN README was written as 0 bytes).
$templatesDir = Join-Path $ProjectRoot "packaging\templates"

$readmeJaTpl = Join-Path $templatesDir "README.txt"
$readmeEnTpl = Join-Path $templatesDir "README.en.txt"
foreach ($p in @($readmeJaTpl, $readmeEnTpl)) {
    if (-not (Test-Path $p)) { throw "Missing README template: $p" }
}

# Read template, substitute __VERSION__, write to bundle root.
# Read as raw bytes → UTF-8 to avoid PS 5.1's auto-codec heuristics.
function _RenderReadme {
    param([string]$Src, [string]$Dst, [string]$Version)
    $bytes = [System.IO.File]::ReadAllBytes($Src)
    $text  = [System.Text.Encoding]::UTF8.GetString($bytes).Replace("__VERSION__", $Version)
    $utf8Bom = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText($Dst, $text, $utf8Bom)
}
_RenderReadme -Src $readmeJaTpl -Dst (Join-Path $bundleRoot "README.txt")    -Version $AppVersion
_RenderReadme -Src $readmeEnTpl -Dst (Join-Path $bundleRoot "README.en.txt") -Version $AppVersion

# Remove legacy README variants left by earlier builds.
foreach ($legacy in @("Portable_README.txt", "README.md", "README.en.md")) {
    $p = Join-Path $bundleRoot $legacy
    if (Test-Path $p) { Remove-Item $p -Force }
}


# --- 5) Download portable 7-Zip CLI + SFX module --------------------------
# Strategy:
#   * 7zr.exe (~600 KB) is the standalone bootstrap that can extract .7z.
#   * 7z2301-extra.7z (~1.5 MB) holds 7za.exe (full CLI) — but does NOT
#     include any .sfx modules (those were dropped from the "extras"
#     package after 23.01).
#   * 7z2409-x64.exe (the main 7-Zip installer, NSIS-built ~1.5 MB) bundles
#     7z.sfx. We extract its contents with 7za.exe — without ever running
#     the installer, so no system-wide install / registry writes happen
#     on the build machine.
# All three downloads are cached in .build-tools\7zip\ so subsequent
# builds skip them.
if (-not $SkipPackage) {
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    $sevenZr   = Join-Path $ToolsDir "7zr.exe"
    $extraArc  = Join-Path $ToolsDir "7z-extra.7z"
    $sevenZa   = Join-Path $ToolsDir "7za.exe"
    $mainExe   = Join-Path $ToolsDir "7z-main.exe"
    $mainDir   = Join-Path $ToolsDir "7z-main"
    $sfxModule = Join-Path $mainDir "7z.sfx"

    if (-not (Test-Path $sevenZr)) {
        Write-Host "[build] Downloading 7zr.exe (~600 KB)"
        Invoke-WebRequest -Uri "https://www.7-zip.org/a/7zr.exe" -OutFile $sevenZr -UseBasicParsing
    }
    # ---- (5a) 7za.exe from the "extras" package -------------------------
    if (-not (Test-Path $sevenZa)) {
        # NOTE: the "extras" cadence is independent of the main release
        # cadence and lags significantly. As of 2026 the latest "extras"
        # is still 23.01 (Jan 2023) — newer main releases like 24.09 do
        # NOT have matching extras packages. Don't bump this URL without
        # first verifying the file actually exists on the server.
        $extraUrl = "https://www.7-zip.org/a/7z2301-extra.7z"
        Write-Host "[build] Downloading 7-Zip extras from $extraUrl (~1.5 MB)"
        Invoke-WebRequest -Uri $extraUrl -OutFile $extraArc -UseBasicParsing
        if (-not (Test-Path $extraArc) -or (Get-Item $extraArc).Length -lt 100KB) {
            throw "7-Zip extras download was incomplete or corrupt"
        }
        & $sevenZr x $extraArc "-o$ToolsDir" -y | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Extracting 7-Zip extras failed (exit $LASTEXITCODE)" }

        # 7z2301-extra.7z layout:
        #   7za.exe          (x86 standalone CLI)
        #   x64\7za.exe      (x64 standalone CLI — preferred when present)
        #   7za.dll, 7zxa.dll, Far\... (Far Manager plugin — not used)
        # No .sfx modules are bundled here — see step (5b) for those.
        $sevenZaSrc = Join-Path $ToolsDir "x64\7za.exe"
        if (Test-Path $sevenZaSrc) {
            Copy-Item $sevenZaSrc $sevenZa -Force
        }
        if (-not (Test-Path $sevenZa)) {
            throw "7za.exe not found in extras package (looked in root and x64/)"
        }
    }

    # ---- (5b) 7z.sfx extracted from the main 7-Zip installer ------------
    # 7-Zip's NSIS installer is itself a 7-Zip-readable archive. We can
    # extract its contents with 7za.exe to retrieve the bundled 7z.sfx
    # WITHOUT ever running the installer — that means no admin prompt,
    # no shell associations, no registry writes on the build machine.
    if (-not (Test-Path $sfxModule)) {
        $mainUrl = "https://www.7-zip.org/a/7z2409-x64.exe"
        Write-Host "[build] Downloading 7-Zip main installer from $mainUrl (~1.5 MB)"
        Invoke-WebRequest -Uri $mainUrl -OutFile $mainExe -UseBasicParsing
        if (-not (Test-Path $mainExe) -or (Get-Item $mainExe).Length -lt 100KB) {
            throw "7-Zip main installer download was incomplete or corrupt"
        }
        Write-Host "[build] Extracting 7z.sfx out of the installer (no install performed)"
        if (Test-Path $mainDir) { Remove-Item -Recurse -Force $mainDir }
        & $sevenZa x $mainExe "-o$mainDir" -y | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Extracting 7-Zip main installer failed (exit $LASTEXITCODE)" }
        # NSIS installers expand into a tree with $PLUGINSDIR / $_OUTDIR
        # sub-folders. 7z.sfx may not be at the root — search recursively.
        if (-not (Test-Path $sfxModule)) {
            $found = Get-ChildItem -Path $mainDir -Recurse -Filter "7z.sfx" -File -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($found) {
                Copy-Item $found.FullName $sfxModule -Force
            }
        }
        if (-not (Test-Path $sfxModule)) {
            $allFiles = (Get-ChildItem -Path $mainDir -Recurse -File -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Name -Unique) -join ", "
            throw "7z.sfx not found in extracted main installer at $mainDir. Files seen: $allFiles"
        }
    }

    # --- 6) Compress dist\image2bvh\ to dist\image2bvh-<ver>.7z -----------
    # LZMA2 ultra + multi-threaded + solid mode. Solid gives ~5-15% extra
    # compression for trees full of small Python files at the cost of slower
    # random access — fine here because the SFX extracts the whole archive
    # in one shot.
    $arc7z = Join-Path $DistDir "$AppName-$AppVersion.7z"
    if (Test-Path $arc7z) { Remove-Item $arc7z -Force }

    Write-Host "[build] Compressing dist\image2bvh\ → $arc7z (LZMA2 ultra, multi-threaded)"
    Write-Host "[build]   This can take 15-30 minutes on a 9 GB tree."
    Push-Location $DistDir
    try {
        & $sevenZa a -t7z -mx=9 -mmt=on -ms=on $arc7z "image2bvh\*"
        if ($LASTEXITCODE -ne 0) { throw "7-Zip compression failed (exit $LASTEXITCODE)" }
    }
    finally {
        Pop-Location
    }

    # --- 7) Build SFX config (UTF-8 BOM is REQUIRED by 7z.sfx) ------------
    # The ;!@Install@!UTF-8! marker is matched byte-for-byte by the SFX
    # extractor — encoding must be UTF-8 with BOM, or the SFX silently
    # ignores the config and falls back to default settings (extracts to
    # %TEMP% with no prompt).
    #
    # 7z.sfx supports: Title, BeginPrompt, Progress, Directory, RunProgram,
    # ExecuteFile, ExecuteParameters. There is NO destination chooser
    # dialog — `Directory` is the fixed extraction target. `.\image2bvh`
    # resolves relative to the CWD when the SFX runs, which under Explorer
    # double-click is the directory containing the .exe.
    $sfxConfig = Join-Path $DistDir "sfx_config.txt"
    $cfgText = @"
;!@Install@!UTF-8!
Title="image2BVH $AppVersion portable"
BeginPrompt="image2BVH portable をこのフォルダの直下 (.\image2bvh\) に解凍します。`r`n`r`nレジストリは一切書き換えません。`r`nアンインストールは解凍された image2bvh\ フォルダを削除するだけです。`r`n`r`n別の場所に解凍したい場合は、この .exe を希望のフォルダへ移動してから実行してください。`r`n`r`n続行しますか?"
Progress="yes"
Directory=".\image2bvh"
;!@InstallEnd@!
"@
    # Force UTF-8 BOM. PS 5.1's ``Set-Content -Encoding utf8`` does emit a
    # BOM, but be explicit so future PowerShell versions don't change this
    # out from under the SFX.
    $utf8Bom = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText($sfxConfig, $cfgText, $utf8Bom)

    # --- 8) Concat 7zSD.sfx + config + archive → portable .exe ------------
    $portableExe = Join-Path $DistDir "$AppName-$AppVersion-portable.exe"
    if (Test-Path $portableExe) { Remove-Item $portableExe -Force }

    Write-Host "[build] Concatenating SFX module + config + archive → $portableExe"
    # cmd's `copy /b` is the canonical way to concatenate binary files
    # without any PowerShell stream re-encoding shenanigans.
    & cmd /c "copy /b `"$sfxModule`" + `"$sfxConfig`" + `"$arc7z`" `"$portableExe`" > NUL"
    if ($LASTEXITCODE -ne 0) { throw "SFX concat failed (exit $LASTEXITCODE)" }

    if (-not (Test-Path $portableExe)) {
        throw "Expected $portableExe missing after concat"
    }
    $sizeGb = [math]::Round((Get-Item $portableExe).Length / 1GB, 2)
    Write-Host "[build] OK: $portableExe  ($sizeGb GB)"

    # Keep the intermediate .7z and config around for debug. They can be
    # removed by hand if you want to reclaim disk space.
}
else {
    Write-Host "[build] Skipping packaging (-SkipPackage)"
}

Write-Host "[build] Done."
