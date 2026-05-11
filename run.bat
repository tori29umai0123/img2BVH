@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "UV_LINK_MODE=copy"

REM --- 1) ensure uv is available ----------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
    echo [image2BVH] 'uv' not found on PATH. Installing uv ...
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [image2BVH] uv install failed. Please install manually: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

REM --- 2) sync deps via uv (creates .venv on first run) -----------------
echo [image2BVH] Syncing dependencies (this can take a while on first run) ...
uv sync --no-dev
if errorlevel 1 (
    echo [image2BVH] dependency install failed.
    pause
    exit /b 1
)

REM --- 3) HuggingFace login check (SAM 3 is gated) ----------------------
call :CheckHfAuth

REM --- 4) launch app -----------------------------------------------------
echo [image2BVH] Launching ...
uv run --no-dev python -m image2bvh %*
goto :AfterLaunch


:CheckHfAuth
REM Skip the prompt if SAM 3 is already downloaded locally.
if exist "%~dp0runtime\models\sam3\config.json" (
    echo [image2BVH] SAM 3 model already present locally, skipping HF login prompt.
    goto :EOF
)
REM Skip the prompt if HF_TOKEN is already in env or the user has logged in.
if defined HF_TOKEN (
    echo [image2BVH] HF_TOKEN already set in environment, skipping login prompt.
    goto :EOF
)
uv run --no-dev huggingface-cli whoami >nul 2>&1
if not errorlevel 1 (
    REM Already logged in via huggingface-cli login.
    goto :EOF
)
echo.
echo =====================================================================
echo  SAM 3 ^(used for person mask generation^) is GATED on HuggingFace.
echo.
echo   1. Apply for access at  https://huggingface.co/facebook/sam3
echo      ^(manual approval by Meta — usually a few hours to a few days^)
echo   2. Create a read token at  https://huggingface.co/settings/tokens
echo   3. Paste the token below ^(or press Enter to skip and configure later^)
echo =====================================================================
set "IMG2BVH_HFTOKEN="
set /p "IMG2BVH_HFTOKEN=HF_TOKEN: "
if not "!IMG2BVH_HFTOKEN!"=="" (
    echo [image2BVH] Saving HF token via huggingface-cli login ...
    uv run --no-dev huggingface-cli login --token "!IMG2BVH_HFTOKEN!"
    set "IMG2BVH_HFTOKEN="
) else (
    echo [image2BVH] Skipped. Without a valid token, SAM 3 download will fail
    echo             until you set HF_TOKEN or run 'huggingface-cli login'.
)
goto :EOF


:AfterLaunch
endlocal
