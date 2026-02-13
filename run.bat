@echo off
setlocal EnableDelayedExpansion
title PixelHand Portable Launcher

:: --- 1. Path Configuration ---
cd /d "%~dp0"
set "TOOLS_DIR=%~dp0.tools"
set "UV_EXE=%TOOLS_DIR%\uv.exe"

:: --- 2. Check and Install UV Locally ---
if not exist "%UV_EXE%" (
    echo [INFO] Portable 'uv' not found. Downloading...
    if exist "%TOOLS_DIR%" rmdir /s /q "%TOOLS_DIR%"
    mkdir "%TOOLS_DIR%"
    powershell -Command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '%TOOLS_DIR%\uv.zip'"
    powershell -Command "Expand-Archive -Path '%TOOLS_DIR%\uv.zip' -DestinationPath '%TOOLS_DIR%' -Force"
    powershell -Command "Get-ChildItem -Path '%TOOLS_DIR%' -Filter 'uv.exe' -Recurse | Move-Item -Destination '%TOOLS_DIR%' -Force"
    if exist "%TOOLS_DIR%\uv.zip" del "%TOOLS_DIR%\uv.zip"
    if not exist "%UV_EXE%" (
        echo [ERROR] Failed to setup uv.exe.
        pause
        exit /b 1
    )
    echo [OK] uv installed successfully.
)

:: --- 3. Select Mode ---
set "MODE=%~1"
if "%MODE%"=="" (
    echo =======================================================
    echo      PixelHand Portable Launcher
    echo =======================================================
    echo Select Execution Provider:
    echo 1. CPU
    echo 2. CUDA
    echo 3. DirectML
    echo 4. WebGPU
    echo.
    set /p "CHOICE=Enter number (1-4): "
    if "!CHOICE!"=="1" set "MODE=cpu"
    if "!CHOICE!"=="2" set "MODE=cuda"
    if "!CHOICE!"=="3" set "MODE=directml"
    if "!CHOICE!"=="4" set "MODE=webgpu"
)

:: Default fallback
if "%MODE%"=="" set "MODE=cpu"

:: Map mode to venv and extra
set "VENV_DIR=.venv-%MODE%"
set "UV_PROJECT_ENVIRONMENT=%~dp0%VENV_DIR%"

echo.
echo [INFO] Mode: %MODE%
echo [INFO] Venv: %VENV_DIR%
echo.

:: --- 4. Launch ---
echo [INFO] Syncing dependencies...
"%UV_EXE%" run --extra %MODE% main.py %*

if errorlevel 1 (
    echo.
    echo !!!!!!!!!!!!!!!!!!!! [APPLICATION CRASHED] !!!!!!!!!!!!!!!!!!!!
    pause
    exit /b 1
)

echo.
echo [INFO] Application exited normally.
pause