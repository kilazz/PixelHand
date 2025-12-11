@echo off
setlocal EnableDelayedExpansion
title PixelHand Portable Launcher (DirectML)

:: --- 1. Path Configuration ---
cd /d "%~dp0"
set "TOOLS_DIR=%~dp0.tools"
set "UV_EXE=%TOOLS_DIR%\uv.exe"

set "UV_CACHE_DIR=%TOOLS_DIR%\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%TOOLS_DIR%\uv-python"
set "UV_TOOL_DIR=%TOOLS_DIR%\uv-tools"
set "UV_SYSTEM_PYTHON=0"

:: === ENVIRONMENT ISOLATION ===
set "UV_PROJECT_ENVIRONMENT=%~dp0.venv-directml"

echo =======================================================
echo      PixelHand Portable Launcher via UV [DirectML]
echo =======================================================
echo.
echo [INFO] Tools Dir: .tools
echo [INFO] Venv Dir:  .venv-directml
echo.

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

:: --- 3. Launch Application ---
echo [INFO] Syncing dependencies for DirectML...
echo.

"%UV_EXE%" run --extra directml main.py %*

if errorlevel 1 (
    echo.
    echo !!!!!!!!!!!!!!!!!!!! [APPLICATION CRASHED] !!!!!!!!!!!!!!!!!!!!
    pause
    exit /b 1
)

echo.
echo [INFO] Application exited normally.
pause