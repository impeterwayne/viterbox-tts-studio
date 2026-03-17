@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
::  Viterbox Studio — Run with Virtual Environment
:: ============================================================
::  Usage:
::    run.bat              → Normal launch
::    run.bat --reinstall  → Force reinstall all dependencies
::    run.bat --cpu        → Force CPU mode (no CUDA)
:: ============================================================

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON=python"
set "REINSTALL=0"
set "FORCE_CPU=0"

:: Parse arguments
for %%A in (%*) do (
    if /I "%%A"=="--reinstall" set "REINSTALL=1"
    if /I "%%A"=="--cpu" set "FORCE_CPU=1"
)

echo.
echo ══════════════════════════════════════════════════════
echo   🎙️  Viterbox Studio Launcher
echo ══════════════════════════════════════════════════════
echo.

:: ── Step 1: Check Python ──────────────────────────────
echo [1/4] Checking Python...
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.10+ and add it to PATH.
    echo    Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%V in ('%PYTHON% --version 2^>^&1') do (
    echo       Found Python %%V
)

:: ── Step 2: Create / Activate venv ────────────────────
echo [2/4] Setting up virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo       Creating new venv...
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo       ✅ venv created
    set "REINSTALL=1"
) else (
    echo       ✅ venv found
)

:: Activate
call "%VENV_DIR%\Scripts\activate.bat"

:: ── Step 3: Install dependencies ──────────────────────
echo [3/4] Checking dependencies...

if "%REINSTALL%"=="1" (
    echo       Installing dependencies (this may take a while^)...
    echo.

    :: Upgrade pip first
    python -m pip install --upgrade pip --quiet

    :: Install PyTorch with CUDA if available
    if "%FORCE_CPU%"=="1" (
        echo       Mode: CPU only
        pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu --quiet
    ) else (
        echo       Mode: CUDA (GPU^)
        pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet
    )
    if errorlevel 1 (
        echo.
        echo ❌ Failed to install PyTorch! See errors above.
        pause
        exit /b 1
    )

    :: Install remaining requirements
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo.
        echo ❌ Failed to install requirements! See errors above.
        pause
        exit /b 1
    )

    :: Install the package itself in editable mode
    pip install -e . --quiet
    if errorlevel 1 (
        echo.
        echo ❌ Failed to install viterbox package! See errors above.
        pause
        exit /b 1
    )

    echo.
    echo       ✅ All dependencies installed
) else (
    :: Quick check — if key packages are missing, trigger install
    python -c "import torch; import librosa; import pyrubberband" >nul 2>&1
    if errorlevel 1 (
        echo       ⚠️  Missing packages detected. Running install...
        echo       (Use --reinstall to force a full reinstall^)
        echo.
        python -m pip install --upgrade pip --quiet
        if "%FORCE_CPU%"=="1" (
            pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu --quiet
        ) else (
            pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet
        )
        if errorlevel 1 (
            echo.
            echo ❌ Failed to install PyTorch! See errors above.
            pause
            exit /b 1
        )
        pip install -r requirements.txt --quiet
        if errorlevel 1 (
            echo.
            echo ❌ Failed to install requirements! See errors above.
            pause
            exit /b 1
        )
        pip install -e . --quiet
        if errorlevel 1 (
            echo.
            echo ❌ Failed to install viterbox package! See errors above.
            pause
            exit /b 1
        )
        echo       ✅ Dependencies installed
    ) else (
        echo       ✅ Dependencies OK
    )
)

:: ── Step 4: Launch the app ────────────────────────────
echo [4/4] Launching Viterbox Studio...
echo.
echo ══════════════════════════════════════════════════════
echo   🌐 Open in browser: http://localhost:7861
echo   Press Ctrl+C to stop the server
echo ══════════════════════════════════════════════════════
echo.

python studio_api.py

:: Deactivate on exit
deactivate >nul 2>&1
pause
