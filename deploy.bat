@echo off
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"

set "TARGET_PY=3.12.0"
set "TARGET_PY_SERIES=3.12"

echo ==============================================
echo Stock Screener one-click deploy
echo ==============================================

set "PY_CMD="
set "CURRENT_PY_VER="

call :detect_python

if not defined PY_CMD (
    echo [WARN] Python %TARGET_PY_SERIES%.x not found.
    call :offer_install_python
    if %errorlevel% neq 0 exit /b 1
    call :detect_python
)

if not defined PY_CMD (
    echo [ERROR] Python %TARGET_PY_SERIES%.x is still not available after install attempt.
)

for /f "delims=" %%v in ('%PY_CMD% -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set "CURRENT_PY_VER=%%v"

echo [INFO] Using Python command: %PY_CMD%
echo [INFO] Detected Python version: %CURRENT_PY_VER%

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)"
if %errorlevel% neq 0 (
    echo [WARN] Required Python major.minor is %TARGET_PY_SERIES%.x.
    call :offer_install_python
    if %errorlevel% neq 0 exit /b 1
    call :detect_python
)

if not defined PY_CMD (
    echo [ERROR] Python %TARGET_PY_SERIES%.x is required by this project.
    exit /b 1
)

for /f "delims=" %%v in ('%PY_CMD% -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set "CURRENT_PY_VER=%%v"
echo [INFO] Final Python version: %CURRENT_PY_VER%

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)"
if %errorlevel% neq 0 (
    echo [ERROR] Python %TARGET_PY_SERIES%.x is required by this project.
    exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment in .venv ...
    %PY_CMD% -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

echo [INFO] Activating virtual environment ...
call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo [INFO] Installing dependencies ...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b 1
)

if /i "%~1"=="--no-run" (
    echo [OK] Deploy finished. App was not started with --no-run.
    exit /b 0
)

echo [INFO] Starting Streamlit app ...
streamlit run app.py

goto :eof

:detect_python
set "PY_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3.12"
        exit /b 0
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python"
        exit /b 0
    )
)

set "LOCAL_PY=%LocalAppData%\Programs\Python\Python312\python.exe"
if exist "%LOCAL_PY%" (
    "%LOCAL_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=%LOCAL_PY%"
        exit /b 0
    )
)

exit /b 0

:offer_install_python
echo [INFO] This project requires Python %TARGET_PY_SERIES%.x (preferred %TARGET_PY%).
set "USER_CHOICE="
set /p USER_CHOICE=Do you agree to download and install Python %TARGET_PY_SERIES%.x now? [Y/N]: 

if /I "!USER_CHOICE!"=="Y" goto :install_python
if /I "!USER_CHOICE!"=="YES" goto :install_python

echo [ERROR] Installation cancelled by user.
exit /b 1

:install_python
where winget >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] winget is not available.
    echo [HINT] Please manually install Python %TARGET_PY% from https://www.python.org/downloads/release/python-3120/
    exit /b 1
)

echo [INFO] Repairing or installing Python %TARGET_PY_SERIES%.x with winget ...
winget repair --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements --force
if %errorlevel% neq 0 (
    winget install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements --force
    if %errorlevel% neq 0 (
        echo [ERROR] winget failed to repair or install Python %TARGET_PY_SERIES%.x.
        echo [HINT] Try manual install: https://www.python.org/downloads/release/python-3120/
        exit /b 1
    )
)

echo [INFO] Python installation completed. Re-detecting interpreter ...
exit /b 0
