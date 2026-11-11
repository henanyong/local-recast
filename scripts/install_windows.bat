@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.11 and check "Add python.exe to PATH".
    exit /b 1
)

python scripts\install_windows.py
exit /b %ERRORLEVEL%
