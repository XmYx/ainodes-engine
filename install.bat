@echo off
rem Thin wrapper around install.py - see `python install.py --help` for options.
setlocal
cd /D "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 install.py %*
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.11 or newer from python.org,
        echo ticking "Add python.exe to PATH", then run this script again.
        exit /b 1
    )
    python install.py %*
)
endlocal
