@echo off
rem Launch aiNodes Engine from its virtual environment.
rem Run install.bat first if nodes_env\ does not exist yet.
setlocal
cd /D "%~dp0"

set "VENV=nodes_env"
if not "%AINODES_VENV%"=="" set "VENV=%AINODES_VENV%"
set "PYTHON=%VENV%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo No environment at %VENV% - run install.bat first.
    exit /b 1
)

"%PYTHON%" main.py %*
endlocal
