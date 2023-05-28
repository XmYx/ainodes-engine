@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%"
set "PYTHON_SCRIPT=%APP_DIR%\main.py"

set "VBS_SCRIPT=%TEMP%\RunPythonScript.vbs"
>"%VBS_SCRIPT%" (
    echo Set oWS = CreateObject("WScript.Shell"^)
    echo oWS.Run "cmd /C nodes_env/Scripts/activate.bat & python.exe ""%PYTHON_SCRIPT%""", 0
)

cscript //nologo "%VBS_SCRIPT%"
del "%VBS_SCRIPT%"

endlocal