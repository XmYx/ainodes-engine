@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%"
set "PYTHON_SCRIPT=%APP_DIR%\main.py"

set "PYTHON_DIR=%SCRIPT_DIR%src\python"
set "PYTHON_LIB_DIR=%PYTHON_DIR%\Lib"
set "PYTHON_SCRIPTS_DIR=%PYTHON_DIR%\Scripts"
set "PATH=%PYTHON_SCRIPTS_DIR%;%PYTHON_LIB_DIR%;%PYTHON_DIR%;%PATH%"


set "VBS_SCRIPT=%TEMP%\RunPythonScript.vbs"
>"%VBS_SCRIPT%" (
    echo Set oWS = CreateObject("WScript.Shell"^)
    echo oWS.Run "cmd /C %SCRIPT_DIR%nodes_env/Scripts/activate.bat & %SCRIPT_DIR%nodes_env/Scripts/python.exe ""%PYTHON_SCRIPT%""", 0
)

cscript //nologo "%VBS_SCRIPT%"
del "%VBS_SCRIPT%"

endlocal