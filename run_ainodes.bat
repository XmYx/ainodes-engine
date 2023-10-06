@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%"
set "PYTHON_SCRIPT=%APP_DIR%\main.py"

set "PYTHON_DIR=%SCRIPT_DIR%src\python"
set "PYTHON_LIB_DIR=%PYTHON_DIR%\Lib"
set "PYTHON_SCRIPTS_DIR=%PYTHON_DIR%\Scripts"
set "PATH=%PYTHON_SCRIPTS_DIR%;%PYTHON_LIB_DIR%;%PYTHON_DIR%;%PATH%"

set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\nodes_env

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

@rem setup installer env
call python ainodes_frontend/main.py %*
