@echo off
setlocal enabledelayedexpansion

cd /D "%~dp0"
set "SCRIPT_DIR=%~dp0"
set "SRC_DIR=%~dp0src"
set "INSTALL_ENV_DIR=%cd%\nodes_env"
set "CONDA_ROOT_PREFIX=%cd%\installer_files\conda"
set "custom_nodes_folder=ai_nodes"
set "repositories_file=config/repositories.txt"
set "src_file=config/src.txt"

REM Check if nodes_env folder exists
if not exist "%INSTALL_ENV_DIR%" (
    echo nodes_env not found. Running setup_ainodes.bat...
    call setup_ainodes.bat
    exit /b
)

REM Delete ai_nodes/ainodes_engine_faceswapper_nodes folder if it exists
if exist "ai_nodes\ainodes_engine_faceswapper_nodes" (
    echo Deleting ai_nodes\ainodes_engine_faceswapper_nodes...
    rmdir /S /Q "ai_nodes\ainodes_engine_faceswapper_nodes"
)


REM Activate the conda venv
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

@REM REM Update all repositories in ai_nodes
@REM for /f "tokens=1,2" %%A in (%repositories_file%) do (
@REM     set "repository_name=%%~nxA"
@REM     echo Updating repository: !repository_name!
@REM     if exist "%custom_nodes_folder%\!repository_name!" (
@REM         cd "%custom_nodes_folder%\!repository_name!"
@REM         git fetch
@REM         git pull
@REM         if exist "requirements.txt" (
@REM             pip install -r requirements.txt
@REM         )
@REM         cd "%~dp0"
@REM     )
@REM )

call git pull

REM Update all repositories in src
for /f "tokens=*" %%A in (%src_file%) do (
    set "repository_name=%%~nxA"
    echo Updating repository: !repository_name!
    if exist "%SRC_DIR%\!repository_name!" (
        cd "%SRC_DIR%\!repository_name!"
        git fetch
        git pull
        if exist "requirements.txt" (
            pip install -r requirements.txt
        )
        cd "%~dp0"
    )
)

REM Update ComfyUI-Manager in src/ComfyUI/custom_nodes
if exist "%SRC_DIR%\deforum\src\ComfyUI\custom_nodes\ComfyUI-Manager" (
    echo Updating ComfyUI-Manager...
    cd "%SRC_DIR%\deforum\src\ComfyUI\custom_nodes\ComfyUI-Manager"
    git fetch
    git pull
    if exist "requirements.txt" (
        pip install -r requirements.txt
    )
    cd "%~dp0"
)

pip install -r requirements.txt

REM Run root folder's main.py
echo Running main.py...
python main.py

:end
endlocal
exit /b