@echo off
setlocal enabledelayedexpansion

set "base_folder=nodes_env"
set "custom_nodes_folder=ai_nodes"
set "src_folder=src"
set "repositories_file=repositories.txt"
set "src_file=src.txt"
set "SCRIPT_DIR=%~dp0"

set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\nodes_env

REM Activate the conda environment
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || (
    echo.
    echo Unable to activate conda environment.
    goto end
)

REM Update the main repo
git pull
pip install -r requirements.txt

REM Update or clone repositories from repositories.txt in ai_nodes folder
for /f "tokens=*" %%A in (%repositories_file%) do (
    set "repository=%%A"
    echo Processing repository: !repository!

    REM If the repository exists, update it. If not, clone it.
    if exist "%custom_nodes_folder%\!repository!" (
        pushd "%custom_nodes_folder%\!repository!"
        git pull
        pip install -r requirements.txt
        popd
    ) else (
        pushd %custom_nodes_folder%
        git clone https://www.github.com/!repository!
        popd
    )
)

REM Update or clone repositories from src.txt in src folder
for /f "tokens=*" %%A in (%src_file%) do (
    set "repository=%%A"
    echo Processing repository: !repository!

    REM If the repository exists, update it. If not, clone it.
    if exist "%src_folder%\!repository!" (
        pushd "%src_folder%\!repository!"
        git pull
        popd
    ) else (
        pushd %src_folder%
        git clone https://www.github.com/!repository!
        popd
    )

    REM Update all first-level subfolders of the repository
    for /d %%B in ("%src_folder%\!repository!\*") do (
        if exist "%%B\.git" (
            pushd "%%B"
            echo Updating subfolder: %%B
            git pull
            popd
        )
    )
)

REM Return to the original directory
cd %SCRIPT_DIR%

REM Deactivate the conda environment
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" deactivate

:end
call run_ainodes.bat
endlocal