@echo off

REM Setting the CONDA_ROOT_PREFIX and INSTALL_ENV_DIR variables
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\nodes_env

REM Activate the Conda environment
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

REM Open a new command prompt window with the environment activated
start cmd

:end
exit /B