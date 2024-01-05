@echo off

cd /D "%~dp0"
set "SRC_DIR=%~dp0src"
if not exist "%SRC_DIR%" mkdir "%SRC_DIR%"

set PATH=%PATH%;%SystemRoot%\system32

echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniconda which can not be silently installed under a path with spaces. && goto end

@rem Check for special characters in installation path
set "SPCHARMESSAGE="WARNING: Special characters were detected in the installation path!" "         This can cause the installation to fail!""
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]" >nul && (
	call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

@rem Fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files

@rem Deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

@rem Configuration
set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\nodes_env
set INSTALL_ENV_NAME=nodes_env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
set conda_exists=F

@rem Check if conda is already installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

@rem Download and install Miniconda if it's not installed
if "%conda_exists%" == "F" (
	echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL% to %INSTALL_DIR%\miniconda_installer.exe

	mkdir "%INSTALL_DIR%"
	call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe" || ( echo. && echo Miniconda failed to download. && goto end )

	echo Installing Miniconda to %CONDA_ROOT_PREFIX%
	start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

	@rem Test the conda binary
	echo Miniconda version:
	call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo Miniconda not found. && goto end )
)

@rem create the installer env
if not exist "%INSTALL_ENV_DIR%" (
	call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.10 || ( echo. && echo Conda environment creation failed. && goto end )
)

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

REM Install requirements
pip install -r requirements.txt
mkdir src
setlocal enabledelayedexpansion

@REM set "repositories_file=config/repositories.txt"
@REM set "custom_nodes_folder=ai_nodes"
@REM
@REM rem Read repositories and branch names from repositories.txt
@REM for /f "tokens=1,2" %%A in (%repositories_file%) do (
@REM   set "repository=%%A"
@REM   set "branch=%%B"
@REM   echo Cloning repository: !repository! Branch: !branch!
@REM
@REM   rem Go to ai_nodes folder and clone the repository
@REM   cd "%custom_nodes_folder%"
@REM   git clone -b !branch! https://www.github.com/!repository!
@REM
@REM   rem Return to the original directory
@REM   cd "%~dp0"
@REM )

@REM rem Go into each top subdirectory of ai_nodes and run pip install
@REM for /d %%B in (%custom_nodes_folder%\*) do (
@REM   echo Installing requirements in directory: %%B
@REM   if exist "%%B\requirements.txt" (
@REM     pushd "%%B"
@REM     pip install -r requirements.txt
@REM     popd
@REM   )
@REM )

@echo off
set "src_file=config/src.txt"

@REM rem Read repositories, branches, and directories from src.txt
@REM for /f "tokens=1,2,3,4" %%A in (%src_file%) do (
@REM     set "repository=%%A"
@REM     set "branch=%%B %%C"
@REM     set "directory=%%D"
@REM     echo Cloning repository: !repository! into !directory!
@REM
@REM     rem Go to src folder, create and change to the target directory
@REM     if not exist "%SRC_DIR%\!directory!" mkdir "%SRC_DIR%\!directory!"
@REM     cd "%SRC_DIR%\!directory!"
@REM
@REM     rem Clone the repository
@REM     git clone https://www.github.com/!repository! . !branch!
@REM
@REM     rem Return to the original directory
@REM     cd "%~dp0"
@REM )


for /f "tokens=1,2,3,4" %%A in (%src_file%) do (
    set "repository=%%A"
    set "branch=%%B %%C"
    set "directory=%%D"
    echo Cloning repository: !repository! into !directory!

    rem Go to src folder
    cd "%SRC_DIR%"

@REM     rem Create the target directory including any necessary parent directories
@REM     if not exist "!directory!" mkdir "!directory!"
@REM
@REM     rem Change to the target directory
@REM     cd "!directory!"

    rem Clone the repository
    git clone https://www.github.com/!repository! !branch! !directory!

    rem Return to the original directory
    cd "%~dp0"
)



cd "%SRC_DIR%\deforum"

call pip install -e .

cd "%~dp0"
REM After cloning all src repositories, navigate to src/ComfyUI
cd src
cd ComfyUI
REM Navigate to custom_nodes and clone the ComfyUI-Manager repository
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

REM Traverse back to the root directory
cd "%~dp0"

cd %SCRIPT_DIR%


set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%"
set "TARGET_PATH=%APP_DIR%\run_ainodes.bat"
set "ICON_PATH=%APP_DIR%\ainodes_frontend\qss\icon.ico"
set "SHORTCUT_NAME=%USERPROFILE%\Desktop\aiNodes - Engine.lnk"

set "VBS_SCRIPT=%TEMP%\CreateShortcut.vbs"
>"%VBS_SCRIPT%" (
    echo Set oWS = WScript.CreateObject("WScript.Shell"^)
    echo sLinkFile = "%SHORTCUT_NAME%"
    echo Set oLink = oWS.CreateShortcut(sLinkFile^)
    echo oLink.TargetPath = "%TARGET_PATH%"
    echo oLink.IconLocation = "%ICON_PATH%"
    echo oLink.WorkingDirectory = "%APP_DIR%"
    echo oLink.WindowStyle = 0
    echo oLink.Save
)

cscript //nologo "%VBS_SCRIPT%"

REM Make sure we dont get the pywin32 error
pip uninstall -y pywin32

cls

echo "AiNodes - Engine installation complete, starting."

REM Create a temporary VBScript file
set "VBSFile=%TEMP%\RunHidden.vbs"
echo Set WshShell = CreateObject("WScript.Shell") >> "%VBSFile%"
echo WshShell.Run chr(34) ^& "%SCRIPT_DIR%run_ainodes.bat" ^& chr(34), 0 >> "%VBSFile%"
echo Set WshShell = Nothing >> "%VBSFile%"

REM Execute the VBScript silently
cscript //nologo "%VBSFile%"

REM Delete the temporary VBScript file
del "%VBSFile%"