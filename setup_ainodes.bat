@echo off

REM Check if the nodes_env folder exists
if exist nodes_env do (
    choice /C YN /M "The 'nodes_env' folder already exists. Do you want to remove it? (Y/N)"
    if errorlevel 2 goto SkipRemove
    rmdir /s /q nodes_env
    :SkipRemove
)

REM Get the full path of the script directory
set "SCRIPT_DIR=%~dp0"

REM Construct the full path to the Python Scripts directory
set "PYTHON_SCRIPTS_DIR=%SCRIPT_DIR%src\python\Scripts"

set "BACKUPPATH=%PATH%"

REM Update the PATH environment variable
set "PATH=%PYTHON_SCRIPTS_DIR%;%PATH%"

REM Run the makesure_pip.py script
%SCRIPT_DIR%src\python\python.exe %SCRIPT_DIR%get-pip.py
%SCRIPT_DIR%src\python\python.exe -m pip install virtualenv


%SCRIPT_DIR%src\python\python.exe -m virtualenv nodes_env

REM Activate the virtual environment
if "%OS%"=="Windows_NT" (
    call nodes_env\Scripts\activate.bat
) else (
    source nodes_env/bin/activate
)

REM Install requirements
pip install -r requirements.txt

setlocal enabledelayedexpansion

set "repositories_file=repositories.txt"
set "custom_nodes_folder=custom_nodes"

rem Read repositories from repositories.txt
for /f "tokens=*" %%A in (%repositories_file%) do (
  set "repository=%%A"
  echo Cloning repository: !repository!

  rem Go to custom_nodes folder and clone the repository
  cd %custom_nodes_folder%
  git clone https://www.github.com/!repository!

  rem Return to the original directory
  cd ..
)

rem Go into each top subdirectory of custom_nodes and run pip install
for /d %%B in (%custom_nodes_folder%\*) do (
  echo Installing requirements in directory: %%B
  if exist "%%B\requirements.txt" (
    pushd "%%B"
    pip install -r requirements.txt
    popd
  )
)

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

REM Clear the screen
cls

REM Notify the user and prompt to run start.bat
echo The setup process is complete.
choice /C YN /M "Do you want to run start.bat? (Y/N)"
if %ERRORLEVEL% equ 1 (
    start %SCRIPT_DIR%run_ainodes.bat
)


