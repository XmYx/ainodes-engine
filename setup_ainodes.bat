REM Set variables
set "PYTHON_ZIP_URL=https://www.python.org/ftp/python/3.10.10/python-3.10.10-embed-amd64.zip"
set "PYTHON_ZIP_FILE=%~dp0src\python\python-3.10.10-embed-amd64.zip"
set "PYTHON_EXTRACT_PATH=%~dp0src\python"
REM Get the full path of the script directory
set "SCRIPT_DIR=%~dp0"
REM Construct the full path to the Python Scripts directory
set "PYTHON_DIR=%SCRIPT_DIR%src\python"
set "PYTHON_LIB_DIR=%PYTHON_DIR%\Lib"
set "PYTHON_SCRIPTS_DIR=%PYTHON_DIR%\Scripts"
set "BACKUPPATH=%PATH%"

REM Check if Python executable exists
if not exist "%PYTHON_EXTRACT_PATH%\python.exe" (
    echo Python executable not found. Downloading Python embeddable package...
    mkdir "%PYTHON_EXTRACT_PATH%"
    curl -o "%PYTHON_ZIP_FILE%" "%PYTHON_ZIP_URL%"
    echo Extracting Python embeddable package...
    powershell -Command "Expand-Archive -Path '%PYTHON_ZIP_FILE%' -DestinationPath '%PYTHON_EXTRACT_PATH%' -Force"
)

REM Install virtualenv using pip
pip install virtualenv

REM Update the PATH environment variable
set "PATH=%PYTHON_SCRIPTS_DIR%;%PYTHON_LIB_DIR%;%PYTHON_DIR%;%PATH%"

REM Install pip
call "%PYTHON_DIR%\python.exe" "%SCRIPT_DIR%get-pip.py"

REM Activate the virtual environment
call "%PYTHON_SCRIPTS_DIR%\activate.bat"

REM Install virtualenv using pip
pip install virtualenv

REM Create virtual environment
virtualenv nodes_env

REM Activate the virtual environment
call nodes_env\Scripts\activate.bat

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

REM Make sure we dont get the pywin32 error
pip uninstall -y pywin32

REM Clear the screen
cls

REM Notify the user and prompt to run start.bat
echo The setup process is complete.
choice /C YN /M "Do you want to run start.bat? (Y/N)"
if %ERRORLEVEL% equ 1 (
    start %SCRIPT_DIR%run_ainodes.bat
)

REM Restore the original PATH environment variable
set "PATH=%PATH%;%BACKUPPATH%"
