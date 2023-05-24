@echo off
if exist nodes_env rmdir /s /q nodes_env

@echo off
setlocal enabledelayedexpansion

set "repositories_file=repositories.txt"
set "custom_nodes_folder=custom_nodes"

rem Read repositories from repositories.txt
for /f "tokens=1 delims=/" %%A in (%repositories_file%) do (
  set "repository=%%A"

  rem Remove folder corresponding to repository
  set "folder_name=!repository:_=!"
  echo Removing folder: %custom_nodes_folder%\!folder_name!
  if exist %custom_nodes_folder%\!folder_name! (
    rmdir /s /q %custom_nodes_folder%\!folder_name!
  )
)

setlocal

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
del "%VBS_SCRIPT%"


call download_node_packs.bat
call create_shortcut.bat

python launcher.py --update

endlocal
