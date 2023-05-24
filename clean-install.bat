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

endlocal

call download_node_packs.bat
call create_shortcut.bat

python launcher.py --update