@echo off
setlocal enabledelayedexpansion

set "base_folder=nodes_env"
set "custom_nodes_folder=custom_nodes"
set "repositories_file=repositories.txt"
set "SCRIPT_DIR=%~dp0"

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
rem Activate virtual environment
call %base_folder%\Scripts\activate.bat

rem Stash and pull changes in the base folder
cd %SCRIPT_DIR%
git pull
pip install -r requirements.txt

rem Process folders in the custom_nodes directory
cd %custom_nodes_folder%

for /d %%F in (*_nodes*) do (
  echo Processing %%F
  cd %%F
  git pull
  pip install -r requirements.txt
  cd ..
)

rem Deactivate virtual environment
call %base_folder%\Scripts\deactivate.bat

endlocal

call %SCRIPT_DIR%\run_ainodes.bat
