@echo off
REM =====================================================
REM Check and activate Conda environment "LocalGaze"
REM =====================================================

REM Get the current directory (where this bat is located)
SET "CURRENT_DIR=%~dp0"

REM Environment name
SET "ENV_NAME=LocalGaze"

REM Environment YAML file (assume in the same folder)
SET "ENV_YML=%CURRENT_DIR%environment.yml"

REM Initialize conda (assumes conda is in PATH; if not, add path here)
CALL conda activate

REM Check if environment exists
conda info --envs | findstr /R /C:"%ENV_NAME%" >nul
IF %ERRORLEVEL% EQU 0 (
    ECHO Environment "%ENV_NAME%" already exists. Activating...
) ELSE (
    ECHO Environment "%ENV_NAME%" does not exist. Creating from "%ENV_YML%"...
    CALL conda env create -f "%ENV_YML%" --prefix "%CURRENT_DIR%%ENV_NAME%"
)

REM Activate the environment
CALL conda activate "%CURRENT_DIR%%ENV_NAME%"

ECHO =====================================================
ECHO Environment "%ENV_NAME%" is now active.
ECHO You can now use Python or other tools in this environment.
ECHO =====================================================
PAUSE
