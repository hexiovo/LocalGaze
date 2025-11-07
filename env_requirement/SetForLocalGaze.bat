@echo off
REM =====================================================
REM Check, create, and activate Conda environment "LocalGaze"
REM Then install dependencies via pip using domestic mirror
REM =====================================================

REM Get the current directory (where this bat is located)
SET "CURRENT_DIR=%~dp0"

REM Environment name
SET "ENV_NAME=LocalGaze"

REM Environment YAML file (assume in the same folder)
SET "ENV_YML=%CURRENT_DIR%environment.yml"

REM Requirements file
SET "REQ_FILE=%CURRENT_DIR%requirements.txt"

REM Initialize conda (ensure conda is in PATH)
CALL conda activate >nul 2>&1

REM Check if environment exists
ECHO Checking if environment "%ENV_NAME%" exists...
conda info --envs | findstr /R /C:"%ENV_NAME%" >nul
IF %ERRORLEVEL% EQU 0 (
    ECHO  Environment "%ENV_NAME%" already exists.
) ELSE (
    ECHO  Environment "%ENV_NAME%" does not exist. Creating from "%ENV_YML%"...
    CALL conda env create -f "%ENV_YML%"
    IF %ERRORLEVEL% NEQ 0 (
        ECHO  Failed to create environment. Please check "%ENV_YML%".
        PAUSE
        EXIT /B 1
    )
)

REM Activate the environment
ECHO Activating environment "%ENV_NAME%"...
CALL conda activate "%ENV_NAME%"
IF %ERRORLEVEL% NEQ 0 (
    ECHO  Failed to activate environment "%ENV_NAME%".
    PAUSE
    EXIT /B 1
)

REM Install dependencies using pip and domestic mirror
IF EXIST "%REQ_FILE%" (
    ECHO Installing dependencies from "%REQ_FILE%" using Tsinghua mirror...
    python -m pip install --upgrade pip
    pip install -r "%REQ_FILE%" -i https://pypi.tuna.tsinghua.edu.cn/simple
    IF %ERRORLEVEL% NEQ 0 (
        ECHO  Some dependencies may have failed to install.
    ) ELSE (
        ECHO  All dependencies installed successfully.
    )
) ELSE (
    ECHO  No "requirements.txt" file found in "%CURRENT_DIR%".
)

ECHO =====================================================
ECHO Environment "%ENV_NAME%" is now ready.
ECHO You can now use Python or run your scripts.
ECHO =====================================================
PAUSE
