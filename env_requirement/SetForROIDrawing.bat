@echo off
REM =====================================================
REM Check, create, and activate Conda environment "ROIDrawing"
REM Then install dependencies via pip using domestic mirror
REM =====================================================

REM Get the current directory (where this bat is located)
SET "CURRENT_DIR=%~dp0"

REM Environment name
SET "ENV_NAME=ROIDrawing"

REM Python version
SET "PYTHON_VER=3.13"

REM Requirements file
SET "REQ_FILE=%CURRENT_DIR%requirements_ROIDrawing.txt"

REM Initialize conda (ensure conda is in PATH)
CALL conda activate >nul 2>&1

REM Check if environment exists
ECHO Checking if environment "%ENV_NAME%" exists...
conda info --envs | findstr /R /C:"%ENV_NAME%" >nul
IF %ERRORLEVEL% EQU 0 (
    ECHO  Environment "%ENV_NAME%" already exists.
) ELSE (
    ECHO  Environment "%ENV_NAME%" does not exist. Creating with Python %PYTHON_VER%...
    CALL conda create -y -n "%ENV_NAME%" python=%PYTHON_VER%
    IF %ERRORLEVEL% NEQ 0 (
        ECHO  Failed to create environment.
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
    ECHO  No "%REQ_FILE%" found in "%CURRENT_DIR%".
)

ECHO =====================================================
ECHO Environment "%ENV_NAME%" is now ready.
ECHO You can now use Python or run your scripts.
ECHO =====================================================
PAUSE
