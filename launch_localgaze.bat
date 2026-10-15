@echo off
setlocal
cd /d "%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto create_venv
goto install_runtime

:create_venv
where py >nul 2>nul
if errorlevel 1 goto missing_python
py -3.12 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if not errorlevel 1 goto use_python312
py -3.10 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if not errorlevel 1 goto use_python310
goto missing_python

:use_python312
py -3.12 -m venv .venv
if errorlevel 1 goto setup_failed
goto install_runtime

:use_python310
py -3.10 -m venv .venv
if errorlevel 1 goto setup_failed

:install_runtime
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed
"%VENV_PY%" launch.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%

:missing_python
echo Python 3.10 or 3.12 (64-bit) and the Windows py launcher are required.
echo Install Python from https://www.python.org/downloads/windows/ and run this file again.
pause
exit /b 1

:setup_failed
echo Setup or startup failed. Check your network connection and the message above.
pause
exit /b 1
