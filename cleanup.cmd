@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0cleanup.ps1"
set "cleanup_exit=%errorlevel%"
echo.
pause
exit /b %cleanup_exit%
