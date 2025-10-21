@echo off
REM AI Navigation Assistant Force Stop Script for Windows
REM Forcefully stops all related processes and windows

setlocal enabledelayedexpansion

echo AI Navigation Assistant - FORCE STOP All Services
echo ==================================================
echo WARNING: This will forcefully terminate all related processes
echo.

REM Kill all Python processes (aggressive approach)
echo Force stopping ALL Python processes...
taskkill /f /im python.exe >nul 2>&1

REM Kill uvicorn processes
echo Force stopping uvicorn processes...
taskkill /f /im uvicorn.exe >nul 2>&1

REM Kill any remaining processes on our ports
echo Force stopping processes on ports 8000, 3000, 3001...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000\|:3000\|:3001"') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Close all command windows
echo Closing all command prompt windows...
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq cmd.exe" /fo table /nh 2^>nul') do (
    if not "%%a"=="%~1" (
        taskkill /f /pid %%a >nul 2>&1
    )
)

REM Wait for processes to terminate
timeout /t 3 /nobreak >nul

echo.
echo Force stop completed. All processes should be terminated.
echo If you still see running processes, restart your computer.
echo.
pause