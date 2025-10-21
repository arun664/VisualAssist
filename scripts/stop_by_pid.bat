@echo off
REM AI Navigation Assistant Stop Script using PID files
REM Stops processes using stored PID files

setlocal enabledelayedexpansion

echo AI Navigation Assistant - Stop Services (PID Method)
echo ====================================================

set "PID_DIR=logs"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

REM Stop backend using PID file
if exist "%PID_DIR%\backend.pid" (
    echo Stopping backend server...
    for /f %%a in (%PID_DIR%\backend.pid) do (
        echo   Killing backend process ID: %%a
        taskkill /f /pid %%a >nul 2>&1
        if !errorlevel! equ 0 (
            echo   Backend stopped successfully
        ) else (
            echo   Backend process may have already stopped
        )
    )
    del "%PID_DIR%\backend.pid" >nul 2>&1
) else (
    echo No backend PID file found, trying port method...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
        echo   Killing process on port 8000: %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

REM Stop frontend using PID file
if exist "%PID_DIR%\frontend.pid" (
    echo Stopping frontend server...
    for /f %%a in (%PID_DIR%\frontend.pid) do (
        echo   Killing frontend process ID: %%a
        taskkill /f /pid %%a >nul 2>&1
        if !errorlevel! equ 0 (
            echo   Frontend stopped successfully
        ) else (
            echo   Frontend process may have already stopped
        )
    )
    del "%PID_DIR%\frontend.pid" >nul 2>&1
) else (
    echo No frontend PID file found, trying port method...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
        echo   Killing process on port 3000: %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

REM Stop client using PID file
if exist "%PID_DIR%\client.pid" (
    echo Stopping client server...
    for /f %%a in (%PID_DIR%\client.pid) do (
        echo   Killing client process ID: %%a
        taskkill /f /pid %%a >nul 2>&1
        if !errorlevel! equ 0 (
            echo   Client stopped successfully
        ) else (
            echo   Client process may have already stopped
        )
    )
    del "%PID_DIR%\client.pid" >nul 2>&1
) else (
    echo No client PID file found, trying port method...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3001 ^| findstr LISTENING') do (
        echo   Killing process on port 3001: %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

echo.
echo All services stopped. PID files cleaned up.
echo.
pause