@echo off
REM AI Navigation Assistant Stop Script for Windows
REM Stops all running components and closes related windows

setlocal enabledelayedexpansion

echo AI Navigation Assistant - Stop All Services
echo =============================================

REM Stop processes by port
echo Stopping backend server (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo   Killing process ID: %%a
    taskkill /f /pid %%a >nul 2>&1
)

echo Stopping frontend server (port 3000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
    echo   Killing process ID: %%a
    taskkill /f /pid %%a >nul 2>&1
)

echo Stopping client server (port 3001)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3001 ^| findstr LISTENING') do (
    echo   Killing process ID: %%a
    taskkill /f /pid %%a >nul 2>&1
)

REM Stop Python processes that might be running AI Navigation components
echo Stopping Python processes related to AI Navigation...
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo table /nh 2^>nul') do (
    for /f "tokens=*" %%b in ('wmic process where "processid=%%a" get commandline /value 2^>nul ^| findstr "main.py\|http.server"') do (
        echo   Stopping Python process: %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

REM Stop uvicorn processes
echo Stopping uvicorn processes...
taskkill /f /im "uvicorn.exe" >nul 2>&1
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo table /nh 2^>nul') do (
    for /f "tokens=*" %%b in ('wmic process where "processid=%%a" get commandline /value 2^>nul ^| findstr "uvicorn"') do (
        echo   Stopping uvicorn process: %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

REM Close command windows with AI Navigation titles
echo Closing AI Navigation command windows...
taskkill /f /fi "WINDOWTITLE eq AI Navigation - Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq AI Navigation - Frontend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq AI Navigation - Client*" >nul 2>&1

REM Alternative method to close cmd windows running our scripts
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq cmd.exe" /fo table /nh 2^>nul') do (
    for /f "tokens=*" %%b in ('wmic process where "processid=%%a" get commandline /value 2^>nul ^| findstr "start.bat\|main.py\|http.server"') do (
        echo   Closing command window: %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

REM Wait a moment for processes to terminate
timeout /t 2 /nobreak >nul

REM Verify services are stopped
echo.
echo Verifying services are stopped...
set "services_running=false"

netstat -aon | findstr :8000 | findstr LISTENING >nul 2>&1
if !errorlevel! equ 0 (
    echo   WARNING: Backend service may still be running on port 8000
    set "services_running=true"
) else (
    echo   Backend service stopped (port 8000)
)

netstat -aon | findstr :3000 | findstr LISTENING >nul 2>&1
if !errorlevel! equ 0 (
    echo   WARNING: Frontend service may still be running on port 3000
    set "services_running=true"
) else (
    echo   Frontend service stopped (port 3000)
)

netstat -aon | findstr :3001 | findstr LISTENING >nul 2>&1
if !errorlevel! equ 0 (
    echo   WARNING: Client service may still be running on port 3001
    set "services_running=true"
) else (
    echo   Client service stopped (port 3001)
)

echo.
if "!services_running!"=="true" (
    echo Some services may still be running. You may need to manually close them.
    echo Check Task Manager for any remaining Python or uvicorn processes.
) else (
    echo All AI Navigation Assistant services have been stopped successfully.
)

echo.
echo Press any key to close this window...
pause >nul