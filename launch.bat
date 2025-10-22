@echo off
title AI Navigation Assistant Launcher

echo ==========================================
echo    🚀 AI Navigation Assistant Launcher
echo ==========================================
echo.
echo Choose your setup option:
echo.
echo [1] Full Development Setup (Backend + Frontend + Client)
echo [2] Backend Only (API Server)  
echo [3] Quick Start (assumes environment ready)
echo [4] Stop All Services
echo [5] Exit
echo.
set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" (
    echo.
    echo Starting Full Development Environment...
    call start-dev.bat
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starting Backend Server Only...
    call start-quick.bat
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Quick Start - All Services...
    call start.bat
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Stopping All Services...
    call stop.bat
    goto end
)

if "%choice%"=="5" (
    goto end
)

echo Invalid choice. Please try again.
pause
goto start

:end
echo.
echo Launcher finished.
pause