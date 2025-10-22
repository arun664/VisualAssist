@echo off
REM AI Navigation Assistant - Local Server Startup Script
REM This script starts the FastAPI backend server locally

echo ========================================
echo   AI Navigation Assistant - Backend
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo ✓ Python detected
python --version

REM Navigate to backend directory
cd /d "%~dp0backend"
if not exist "main.py" (
    echo ERROR: main.py not found in backend directory
    echo Please ensure you're running this from the project root
    pause
    exit /b 1
)

echo ✓ Backend directory found

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated

REM Install/update dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check requirements.txt and your internet connection
    pause
    exit /b 1
)

echo ✓ Dependencies installed

REM Set environment variables for local development
set ENVIRONMENT=development
set SERVER_HOST=0.0.0.0
set SERVER_PORT=8000
set DEBUG=true
set RELOAD=true

echo.
echo ========================================
echo   Starting AI Navigation Assistant
echo ========================================
echo   Environment: Development
echo   Host: %SERVER_HOST%
echo   Port: %SERVER_PORT%
echo   Debug: %DEBUG%
echo   Auto-reload: %RELOAD%
echo ========================================
echo.
echo Server will be available at:
echo   Local:    http://localhost:8000
echo   Network:  http://%COMPUTERNAME%:8000
echo.
echo Opening Frontend and Client interfaces...
echo ========================================
echo.

REM Start the FastAPI server in background
echo Starting backend server...
start "AI Navigation Backend" /min python main.py

REM Wait a moment for server to start
timeout /t 3 /nobreak >nul

REM Start a simple HTTP server for frontend in separate terminal
echo Starting frontend server...
cd /d "%~dp0"
start "AI Navigation Frontend" cmd /k "cd frontend && python -m http.server 3000"

REM Wait a moment for frontend server to start
timeout /t 2 /nobreak >nul

REM Start a simple HTTP server for client in separate terminal
echo Starting client server...
start "AI Navigation Client" cmd /k "cd client && python -m http.server 3001"

REM Wait a moment for client server to start
timeout /t 2 /nobreak >nul

REM Open frontend in default browser
echo Opening Frontend in browser...
start http://localhost:3000

REM Wait a moment then open client in new browser window
timeout /t 2 /nobreak >nul
echo Opening Client in browser...
start http://localhost:3001

echo.
echo ========================================
echo   All services started successfully!
echo ========================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo   Client:   http://localhost:3001
echo.
echo Press any key to stop all services...
echo ========================================
pause >nul

REM Stop all services when user presses a key
echo Stopping all services...
taskkill /f /fi "WindowTitle eq AI Navigation Backend*" 2>nul
taskkill /f /fi "WindowTitle eq AI Navigation Frontend*" 2>nul
taskkill /f /fi "WindowTitle eq AI Navigation Client*" 2>nul

REM Pause on exit so user can see any error messages
if errorlevel 1 (
    echo.
    echo ========================================
    echo   Server stopped with errors
    echo ========================================
    pause
) else (
    echo.
    echo ========================================
    echo   Server stopped normally
    echo ========================================
)