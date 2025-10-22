@echo off
REM AI Navigation Assistant - Complete Development Environment Setup
REM Starts backend, frontend, and client with auto-opening in browser

echo =========================================
echo   AI Navigation Assistant - Full Setup
echo =========================================
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
echo.

REM Navigate to project root
cd /d "%~dp0"

REM Setup backend environment
echo [1/4] Setting up Backend Environment...
cd backend
if not exist "venv" (
    echo Creating virtual environment for backend...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo ✓ Backend environment ready
cd ..

echo.
echo [2/4] Starting Backend Server...
REM Start backend server in minimized window
start "AI Navigation Backend" /min cmd /c "cd backend && call venv\Scripts\activate.bat && set ENVIRONMENT=development && python main.py"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Check if backend is responding
:check_backend
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo Waiting for backend to start...
    timeout /t 2 /nobreak >nul
    goto check_backend
)
echo ✓ Backend server started successfully

echo.
echo [3/4] Starting Frontend & Client Servers...

REM Start frontend server
start "AI Navigation Frontend" cmd /k "echo AI Navigation Frontend Server && echo Available at: http://localhost:3000 && echo. && cd frontend && python -m http.server 3000"

REM Start client server  
start "AI Navigation Client" cmd /k "echo AI Navigation Client Server && echo Available at: http://localhost:3001 && echo. && cd client && python -m http.server 3001"

REM Wait for servers to start
timeout /t 3 /nobreak >nul

echo ✓ Frontend server started on http://localhost:3000
echo ✓ Client server started on http://localhost:3001

echo.
echo [4/4] Opening Web Interfaces...

REM Open frontend in default browser
echo Opening Frontend interface...
start http://localhost:3000
timeout /t 2 /nobreak >nul

REM Open client in new browser window/tab
echo Opening Client interface...
start http://localhost:3001

echo.
echo =========================================
echo   🚀 All Services Running Successfully!
echo =========================================
echo.
echo   Backend API:     http://localhost:8000
echo   Frontend UI:     http://localhost:3000  
echo   Client Device:   http://localhost:3001
echo.
echo   Health Check:    http://localhost:8000/health
echo   API Docs:        http://localhost:8000/docs
echo.
echo =========================================
echo   Press any key to STOP all services
echo =========================================
pause >nul

REM Cleanup - Stop all services
echo.
echo Stopping all services...
echo ✓ Stopping backend server...
for /f "tokens=2" %%i in ('tasklist /fi "WindowTitle eq AI Navigation Backend*" /fo csv /nh 2^>nul') do taskkill /f /pid %%i 2>nul

echo ✓ Stopping frontend server...
for /f "tokens=2" %%i in ('tasklist /fi "WindowTitle eq AI Navigation Frontend*" /fo csv /nh 2^>nul') do taskkill /f /pid %%i 2>nul

echo ✓ Stopping client server...
for /f "tokens=2" %%i in ('tasklist /fi "WindowTitle eq AI Navigation Client*" /fo csv /nh 2^>nul') do taskkill /f /pid %%i 2>nul

REM Also kill any remaining Python HTTP servers on these ports
netstat -ano | findstr :8000 >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /f /pid %%a 2>nul
)
netstat -ano | findstr :3000 >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do taskkill /f /pid %%a 2>nul
)
netstat -ano | findstr :3001 >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3001') do taskkill /f /pid %%a 2>nul
)

echo.
echo =========================================
echo   All services stopped successfully
echo =========================================
echo.