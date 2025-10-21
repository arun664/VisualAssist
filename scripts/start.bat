@echo off
REM AI Navigation Assistant Startup Script for Windows
REM Supports both development and production modes

setlocal enabledelayedexpansion

REM Default values
set ENVIRONMENT=development
set COMPONENT=all
set VERBOSE=false
set HELP=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="-e" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--environment" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-c" (
    set COMPONENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--component" (
    set COMPONENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-v" (
    set VERBOSE=true
    shift
    goto :parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto :parse_args
)
if "%~1"=="-h" (
    set HELP=true
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    set HELP=true
    shift
    goto :parse_args
)
echo Unknown option: %~1
goto :show_help

:args_done

REM Show help if requested
if "%HELP%"=="true" goto :show_help

REM Validate environment
if not "%ENVIRONMENT%"=="development" if not "%ENVIRONMENT%"=="production" (
    echo Error: Environment must be 'development' or 'production'
    exit /b 1
)

REM Validate component
if not "%COMPONENT%"=="all" if not "%COMPONENT%"=="backend" if not "%COMPONENT%"=="frontend" if not "%COMPONENT%"=="client" (
    echo Error: Component must be 'all', 'backend', 'frontend', or 'client'
    exit /b 1
)

REM Get script directory and project root
set SCRIPT_DIR=%~dp0
for %%i in ("%SCRIPT_DIR%..") do set PROJECT_ROOT=%%~fi

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

echo AI Navigation Assistant Startup
echo ===============================
echo Environment: %ENVIRONMENT%
echo Component: %COMPONENT%
echo Project Root: %PROJECT_ROOT%
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is required but not installed
    exit /b 1
)

REM Create necessary directories
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"
if not exist "%PROJECT_ROOT%\backend\logs" mkdir "%PROJECT_ROOT%\backend\logs"
if not exist "%PROJECT_ROOT%\backend\models" mkdir "%PROJECT_ROOT%\backend\models"

REM Start the requested component(s)
if "%COMPONENT%"=="all" goto :start_all
if "%COMPONENT%"=="backend" goto :start_backend
if "%COMPONENT%"=="frontend" goto :start_frontend
if "%COMPONENT%"=="client" goto :start_client

:start_all
echo Starting all components in separate windows...
echo Project root: %PROJECT_ROOT%
echo Starting backend server...
start "AI Navigation - Backend" cmd /k "cd /d "%PROJECT_ROOT%" && python backend\main.py"
timeout /t 3 /nobreak >nul
echo Starting frontend server...
start "AI Navigation - Frontend" cmd /k "cd /d "%PROJECT_ROOT%" && python scripts\simple_server.py 3000 frontend"
timeout /t 1 /nobreak >nul
echo Starting client server...
start "AI Navigation - Client" cmd /k "cd /d "%PROJECT_ROOT%" && python scripts\simple_server.py 3001 client"
echo.
echo All components started in separate windows:
echo - Backend:  http://localhost:8000
echo - Frontend: http://localhost:3000  
echo - Client:   http://localhost:3001
echo - API Docs: http://localhost:8000/docs
echo.
echo Press any key to close this window...
pause >nul
goto :end

:start_backend
echo Starting backend server...

REM Set environment variables
set ENVIRONMENT=%ENVIRONMENT%

REM Activate virtual environment if it exists
if exist "%PROJECT_ROOT%\backend\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\backend\venv\Scripts\activate.bat"
)

if "%ENVIRONMENT%"=="development" (
    python backend\main.py
) else (
    python backend\main.py
)
goto :end

:start_frontend
echo Starting frontend server...
echo Current directory: %CD%
echo Serving from: %CD%\frontend
python scripts\simple_server.py 3000 frontend
echo Frontend server started
echo Frontend URL: http://localhost:3000
echo.
echo Press any key to close this window...
pause >nul
goto :end

:start_client
echo Starting client server...
echo Current directory: %CD%
echo Serving from: %CD%\client
python scripts\simple_server.py 3001 client
echo Client server started
echo Client URL: http://localhost:3001
echo.
echo Press any key to close this window...
pause >nul
goto :end

:show_help
echo AI Navigation Assistant Startup Script
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo OPTIONS:
echo     -e, --environment ENV    Set environment (development^|production) [default: development]
echo     -c, --component COMP     Start specific component (all^|backend^|frontend^|client) [default: all]
echo     -v, --verbose           Enable verbose output
echo     -h, --help              Show this help message
echo.
echo EXAMPLES:
echo     %~nx0                                    # Start all components in development mode
echo     %~nx0 -e production                      # Start all components in production mode
echo     %~nx0 -c backend                         # Start only backend in development mode
echo     %~nx0 -e production -c backend -v        # Start only backend in production mode with verbose output
echo.
echo COMPONENTS:
echo     all         Start all components (backend, frontend, client)
echo     backend     Start only the backend server
echo     frontend    Start only the frontend server
echo     client      Start only the client server
echo.
echo ENVIRONMENTS:
echo     development Use development configuration with hot reload
echo     production  Use production configuration with process management
echo.
goto :end

:end
endlocal