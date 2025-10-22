@echo off
REM AI Navigation Assistant - Complete Application Docker Build Script (Windows)

setlocal enabledelayedexpansion

echo 🐳 Building AI Navigation Assistant - Complete Application Docker Image
echo ======================================================================

REM Configuration
set IMAGE_NAME=ai-navigation-complete
set TAG=latest
set DOCKERFILE=Dockerfile
set NO_CACHE=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto build_start
if "%~1"=="--tag" (
    set TAG=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--name" (
    set IMAGE_NAME=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--no-cache" (
    set NO_CACHE=--no-cache
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --tag TAG        Set image tag (default: latest)
    echo   --name NAME      Set image name (default: ai-navigation-complete)
    echo   --no-cache       Build without using cache
    echo   --help           Show this help message
    echo.
    echo This builds a complete Docker image containing:
    echo   - Frontend (HTML/CSS/JS client)
    echo   - Backend (Python FastAPI server)
    echo   - Nginx (web server and reverse proxy)
    echo   - All dependencies and configurations
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:build_start
set FULL_IMAGE_NAME=%IMAGE_NAME%:%TAG%

echo 📋 Build Configuration:
echo    Image Name: %FULL_IMAGE_NAME%
echo    Dockerfile: %DOCKERFILE%
echo    Context: %CD%
if "%NO_CACHE%"=="--no-cache" (
    echo    Cache: Disabled
) else (
    echo    Cache: Enabled
)
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not in PATH
    exit /b 1
)

REM Check if Dockerfile exists
if not exist "%DOCKERFILE%" (
    echo ❌ Dockerfile not found: %DOCKERFILE%
    exit /b 1
)

REM Check if required directories exist
if not exist "client" (
    echo ❌ Client directory not found
    exit /b 1
)

if not exist "backend" (
    echo ❌ Backend directory not found
    exit /b 1
)

REM Create docker directory if it doesn't exist
if not exist "docker" mkdir docker

echo 🔍 Pre-build checks:
echo    ✓ Docker available
echo    ✓ Dockerfile exists
echo    ✓ Client directory exists
echo    ✓ Backend directory exists
echo    ✓ Docker config directory ready
echo.

REM Build the Docker image
echo 🔨 Building complete Docker image...
echo Command: docker build %NO_CACHE% -f %DOCKERFILE% -t %FULL_IMAGE_NAME% .
echo.

REM Record build start time
for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set BUILD_START_TIME=%%i

docker build %NO_CACHE% -f "%DOCKERFILE%" -t "%FULL_IMAGE_NAME%" .
if errorlevel 1 (
    echo.
    echo ❌ Docker build failed!
    echo.
    echo 🔍 Troubleshooting tips:
    echo    1. Check Docker daemon is running
    echo    2. Ensure sufficient disk space
    echo    3. Try building with --no-cache flag
    echo    4. Check network connectivity for package downloads
    echo    5. Review build logs above for specific errors
    exit /b 1
)

REM Calculate build duration
for /f %%i in ('powershell -command "Get-Date -UFormat %%s"') do set BUILD_END_TIME=%%i
set /a BUILD_DURATION=%BUILD_END_TIME% - %BUILD_START_TIME%

echo.
echo ✅ Docker image built successfully!
echo    Image: %FULL_IMAGE_NAME%
echo    Build time: %BUILD_DURATION%s

echo.
echo 📊 Image Information:
docker images %IMAGE_NAME%

echo.
echo 🚀 To run the complete application:
echo    docker run -p 80:80 -p 8000:8000 %FULL_IMAGE_NAME%
echo.
echo 🌐 To run on custom port:
echo    docker run -p 3000:80 -p 8001:8000 %FULL_IMAGE_NAME%
echo.
echo 🔍 To run with environment variables:
echo    docker run -p 80:80 -p 8000:8000 -e ENVIRONMENT=development %FULL_IMAGE_NAME%
echo.
echo 🐛 To run interactively for debugging:
echo    docker run -it -p 80:80 -p 8000:8000 %FULL_IMAGE_NAME% /bin/bash
echo.
echo 📋 Application will be available at:
echo    Frontend: http://localhost:80
echo    Backend API: http://localhost:80/api
echo    Health Check: http://localhost:80/health

endlocal