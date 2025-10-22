@echo off
REM AI Navigation Assistant Backend - Docker Run Script (Windows)

setlocal enabledelayedexpansion

echo 🚀 Running AI Navigation Assistant Backend Docker Container
echo ==========================================================

REM Configuration
set IMAGE_NAME=ai-navigation-backend
set TAG=latest
set CONTAINER_NAME=ai-navigation-backend
set PORT=8000
set ENVIRONMENT=production
set DETACH=false
set INTERACTIVE=false
set BUILD=false
set SHOW_LOGS=false
set STOP_CONTAINER=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto run_start
if "%~1"=="--simple" (
    set TAG=simple
    set CONTAINER_NAME=ai-navigation-backend-simple
    echo 📦 Using simple version
    shift
    goto parse_args
)
if "%~1"=="--dev" (
    set ENVIRONMENT=development
    echo 🔧 Using development environment
    shift
    goto parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--name" (
    set CONTAINER_NAME=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--detach" (
    set DETACH=true
    shift
    goto parse_args
)
if "%~1"=="-d" (
    set DETACH=true
    shift
    goto parse_args
)
if "%~1"=="--interactive" (
    set INTERACTIVE=true
    shift
    goto parse_args
)
if "%~1"=="-it" (
    set INTERACTIVE=true
    shift
    goto parse_args
)
if "%~1"=="--build" (
    set BUILD=true
    shift
    goto parse_args
)
if "%~1"=="--logs" (
    set SHOW_LOGS=true
    shift
    goto parse_args
)
if "%~1"=="--stop" (
    set STOP_CONTAINER=true
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --simple         Use simple version (minimal dependencies)
    echo   --dev            Use development environment
    echo   --port PORT      Set port mapping (default: 8000)
    echo   --name NAME      Set container name
    echo   --detach, -d     Run in detached mode
    echo   --interactive, -it  Run in interactive mode
    echo   --build          Build image before running
    echo   --logs           Show container logs
    echo   --stop           Stop running container
    echo   --help           Show this help message
    echo.
    echo Examples:
    echo   %0                    # Run normally
    echo   %0 --dev --port 8001  # Run in dev mode on port 8001
    echo   %0 --simple -d        # Run simple version detached
    echo   %0 --build            # Build and run
    echo   %0 --logs             # Show logs of running container
    echo   %0 --stop             # Stop running container
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:run_start
set FULL_IMAGE_NAME=%IMAGE_NAME%:%TAG%

REM Stop container if requested
if "%STOP_CONTAINER%"=="true" (
    echo 🛑 Stopping container: %CONTAINER_NAME%
    docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
    if not errorlevel 1 (
        docker stop %CONTAINER_NAME%
        docker rm %CONTAINER_NAME%
        echo ✅ Container stopped and removed
    ) else (
        echo ℹ️  Container %CONTAINER_NAME% is not running
    )
    exit /b 0
)

REM Show logs if requested
if "%SHOW_LOGS%"=="true" (
    echo 📋 Showing logs for container: %CONTAINER_NAME%
    docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
    if not errorlevel 1 (
        docker logs -f %CONTAINER_NAME%
    ) else (
        echo ❌ Container %CONTAINER_NAME% is not running
        exit /b 1
    )
    exit /b 0
)

REM Build image if requested
if "%BUILD%"=="true" (
    echo 🔨 Building image first...
    if "%TAG%"=="simple" (
        call build-docker.bat --simple
    ) else (
        call build-docker.bat
    )
    echo.
)

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not in PATH
    exit /b 1
)

REM Check if image exists
docker images -q %FULL_IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker image not found: %FULL_IMAGE_NAME%
    echo 💡 Build it first with: build-docker.bat
    echo 💡 Or use --build flag to build automatically
    exit /b 1
)

REM Stop existing container if running
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if not errorlevel 1 (
    echo 🔄 Stopping existing container: %CONTAINER_NAME%
    docker stop %CONTAINER_NAME%
    docker rm %CONTAINER_NAME%
)

echo 📋 Run Configuration:
echo    Image: %FULL_IMAGE_NAME%
echo    Container: %CONTAINER_NAME%
echo    Port: %PORT%:8000
echo    Environment: %ENVIRONMENT%
echo.

REM Prepare docker run command
set DOCKER_CMD=docker run

REM Add run options
if "%DETACH%"=="true" (
    set DOCKER_CMD=!DOCKER_CMD! -d
) else if "%INTERACTIVE%"=="true" (
    set DOCKER_CMD=!DOCKER_CMD! -it
) else (
    set DOCKER_CMD=!DOCKER_CMD! -it
)

REM Add port mapping and other options
set DOCKER_CMD=!DOCKER_CMD! --name %CONTAINER_NAME%
set DOCKER_CMD=!DOCKER_CMD! -p %PORT%:8000
set DOCKER_CMD=!DOCKER_CMD! -e ENVIRONMENT=%ENVIRONMENT%
set DOCKER_CMD=!DOCKER_CMD! -e SERVER_HOST=0.0.0.0
set DOCKER_CMD=!DOCKER_CMD! -e SERVER_PORT=8000

REM Add volume mounts for persistence
set DOCKER_CMD=!DOCKER_CMD! -v "%CD%\logs:/app/logs"
set DOCKER_CMD=!DOCKER_CMD! -v "%CD%\models:/app/models"

REM Add image name
set DOCKER_CMD=!DOCKER_CMD! %FULL_IMAGE_NAME%

echo 🚀 Starting container...
echo Command: !DOCKER_CMD!
echo.

REM Run the container
!DOCKER_CMD!
if errorlevel 1 (
    echo.
    echo ❌ Failed to start container!
    exit /b 1
)

echo.
if "%DETACH%"=="true" (
    echo ✅ Container started in detached mode!
    echo    Container: %CONTAINER_NAME%
    echo    URL: http://localhost:%PORT%
    echo.
    echo 📋 Useful commands:
    echo    docker logs -f %CONTAINER_NAME%    # View logs
    echo    docker exec -it %CONTAINER_NAME% /bin/bash  # Access container
    echo    docker stop %CONTAINER_NAME%       # Stop container
    echo    run-docker.bat --logs            # Show logs
    echo    run-docker.bat --stop            # Stop container
) else (
    echo ✅ Container finished running
)

endlocal