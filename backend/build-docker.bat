@echo off
REM AI Navigation Assistant Backend - Docker Build Script (Windows)

setlocal enabledelayedexpansion

echo 🐳 Building AI Navigation Assistant Backend Docker Image
echo =======================================================

REM Configuration
set IMAGE_NAME=ai-navigation-backend
set TAG=latest
set DOCKERFILE=Dockerfile

REM Parse command line arguments
:parse_args
if "%~1"=="" goto build_start
if "%~1"=="--simple" (
    set DOCKERFILE=Dockerfile.simple
    set TAG=simple
    echo 📦 Using simplified Dockerfile
    shift
    goto parse_args
)
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
if "%~1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --simple     Use Dockerfile.simple (minimal dependencies)
    echo   --tag TAG    Set image tag (default: latest)
    echo   --name NAME  Set image name (default: ai-navigation-backend)
    echo   --help       Show this help message
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
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not in PATH
    exit /b 1
)

REM Change to backend directory if not already there
if not exist "%DOCKERFILE%" (
    if exist "backend\%DOCKERFILE%" (
        cd backend
        echo 📁 Changed to backend directory
    ) else (
        echo ❌ Dockerfile not found: %DOCKERFILE%
        exit /b 1
    )
)

REM Build the Docker image
echo 🔨 Building Docker image...
echo Command: docker build -f %DOCKERFILE% -t %FULL_IMAGE_NAME% .
echo.

docker build -f "%DOCKERFILE%" -t "%FULL_IMAGE_NAME%" .
if errorlevel 1 (
    echo.
    echo ❌ Docker build failed!
    exit /b 1
)

echo.
echo ✅ Docker image built successfully!
echo    Image: %FULL_IMAGE_NAME%

echo.
echo 📊 Image Information:
docker images %IMAGE_NAME%

echo.
echo 🚀 To run the container:
echo    docker run -p 8000:8000 %FULL_IMAGE_NAME%
echo.
echo 🔍 To run with environment variables:
echo    docker run -p 8000:8000 -e ENVIRONMENT=development %FULL_IMAGE_NAME%
echo.
echo 🐛 To run interactively for debugging:
echo    docker run -it -p 8000:8000 %FULL_IMAGE_NAME% /bin/bash

endlocal