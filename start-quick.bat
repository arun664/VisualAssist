@echo off
REM Quick start script - assumes environment is already set up
echo Starting AI Navigation Assistant Backend...

cd /d "%~dp0backend"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Set development environment
set ENVIRONMENT=development

REM Start server
python main.py