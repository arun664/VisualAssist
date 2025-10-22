@echo off
REM Stop AI Navigation Assistant Backend Server
echo Stopping AI Navigation Assistant Backend...

REM Find and kill Python processes running main.py
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo table /nh ^| findstr main.py') do (
    echo Stopping process %%i...
    taskkill /pid %%i /f
)

REM Also kill uvicorn processes
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo table /nh ^| findstr uvicorn') do (
    echo Stopping uvicorn process %%i...
    taskkill /pid %%i /f
)

echo Backend server stopped.
pause