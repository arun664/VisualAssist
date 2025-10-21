@echo off
REM Start AI Navigation Assistant with proxy bypass

echo AI Navigation Assistant - No Proxy Mode
echo ========================================

REM Set environment variables to bypass proxy
set HTTP_PROXY=
set HTTPS_PROXY=
set NO_PROXY=localhost,127.0.0.1,*.local

REM Disable proxy for current session
set REQUESTS_CA_BUNDLE=
set CURL_CA_BUNDLE=

echo Proxy settings bypassed for this session
echo Starting servers with direct connection...
echo.

REM Call the regular start script with proxy disabled
call "%~dp0start.bat" %*