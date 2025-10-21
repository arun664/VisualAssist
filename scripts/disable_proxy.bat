@echo off
REM Disable proxy settings for AI Navigation Assistant development

echo Disabling proxy settings for local development...
echo ================================================

REM Disable system proxy via registry
echo Disabling system proxy settings...
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyEnable /t REG_DWORD /d 0 /f >nul 2>&1

REM Clear proxy server settings
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer /f >nul 2>&1

REM Set proxy bypass for localhost
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyOverride /t REG_SZ /d "localhost;127.0.0.1;*.local" /f >nul 2>&1

REM Disable proxy for Python/pip
echo Configuring Python to bypass proxy...
set HTTP_PROXY=
set HTTPS_PROXY=
set NO_PROXY=localhost,127.0.0.1,*.local

REM Create environment file for proxy bypass
echo Creating proxy bypass configuration...
echo # Proxy bypass configuration for AI Navigation Assistant > .env.proxy
echo HTTP_PROXY= >> .env.proxy
echo HTTPS_PROXY= >> .env.proxy
echo NO_PROXY=localhost,127.0.0.1,*.local >> .env.proxy

REM Flush DNS to ensure clean resolution
echo Flushing DNS cache...
ipconfig /flushdns >nul 2>&1

echo.
echo Proxy settings disabled for local development.
echo Please restart your browser and try accessing:
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8000
echo - Client:   http://localhost:3001
echo.
echo If issues persist, you may need to:
echo 1. Restart your browser completely
echo 2. Check Windows Firewall settings
echo 3. Run as Administrator if needed
echo.
pause