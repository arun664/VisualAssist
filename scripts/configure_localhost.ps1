# Configure Windows to allow direct localhost access
# Run this script as Administrator for best results

Write-Host "Configuring Windows for localhost development..." -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Host "Warning: Not running as Administrator. Some changes may not apply." -ForegroundColor Yellow
    Write-Host "For best results, run PowerShell as Administrator and re-run this script." -ForegroundColor Yellow
    Write-Host ""
}

try {
    # Disable proxy for Internet Explorer/Edge (affects system-wide settings)
    Write-Host "Disabling system proxy settings..." -ForegroundColor Yellow
    
    $regPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"
    Set-ItemProperty -Path $regPath -Name "ProxyEnable" -Value 0 -ErrorAction SilentlyContinue
    Remove-ItemProperty -Path $regPath -Name "ProxyServer" -ErrorAction SilentlyContinue
    Set-ItemProperty -Path $regPath -Name "ProxyOverride" -Value "localhost;127.0.0.1;*.local" -ErrorAction SilentlyContinue
    
    Write-Host "✓ System proxy disabled" -ForegroundColor Green

    # Configure Windows Firewall to allow Python HTTP servers
    Write-Host "Configuring Windows Firewall..." -ForegroundColor Yellow
    
    # Allow Python through firewall
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pythonPath) {
        netsh advfirewall firewall delete rule name="Python HTTP Server" >$null 2>&1
        netsh advfirewall firewall add rule name="Python HTTP Server" dir=in action=allow program="$pythonPath" >$null 2>&1
        Write-Host "✓ Python allowed through firewall" -ForegroundColor Green
    }
    
    # Allow specific ports
    $ports = @(3000, 3001, 8000)
    foreach ($port in $ports) {
        netsh advfirewall firewall delete rule name="AI Navigation Port $port" >$null 2>&1
        netsh advfirewall firewall add rule name="AI Navigation Port $port" dir=in action=allow protocol=TCP localport=$port >$null 2>&1
        Write-Host "✓ Port $port allowed through firewall" -ForegroundColor Green
    }

    # Flush DNS cache
    Write-Host "Flushing DNS cache..." -ForegroundColor Yellow
    ipconfig /flushdns >$null 2>&1
    Write-Host "✓ DNS cache flushed" -ForegroundColor Green

    # Set environment variables for current session
    Write-Host "Setting environment variables..." -ForegroundColor Yellow
    $env:HTTP_PROXY = ""
    $env:HTTPS_PROXY = ""
    $env:NO_PROXY = "localhost,127.0.0.1,*.local"
    Write-Host "✓ Environment variables set" -ForegroundColor Green

    # Test localhost connectivity
    Write-Host "Testing localhost connectivity..." -ForegroundColor Yellow
    
    $testResults = @()
    
    # Test basic localhost resolution
    try {
        $result = Test-NetConnection -ComputerName "localhost" -Port 80 -InformationLevel Quiet -WarningAction SilentlyContinue
        $testResults += "Localhost resolution: $(if($result) {'✓ OK'} else {'✗ Failed'})"
    } catch {
        $testResults += "Localhost resolution: ✗ Failed"
    }
    
    # Test 127.0.0.1 resolution
    try {
        $result = Test-NetConnection -ComputerName "127.0.0.1" -Port 80 -InformationLevel Quiet -WarningAction SilentlyContinue
        $testResults += "127.0.0.1 resolution: $(if($result) {'✓ OK'} else {'✗ Failed'})"
    } catch {
        $testResults += "127.0.0.1 resolution: ✗ Failed"
    }
    
    foreach ($test in $testResults) {
        if ($test -match "✓") {
            Write-Host $test -ForegroundColor Green
        } else {
            Write-Host $test -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "Configuration completed!" -ForegroundColor Green
    Write-Host "You can now start the AI Navigation Assistant with:" -ForegroundColor Cyan
    Write-Host "  scripts\start_no_proxy.bat" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use the regular start script:" -ForegroundColor Cyan
    Write-Host "  scripts\start.bat" -ForegroundColor White
    Write-Host ""
    Write-Host "Access URLs:" -ForegroundColor Cyan
    Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
    Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
    Write-Host "  Client:   http://localhost:3001" -ForegroundColor White

} catch {
    Write-Host "Error during configuration: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Try running this script as Administrator for full functionality." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")