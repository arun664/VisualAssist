# AI Navigation Assistant Stop Script for Windows (PowerShell)
# Stops all running components and closes related windows

Write-Host "AI Navigation Assistant - Stop All Services" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Function to stop processes by port
function Stop-ProcessByPort {
    param([int]$Port, [string]$ServiceName)
    
    Write-Host "Stopping $ServiceName (port $Port)..." -ForegroundColor Yellow
    
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "  Stopping process: $($process.Name) (PID: $($process.Id))" -ForegroundColor Gray
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        Write-Host "  No processes found on port $Port" -ForegroundColor Gray
    }
}

# Function to stop processes by name pattern
function Stop-ProcessByPattern {
    param([string]$Pattern, [string]$Description)
    
    Write-Host "Stopping $Description..." -ForegroundColor Yellow
    
    $processes = Get-Process | Where-Object { 
        $_.ProcessName -match $Pattern -or 
        $_.MainWindowTitle -match $Pattern -or
        ($_.CommandLine -and $_.CommandLine -match $Pattern)
    }
    
    foreach ($process in $processes) {
        try {
            Write-Host "  Stopping process: $($process.Name) (PID: $($process.Id))" -ForegroundColor Gray
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
        catch {
            Write-Host "  Failed to stop process: $($process.Name)" -ForegroundColor Red
        }
    }
}

# Stop services by port
Stop-ProcessByPort -Port 8000 -ServiceName "Backend Server"
Stop-ProcessByPort -Port 3000 -ServiceName "Frontend Server"
Stop-ProcessByPort -Port 3001 -ServiceName "Client Server"

# Stop Python processes related to our application
Write-Host "Stopping Python processes related to AI Navigation..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $cmdLine = ""
    try {
        $cmdLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    } catch {}
    
    $cmdLine -match "main\.py|http\.server|uvicorn|fastapi" -or
    $_.MainWindowTitle -match "AI Navigation"
}

foreach ($process in $pythonProcesses) {
    Write-Host "  Stopping Python process: $($process.Name) (PID: $($process.Id))" -ForegroundColor Gray
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}

# Stop uvicorn processes
Write-Host "Stopping uvicorn processes..." -ForegroundColor Yellow
Get-Process uvicorn -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Stopping uvicorn process: PID $($_.Id)" -ForegroundColor Gray
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

# Close command windows with AI Navigation titles
Write-Host "Closing AI Navigation command windows..." -ForegroundColor Yellow
$cmdProcesses = Get-Process cmd -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -match "AI Navigation" -or
    $_.MainWindowTitle -match "start\.bat"
}

foreach ($process in $cmdProcesses) {
    Write-Host "  Closing command window: PID $($process.Id)" -ForegroundColor Gray
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}

# Wait for processes to terminate
Write-Host "Waiting for processes to terminate..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Verify services are stopped
Write-Host ""
Write-Host "Verifying services are stopped..." -ForegroundColor Cyan
$servicesRunning = $false

$port8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
$port3000 = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
$port3001 = Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue

if ($port8000) {
    Write-Host "  WARNING: Backend service may still be running on port 8000" -ForegroundColor Red
    $servicesRunning = $true
} else {
    Write-Host "  Backend service stopped (port 8000)" -ForegroundColor Green
}

if ($port3000) {
    Write-Host "  WARNING: Frontend service may still be running on port 3000" -ForegroundColor Red
    $servicesRunning = $true
} else {
    Write-Host "  Frontend service stopped (port 3000)" -ForegroundColor Green
}

if ($port3001) {
    Write-Host "  WARNING: Client service may still be running on port 3001" -ForegroundColor Red
    $servicesRunning = $true
} else {
    Write-Host "  Client service stopped (port 3001)" -ForegroundColor Green
}

Write-Host ""
if ($servicesRunning) {
    Write-Host "Some services may still be running. You may need to manually close them." -ForegroundColor Red
    Write-Host "Check Task Manager for any remaining Python or uvicorn processes." -ForegroundColor Yellow
} else {
    Write-Host "All AI Navigation Assistant services have been stopped successfully." -ForegroundColor Green
}

Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")