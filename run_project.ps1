# Smart Fleet Intelligence System - Unified Launcher
# This script starts both the FastAPI backend and Vite frontend concurrently.

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "SETTING UP SMART FLEET INTELLIGENCE SYSTEM (FULL STACK)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Kill any existing processes on ports 8000 (Backend) and 5173 (Frontend)
Write-Host "[1/3] Cleaning up existing ports..." -ForegroundColor Yellow
$ports = @(8000, 5173)
foreach ($port in $ports) {
    $proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $proc.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "      Closed process on port $port"
    }
}

# 2. Start Backend (FastAPI)
Write-Host "[2/3] Starting Backend (FastAPI) on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Normal

# 3. Start Frontend (Vite)
Write-Host "[3/3] Starting Frontend (Vite) on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -WindowStyle Normal

Write-Host "`nSUCCESS: System is booting up!" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000/api/v1"
Write-Host "Frontend Dashboard: http://localhost:5173/fleet"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Press any key to close this window (services will keep running in their own windows)..."
# Pause
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
