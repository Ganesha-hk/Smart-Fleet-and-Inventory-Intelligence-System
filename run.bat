@echo off
set ROOT_DIR=%~dp0

echo Starting Backend...
cd /d "%ROOT_DIR%backend"
start cmd /k "set PYTHONPATH=%ROOT_DIR% && python -m uvicorn app.main:app --reload"

echo Starting Frontend...
cd /d "%ROOT_DIR%frontend"
start cmd /k "npm run dev"
