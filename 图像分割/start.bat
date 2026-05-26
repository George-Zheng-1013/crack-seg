@echo off
cd /d "%~dp0"
call conda activate detection 2>nul
echo Starting server at http://localhost:8000
start "" "http://localhost:8000"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
