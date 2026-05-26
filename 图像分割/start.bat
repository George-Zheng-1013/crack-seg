@echo off
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%CD%\.ultralytics
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
call conda activate py312 2>nul
echo Starting server at http://localhost:8000
start "" "http://localhost:8000"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
