@echo off
cd /d "%~dp0"
echo Iris UI  ->  http://localhost:8000
python -m uvicorn backend.server:app --port 8000
