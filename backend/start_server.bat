@echo off
echo Starting Ayuda Backend Server...
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause 