@echo off
echo Starting Ayuda Backend Server...
echo.

REM Add current directory to Python path so app module can be found
set PYTHONPATH=%PYTHONPATH%;%CD%
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause 