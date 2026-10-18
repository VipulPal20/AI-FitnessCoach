@echo off
setlocal

rem FitnessCoach one-click Windows launcher.
rem Double-click this file from File Explorer.

cd /d "%~dp0"
set "APP_ROOT=%~dp0"
set "PYTHON=%APP_ROOT%.venv\Scripts\python.exe"
set "HOST=127.0.0.1"
set "PORT=8001"
set "URL=http://%HOST%:%PORT%/"
set "HEALTH=%URL%api/v1/health"

rem Already running? Just open the dashboard.
curl.exe -s -f "%HEALTH%" >nul 2>&1 && goto open

if not exist "%PYTHON%" (
    echo [ERROR] Python virtual environment was not found:
    echo         %PYTHON%
    echo.
    echo Create the environment and install requirements first:
    echo         python -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "%APP_ROOT%models\exercise_starter_v3.joblib" (
    echo [WARNING] The v3 exercise model is not present at:
    echo          %APP_ROOT%models\exercise_starter_v3.joblib
    echo          The site will open, but AI health may show Degraded until the model is trained.
    echo.
)

echo Starting FitnessCoach at %URL%
echo Close the FitnessCoach Server window to stop the application.
start "FitnessCoach Server" cmd /k ""%PYTHON%" -m uvicorn app:app --host %HOST% --port %PORT%"

rem Loading MediaPipe and the model takes a while on first start; wait for /health.
echo Waiting for the AI engine to load (up to 2 minutes)...
for /l %%i in (1,1,120) do (
    curl.exe -s -f "%HEALTH%" >nul 2>&1 && goto open
    "%SystemRoot%\System32\timeout.exe" /t 1 /nobreak >nul
)
echo [WARNING] The server did not report healthy yet. Check the "FitnessCoach Server" window for errors.

:open
start "" "%URL%"
endlocal
