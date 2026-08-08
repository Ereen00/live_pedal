@echo off
REM Launch live_pedal with the project's virtualenv, whatever Python is on PATH.
REM
REM   run.bat                      default rig
REM   run.bat -c lead              a preset
REM   run.bat --list-devices       any run.py argument works

setlocal
set "HERE=%~dp0"

if exist "%HERE%.venv\Scripts\python.exe" (
    "%HERE%.venv\Scripts\python.exe" "%HERE%run.py" %*
) else (
    echo No virtualenv found at %HERE%.venv
    echo.
    echo Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo     python tools/fetch_model.py
    exit /b 1
)

endlocal
