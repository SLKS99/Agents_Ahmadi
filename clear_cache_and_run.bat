@echo off
echo Clearing Streamlit cache and restarting app...

REM Clear Streamlit cache
streamlit cache clear

REM Change to the app directory
cd /d "%~dp0"

REM Run the app
powershell -ExecutionPolicy Bypass -File run_streamlit.ps1

pause