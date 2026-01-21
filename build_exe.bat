@echo off
echo Building POLARIS Hypothesis Agent as standalone executable...
echo This may take several minutes...

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Create the executable
pyinstaller --clean --noconfirm polaris_app.spec

echo.
echo Build complete! Check the 'dist' folder for the executable.
echo You can copy 'POLARIS_Hypothesis_Agent.exe' to your desktop or anywhere else.
pause