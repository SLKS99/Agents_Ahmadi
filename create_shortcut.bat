@echo off
echo Creating desktop shortcut for POLARIS Hypothesis Agent...

REM Run the VBScript to create the shortcut
cscript //nologo create_desktop_shortcut.vbs

echo.
echo Desktop shortcut created successfully!
echo You can now double-click 'POLARIS Hypothesis Agent' on your desktop to start the app.
echo.
pause