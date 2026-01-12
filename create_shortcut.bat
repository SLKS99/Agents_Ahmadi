@echo off
echo Creating desktop shortcut for POLARIS Hypothesis Agent...

REM Get the desktop path
for /f "tokens=*" %%i in ('powershell -command "[Environment]::GetFolderPath('Desktop')"') do set DESKTOP=%%i

REM Create the shortcut using PowerShell
powershell -command "
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('%DESKTOP%\POLARIS Hypothesis Agent.lnk')
$Shortcut.TargetPath = 'cmd.exe'
$Shortcut.Arguments = '/c cd /d ""%~dp0"" && run_app.bat'
$Shortcut.WorkingDirectory = '%~dp0'
$Shortcut.Description = 'POLARIS Hypothesis Agent - Streamlit App'
$Shortcut.IconLocation = 'C:\Windows\System32\SHELL32.dll,13'
$Shortcut.Save()
"

echo Desktop shortcut created successfully!
echo You can now double-click 'POLARIS Hypothesis Agent' on your desktop to start the app.
pause