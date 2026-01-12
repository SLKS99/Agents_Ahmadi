Set WshShell = CreateObject("WScript.Shell")
Set objShell = CreateObject("Shell.Application")

' Get the script directory
strScriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Create shortcut on desktop using WshShell
strDesktop = WshShell.SpecialFolders("Desktop")
Set objShortCut = WshShell.CreateShortcut(strDesktop & "\POLARIS Hypothesis Agent.lnk")

' Configure shortcut
objShortCut.TargetPath = "cmd.exe"
objShortCut.Arguments = "/c ""cd /d " & strScriptDir & " && run_app.bat"""
objShortCut.WorkingDirectory = strScriptDir
objShortCut.Description = "POLARIS Hypothesis Agent - Streamlit App"
objShortCut.IconLocation = "C:\Windows\System32\SHELL32.dll,13" ' Python-like icon
objShortCut.WindowStyle = 1 ' Normal window
objShortCut.HotKey = "CTRL+ALT+P" ' Optional hotkey

' Save shortcut
objShortCut.Save

MsgBox "Desktop shortcut created! You can now double-click 'POLARIS Hypothesis Agent' on your desktop to start the app.", vbInformation, "Shortcut Created"