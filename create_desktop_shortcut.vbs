Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the script directory
strScriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Get desktop path - try multiple methods
strDesktop = WshShell.SpecialFolders("Desktop")
If strDesktop = "" Then
    ' Fallback: use environment variable
    strDesktop = WshShell.ExpandEnvironmentStrings("%USERPROFILE%\Desktop")
End If
If strDesktop = "" Then
    ' Last resort: use shell object
    Set objShell = CreateObject("Shell.Application")
    strDesktop = objShell.Namespace("shell:Desktop").Self.Path
End If

' Create shortcut
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