# POLARIS Desktop App Options

This document explains how to create desktop applications from your Streamlit app.

## App Versions

- **`streamlit_app_clean.py`** - Clean multi-page version with navigation sidebar (recommended)
- **`streamlit_app.py`** - Regular version (legacy, may be removed in future)

The desktop shortcuts and executable will run the **clean multi-page version** by default.

## Option 1: Simple Desktop Shortcut (Easiest)

### Method A: Run the VBScript
1. Double-click `create_desktop_shortcut.vbs`
2. A shortcut called "POLARIS Hypothesis Agent" will appear on your desktop
3. Double-click the shortcut to run the app

### Method B: Manual Shortcut Creation
1. Right-click on desktop → New → Shortcut
2. Location: `cmd.exe /c "cd /d C:\path\to\polaris_ahmadi && run_app.bat"`
3. Name: "POLARIS Hypothesis Agent"
4. Right-click shortcut → Properties → Change icon (use a Python icon)

## Option 2: Standalone Executable (Professional)

### Build the Executable
1. Run `build_exe.bat` (this may take 5-10 minutes)
2. The executable will be created in the `dist` folder
3. Copy `POLARIS_Hypothesis_Agent.exe` to your desktop

### Requirements for Building
- All Python dependencies installed (`pip install -r requirements.txt`)
- PyInstaller installed (`pip install pyinstaller`)

## Option 3: Windows Service (Always Running)

Use the Task Scheduler method mentioned in the main README for auto-startup.

## Files Created

- `run_app.bat` - Batch file wrapper for PowerShell script
- `create_desktop_shortcut.vbs` - Creates desktop shortcut automatically
- `polaris_app.spec` - PyInstaller configuration
- `build_exe.bat` - Builds standalone executable

## Troubleshooting

### App Won't Start from Shortcut
- Make sure all file paths are correct
- Check that Python is in your PATH
- Try running `run_streamlit.ps1` directly first

### Build Fails
- Ensure all dependencies are installed
- Check that you have enough disk space (build can be 500MB+)
- Some antivirus software may block the executable creation

### Icon Issues
- The VBScript uses a default Windows icon
- For custom icons, add an `icon.ico` file and update the spec file