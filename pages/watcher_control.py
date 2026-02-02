"""
Watcher Control Page - Start/stop the watcher server and monitor its status.
"""

import streamlit as st
import requests
import os
import subprocess
import sys
import signal
import threading
import time
from pathlib import Path
from datetime import datetime

from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("👀 Watcher Control")
st.markdown("Control the file system watcher that automatically triggers curve fitting when files are uploaded.")

tabs = st.tabs(["Configuration", "Server", "Watching", "Logs", "Directory", "Help"])

with tabs[0]:
    # Watcher configuration (moved from Settings)
    st.markdown("##### ⚙️ Watcher Configuration")
    st.markdown("Configure the watcher settings used by the server and observer.")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        watcher_directory = st.text_input(
            "Watcher Directory:",
            value=st.session_state.get("watcher_directory", r"C:\Users\shery\OneDrive - University of Tennessee\Data Experiment files"),
            help="Full path to directory that the watcher should monitor for file changes. Do not include quotes.",
            key="watcher_dir_input"
        )
        watcher_directory = watcher_directory.strip().strip('"').strip("'")
        st.session_state.watcher_directory = watcher_directory

        watcher_results_dir = st.text_input(
            "Results Directory:",
            value=st.session_state.get("watcher_results_dir", "results"),
            help="Directory where curve fitting results are saved",
            key="watcher_results_dir_input"
        )
        st.session_state.watcher_results_dir = watcher_results_dir

    with col_cfg2:
        watcher_enabled = st.checkbox(
            "Enable Watcher",
            value=st.session_state.get("watcher_enabled", False),
            help="Enable automatic file system watching for workflow automation",
            key="watcher_enabled_input"
        )
        st.session_state.watcher_enabled = watcher_enabled

        watcher_port = st.number_input(
            "Watcher Server Port:",
            min_value=1000,
            max_value=9999,
            value=st.session_state.get("watcher_port", 8000),
            help="Port for the watcher HTTP server",
            key="watcher_port_input"
        )
        st.session_state.watcher_port = watcher_port

    st.divider()

    # Get watcher configuration from settings
    watcher_enabled = st.session_state.get("watcher_enabled", False)
    watcher_directory = st.session_state.get("watcher_directory", r"C:\Users\shery\OneDrive - University of Tennessee\Data Experiment files")
    # Strip quotes if present
    watcher_directory = watcher_directory.strip().strip('"').strip("'")
    watcher_port = st.session_state.get("watcher_port", 8000)
    watcher_server_url = f"http://localhost:{watcher_port}"

    st.info(f"**Watcher Server URL:** `{watcher_server_url}`")
    st.info(f"**Watch Directory:** `{watcher_directory}`")

    # Check if directory exists
    watch_path = Path(watcher_directory).expanduser()
    if watch_path.exists() and watch_path.is_dir():
        st.success("Watch directory exists and is accessible")
        try:
            file_count = len(list(watch_path.glob("*")))
            st.caption(f"Directory contains {file_count} items")
        except:
            pass
    else:
        st.warning(f"Watch directory does not exist or is not accessible: `{watcher_directory}`")
        st.info("**Troubleshooting:**")
        st.info("1. Check the path in Settings → Experiment tab")
        st.info("2. Make sure the path doesn't have quotes around it")
        st.info("3. Verify OneDrive is synced and the folder exists locally")
        st.info("4. Check file permissions for the directory")

# Function to read log file
def read_log_file(log_file_path):
    """Read the log file and return its contents."""
    try:
        if log_file_path.exists():
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        return ""
    except Exception as e:
        return f"Error reading log file: {str(e)}"

# Function to start watcher server as subprocess
def start_watcher_server():
    """Start the watcher server as a background subprocess."""
    try:
        # Check if already running
        try:
            response = requests.get(f"{watcher_server_url}/health", timeout=1)
            if response.status_code == 200:
                return True, "Server is already running"
        except:
            pass
        
        # Get API key from session state
        api_key = st.session_state.get("api_key", "")
        if not api_key:
            return False, "No API key found. Please set your API key in Settings → General tab."
        
        # Get the project root directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # Go up from pages/ to polaris_ahmadi/
        
        # Create logs directory if it doesn't exist
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"watcher_{timestamp}.log"
        
        # Prepare environment with API key
        env = os.environ.copy()
        env["GEMINI_API_KEY"] = api_key
        env["GOOGLE_API_KEY"] = api_key
        
        # Start the watcher server as a subprocess
        # Use CREATE_NEW_PROCESS_GROUP on Windows to allow proper termination
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        # Open log file for writing
        log_file_handle = open(log_file, 'w', encoding='utf-8')
        
        process = subprocess.Popen(
            [sys.executable, "-m", "watcher.server"],
            cwd=str(project_root),
            env=env,
            stdout=log_file_handle,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            creationflags=creation_flags,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Store process handle and log file path in session state
        st.session_state.watcher_server_process = process
        st.session_state.watcher_server_pid = process.pid
        st.session_state.watcher_log_file = str(log_file)
        st.session_state.watcher_log_file_handle = log_file_handle
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Verify it's running
        try:
            response = requests.get(f"{watcher_server_url}/health", timeout=2)
            if response.status_code == 200:
                return True, f"Server started successfully (PID: {process.pid})"
            else:
                return False, "Server started but health check failed"
        except:
            return False, "Server process started but not responding yet. Try refreshing."
            
    except Exception as e:
        return False, f"Failed to start server: {str(e)}"

# Function to stop watcher server
def stop_watcher_server():
    """Stop the watcher server subprocess."""
    try:
        # First try graceful shutdown via API
        try:
            response = requests.post(f"{watcher_server_url}/watch/stop", timeout=2)
        except:
            pass
        
        # Also try to kill any process using port 8000 (Windows)
        if sys.platform == "win32":
            try:
                import subprocess as sp
                # Find process using port 8000
                result = sp.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if ':8000' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) > 0:
                            pid = parts[-1]
                            try:
                                sp.run(["taskkill", "/F", "/PID", pid], timeout=5, capture_output=True)
                                logger.info(f"Killed process {pid} using port 8000")
                            except:
                                pass
            except Exception as e:
                logger.debug(f"Could not kill process on port 8000: {e}")
        
        # Close log file handle if open
        if "watcher_log_file_handle" in st.session_state:
            try:
                st.session_state.watcher_log_file_handle.close()
            except:
                pass
            del st.session_state.watcher_log_file_handle
        
        # Then terminate the process if we have a handle
        if "watcher_server_process" in st.session_state:
            process = st.session_state.watcher_server_process
            try:
                if sys.platform == "win32":
                    # On Windows, use terminate() which sends SIGTERM
                    process.terminate()
                else:
                    # On Unix, send SIGTERM
                    process.send_signal(signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    process.kill()
                    process.wait()
                
                del st.session_state.watcher_server_process
                if "watcher_server_pid" in st.session_state:
                    del st.session_state.watcher_server_pid
                if "watcher_log_file" in st.session_state:
                    del st.session_state.watcher_log_file
                
                return True, "Server stopped successfully"
            except Exception as e:
                return False, f"Error stopping process: {str(e)}"
        else:
            # Check if server is running on port 8000
            try:
                response = requests.get(f"{watcher_server_url}/health", timeout=1)
                if response.status_code == 200:
                    return False, "Server is running but not managed by this session. Stop it manually or restart Streamlit."
            except:
                pass
            return False, "No server process found in session state"
    except Exception as e:
        return False, f"Failed to stop server: {str(e)}"

with tabs[1]:
    # Check if watcher server is running
    server_running = False
    server_status = "Unknown"

    try:
        response = requests.get(f"{watcher_server_url}/health", timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            server_running = True
            st.session_state.watcher_server_running = True
            st.success("Watcher server is running")
            
            observer_running = health_data.get("observer_running", False)
            watch_dir = health_data.get("watch_dir")
            
            if observer_running:
                st.success("File system observer is ACTIVE - Ready to detect files!")
                if watch_dir:
                    st.info(f"Watching directory: `{watch_dir}`")
                    st.caption("Add CSV/Excel files to this directory to trigger curve fitting")
            else:
                st.warning("Observer is NOT running")
                st.info("Click 'Start Watching' in the Watching tab to begin monitoring your directory")
                if watch_dir:
                    st.caption(f"Will watch: `{watch_dir}`")
            
            server_status = "running"
        else:
            st.error("Watcher server responded with error")
            st.session_state.watcher_server_running = False
            server_status = "error"
    except requests.exceptions.RequestException:
        server_running = False
        st.session_state.watcher_server_running = False
        server_status = "stopped"
        
        # Check if we have a process handle but server isn't responding
        if "watcher_server_process" in st.session_state:
            process = st.session_state.watcher_server_process
            if process.poll() is None:
                # Process is still running but server not responding
                st.warning("Server process exists but not responding. Try stopping and restarting.")
            else:
                # Process has died
                st.warning("Server process has stopped. You can start it again below.")
                if "watcher_server_process" in st.session_state:
                    del st.session_state.watcher_server_process
                if "watcher_server_pid" in st.session_state:
                    del st.session_state.watcher_server_pid
        st.info("Server is not running. Use the buttons below to start it.")

    st.divider()

    # Server Control Section
    st.markdown("##### 🖥️ Server Control")

    col_server1, col_server2, col_server3 = st.columns(3)

    with col_server1:
        if server_running:
            st.button("Server Running", use_container_width=True, disabled=True)
        else:
            if st.button("Start Server", use_container_width=True, type="primary", key="start_server_btn"):
                # Check if port is already in use
                port_in_use = False
                try:
                    test_response = requests.get(f"{watcher_server_url}/health", timeout=1)
                    if test_response.status_code == 200:
                        st.warning("Port 8000 is already in use. Server may already be running.")
                        st.info("Try clicking 'Refresh Status' to see if server is running.")
                        port_in_use = True
                except:
                    pass  # Port appears free, continue
                
                if not port_in_use:
                    success, message = start_watcher_server()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                        if "10048" in str(message) or "already in use" in str(message).lower() or "socket" in str(message).lower():
                            st.warning("**Port 8000 is already in use.**")
                            st.info("**Solutions:**")
                            st.info("1. Stop the existing server using 'Stop Server' button")
                            st.info("2. Or manually kill the process: Open PowerShell and run:")
                            st.code("netstat -ano | findstr :8000\ntaskkill /F /PID <PID>", language="bash")
                            st.info("3. Or restart Streamlit to clear the session")
                    if "10048" in str(message) or "already in use" in str(message).lower() or "socket" in str(message).lower():
                        st.warning("**Port 8000 is already in use.**")
                        st.info("**Solutions:**")
                        st.info("1. Stop the existing server using 'Stop Server' button")
                        st.info("2. Or manually kill the process: Open PowerShell and run:")
                        st.code("netstat -ano | findstr :8000\ntaskkill /F /PID <PID>", language="bash")
                        st.info("3. Or restart Streamlit to clear the session")

    with col_server2:
        if server_running:
            if st.button("Stop Server", use_container_width=True, type="secondary", key="stop_server_btn"):
                success, message = stop_watcher_server()
                if success:
                    st.success(message)
                else:
                    st.warning(message)
                st.rerun()
        else:
            st.button("Server Stopped", use_container_width=True, disabled=True)

    with col_server3:
        if st.button("Refresh Status", use_container_width=True, key="refresh_status_control"):
            st.rerun()

with tabs[2]:
    # Watching Control Section (only if server is running)
    if server_running:
        st.markdown("##### 👁️ Watching Control")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Start Watching", use_container_width=True, type="primary", key="start_watching_btn"):
                try:
                    # Ensure path doesn't have quotes
                    clean_path = watcher_directory.strip().strip('"').strip("'")
                    st.info(f"Starting to watch: `{clean_path}`")
                    
                    response = requests.post(
                        f"{watcher_server_url}/watch/start",
                        params={"watch_directory": clean_path},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Started watching directory!")
                        if result.get("watch_dir"):
                            st.info(f"Watching: `{result['watch_dir']}`")
                        st.rerun()
                    else:
                        error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                        error_msg = error_data.get('error', error_data.get('detail', f'HTTP {response.status_code}'))
                        st.error(f"❌ Error starting watcher: {error_msg}")
                        if error_data.get('watch_dir'):
                            st.info(f"Current watch directory: `{error_data['watch_dir']}`")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The server may be busy. Try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Could not connect to watcher server: {e}")

        with col2:
            if st.button("Stop Watching", use_container_width=True, key="stop_watching_btn"):
                try:
                    response = requests.post(f"{watcher_server_url}/watch/stop", timeout=5)
                    if response.status_code == 200:
                        st.success("Stopped watching")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to watcher server: {e}")

        with col3:
            if st.button("Refresh Watching Status", use_container_width=True, key="refresh_watching_status"):
                st.rerun()
        
        # Manual scan button
        st.divider()
        st.markdown("**Manual Actions**")
        col_scan1, col_scan2 = st.columns(2)
        
        with col_scan1:
            if st.button("Scan Directory for Files", use_container_width=True, key="scan_directory_btn"):
                try:
                    response = requests.post(f"{watcher_server_url}/scan-directory", timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Scanned directory successfully!")
                        st.info(f"Found {result.get('files_found', 0)} CSV/Excel files")
                        st.info(f"Processed {result.get('files_processed', 0)} files")
                        if result.get('processed_files'):
                            st.json(result['processed_files'])
                        st.rerun()
                    else:
                        error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                        st.error(f"Error: {error_data.get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to watcher server: {e}")
        
        with col_scan2:
            test_file_path = st.text_input(
                "Test File Path:",
                value="",
                help="Enter full path to a CSV/Excel file to test detection",
                key="test_file_path_input"
            )
        if st.button("Test File Detection", use_container_width=True, key="test_file_detection_btn"):
                if test_file_path:
                    try:
                        response = requests.post(
                            f"{watcher_server_url}/test-file-detection",
                            params={"file_path": test_file_path},
                            timeout=30
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"{result.get('message', 'Test completed')}")
                            st.rerun()
                        else:
                            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                            st.error(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Could not connect to watcher server: {e}")
                else:
                    st.warning("Please enter a file path")
    else:
        st.info("Start the watcher server in the Server tab to enable watching controls.")

with tabs[3]:
    # Log Viewer Section
    st.markdown("##### 📜 Watcher Server Logs")

    # Get log file path
    log_file_path = None
    if "watcher_log_file" in st.session_state:
        log_file_path = Path(st.session_state.watcher_log_file)
    elif server_running:
        # Try to find the most recent log file
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        logs_dir = project_root / "logs"
        if logs_dir.exists():
            log_files = sorted(logs_dir.glob("watcher_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            if log_files:
                log_file_path = log_files[0]

    if log_file_path and log_file_path.exists():
        # Auto-refresh checkbox
        auto_refresh = st.checkbox("Auto-refresh logs", value=True, key="auto_refresh_logs")
        
        # Read and display logs
        log_content = read_log_file(log_file_path)
        
        if log_content:
            # Show last N lines (most recent)
            lines = log_content.split('\n')
            max_lines = 200  # Show last 200 lines
            if len(lines) > max_lines:
                display_lines = lines[-max_lines:]
                st.caption(f"Showing last {max_lines} lines of {len(lines)} total lines")
            else:
                display_lines = lines
            
            # Display in code block with scrolling
            log_display = '\n'.join(display_lines)
            st.code(log_display, language=None)
            
            # Show log file info
            st.caption(f"Log file: `{log_file_path.name}` | Size: {log_file_path.stat().st_size / 1024:.2f} KB")
            
            # Auto-refresh if enabled
            if auto_refresh and server_running:
                time.sleep(1)
                st.rerun()
        else:
            st.info("Log file is empty. Logs will appear here once the server starts processing events.")
    else:
        if server_running:
            st.info("Log file not found. Logs will appear here once events are processed.")
        else:
            st.info("Start the watcher server to see logs here.")

with tabs[4]:
    # Show watch directory contents
    watch_path = Path(watcher_directory)
    if watch_path.exists():
        st.markdown("##### 📁 Watch Directory Contents")
        files = list(watch_path.glob("*"))
        if files:
            file_list = []
            for f in files:
                if f.is_file():
                    file_list.append({
                        "File": f.name,
                        "Size": f"{f.stat().st_size / 1024:.2f} KB",
                        "Modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
            if file_list:
                import pandas as pd
                df = pd.DataFrame(file_list)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Watch directory is empty. Upload CSV/Excel files here to trigger curve fitting.")
    else:
        st.warning(f"Watch directory `{watcher_directory}` does not exist. It will be created when you start watching.")

with tabs[5]:
    # Instructions
    with st.expander("📖 How to Use the Watcher", expanded=True):
        st.markdown("""
        ### Quick Start
        
        1. **Set API Key** (if not already set):
           - Go to Settings → General tab
           - Enter your Google Gemini API key
           - The watcher server will automatically use this key
        
        2. **Start Watcher Server**:
           - Click "Start Server" above
           - The server will start in the background using your API key from Settings
           - Wait for the status to show "Watcher server is running"
        
        3. **Configure Watch Directory**:
           - Use the Configuration tab to set the watcher directory
        
        4. **Start Watching**:
           - Click "Start Watching" in the Watching tab
        
        ### How It Works
        
        - **Automatic Detection**: When you upload a CSV or Excel file to the watch directory, the watcher automatically detects it
        - **Auto-Trigger**: Curve fitting is automatically triggered for detected data files
        - **Jupyter Upload**: If enabled in Settings, files are automatically uploaded to your Jupyter server
        - **API Key Sharing**: The API key from your Streamlit session is automatically passed to the watcher server
        
        ### Supported File Types
        
        - `.csv` - CSV spectral data files
        - `.xlsx` / `.xls` - Excel spectral data files
        
        ### Tips
        
        - The watcher server runs in the background - you can stop it anytime with "Stop Server"
        - Files are processed as soon as they appear in the directory
        - The API key from Settings is automatically used - no need to set environment variables
        - If you change your API key in Settings, restart the watcher server to use the new key
        """)
