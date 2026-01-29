import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("Settings")
st.markdown("Adjust the settings for the various agents and workflows below.")

general, experiment, watcher, cache = st.tabs(
    ["General", "Experiment", "Watcher", "Cache"]
)

with general:

    # Changing API Key
    st.markdown("##### API Key Configuration")

    # Track editing mode
    if "editing" not in st.session_state:
        st.session_state.editing = False

    if st.session_state.api_key and not st.session_state.editing:
        st.success("Your API key was loaded from session state successfully.")

        if st.button("Edit API Key"):
            st.session_state.editing = True
            # Clear the input field when entering edit mode
            if "api_key_input" in st.session_state:
                del st.session_state.api_key_input
            st.rerun()

    else:
        # Show current status
        if st.session_state.get('api_key_source') == 'environment':
            st.info("📝 Currently using API key from environment variables. Click 'Edit API Key' to set a custom one.")
        elif st.session_state.get('api_key_source') == 'secrets':
            st.info("📝 Currently using API key from Streamlit secrets. Click 'Edit API Key' to set a custom one.")

        api_key_input = st.text_input(
            "Google Gemini API Key:",
            value="" if st.session_state.editing else st.session_state.get('api_key', ''),
            type="password",
            help="Enter your Google Gemini API key. It will be saved securely and persist across page navigations.",
            key="api_key_input",
            placeholder="Enter your API key here..." if st.session_state.editing else None,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save API Key", use_container_width=True):
                if api_key_input and api_key_input.strip():
                    api_key = api_key_input.strip()
                    st.session_state.api_key = api_key
                    st.session_state.api_key_source = "user"
                    st.session_state.editing = False
                    # Also save to environment variable for persistence across page navigations
                    os.environ['GEMINI_API_KEY'] = api_key
                    # Try to save to .env file for persistence across app restarts
                    try:
                        env_path = Path(__file__).parent.parent / '.env'
                        with open(env_path, 'w') as f:
                            f.write(f'GEMINI_API_KEY={api_key}\n')
                    except Exception:
                        pass  # Ignore if we can't write to .env file
                    st.success("Your API key has been saved successfully!")
                    st.rerun()
                else:
                    st.error("Please enter your API key and try again.")

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.editing = False
                st.rerun()

    st.markdown("---")
    st.markdown("##### Workflow & Routing")

    # Routing mode control
    routing_mode = st.segmented_control(
        "Routing Mode",
        options=["Autonomous (LLM)", "Manual"],
        default=st.session_state.get("routing_mode", "Autonomous (LLM)"),
    )
    st.session_state.routing_mode = routing_mode

    # Manual workflow configuration
    if routing_mode == "Manual":
        st.info(
            "In manual mode, agents run in the order you specify below whenever the router is invoked."
        )

        available_agents = ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"]
        current_workflow = st.session_state.get(
            "manual_workflow", ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"]
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            step1 = st.selectbox(
                "Step 1",
                options=available_agents,
                index=available_agents.index(current_workflow[0])
                if current_workflow and current_workflow[0] in available_agents
                else 0,
            )
        with col_b:
            step2 = st.selectbox(
                "Step 2",
                options=available_agents,
                index=available_agents.index(current_workflow[1])
                if len(current_workflow) > 1 and current_workflow[1] in available_agents
                else 1,
            )
        with col_c:
            step3 = st.selectbox(
                "Step 3",
                options=available_agents,
                index=available_agents.index(current_workflow[2])
                if len(current_workflow) > 2 and current_workflow[2] in available_agents
                else 2,
            )

        new_workflow = [step1, step2, step3]
        st.session_state.manual_workflow = new_workflow

        if st.button("Reset Workflow Progress"):
            st.session_state.workflow_index = 0
            st.success("Workflow progress reset. The next routed call will start at Step 1.")

with experiment:
    # Experiment Configuration
    st.markdown("##### Experiment Configuration")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("**Jupyter Server Configuration**")
        st.info("Configure once - used by all agents (Experiments, Curve Fitting, etc.)")
        
        jupyter_url = st.text_input(
            "Jupyter Server URL:",
            value=st.session_state.jupyter_config["server_url"],
            help="Base URL only (e.g., http://10.140.141.160:48888/) - do NOT include /tree/ path",
            key="jupyter_url_input"
        )
        st.session_state.jupyter_config["server_url"] = jupyter_url
        
        jupyter_token = st.text_input(
            "Jupyter Token:",
            value=st.session_state.jupyter_config.get("token", ""),
            type="password",
            help="Authentication token for Jupyter server",
            key="jupyter_token_input"
        )
        st.session_state.jupyter_config["token"] = jupyter_token
        
        jupyter_notebook_path = st.text_input(
            "Base Notebook Path/Directory:",
            value=st.session_state.jupyter_config.get("notebook_path", "Automated Agent"),
            help="Base directory path in Jupyter (e.g., 'Automated Agent'). Curve fitting will create subfolders with filename_date",
            key="jupyter_notebook_path_input"
        )
        st.session_state.jupyter_config["notebook_path"] = jupyter_notebook_path
        
        jupyter_upload_enabled = st.checkbox(
            "Enable Auto-Upload to Jupyter",
            value=st.session_state.jupyter_config.get("upload_enabled", False),
            help="Automatically upload generated files to Jupyter server (applies to all agents)",
            key="jupyter_upload_enabled_input"
        )
        st.session_state.jupyter_config["upload_enabled"] = jupyter_upload_enabled
    
    with col_exp2:
        st.markdown("**Experiment Memory**")
        experiment_memory_file = st.text_input(
            "Experiment Memory File:",
            value=st.session_state.get("experiment_memory_file", "experiment_memory.json"),
            help="File to store completed experiment records",
            key="exp_memory_file_input"
        )
        st.session_state.experiment_memory_file = experiment_memory_file
        
        experiment_data_dir = st.text_input(
            "Experiment Data Directory:",
            value=st.session_state.get("experiment_data_dir", "data"),
            help="Directory where experiment data and memory files are stored",
            key="exp_data_dir_input"
        )
        st.session_state.experiment_data_dir = experiment_data_dir
    
    st.divider()
    
    # Watcher Configuration
    st.markdown("##### Watcher Configuration")
    st.markdown("Configure the file system watcher that automatically triggers curve fitting when files are uploaded.")
    
    col_wat1, col_wat2 = st.columns(2)
    
    with col_wat1:
        watcher_directory = st.text_input(
            "Watcher Directory:",
            value=st.session_state.get("watcher_directory", r"C:\Users\shery\OneDrive - University of Tennessee\Data Experiment files"),
            help="Full path to directory that the watcher should monitor for file changes (e.g., C:\\Users\\shery\\OneDrive - University of Tennessee\\Data Experiment files). Do not include quotes.",
            key="watcher_dir_input"
        )
        # Strip quotes if user accidentally added them
        watcher_directory = watcher_directory.strip().strip('"').strip("'")
        st.session_state.watcher_directory = watcher_directory
        
        # Show if directory exists
        watch_path = Path(watcher_directory).expanduser()
        if watch_path.exists() and watch_path.is_dir():
            st.success(f"✅ Directory exists and is accessible")
            # Show file count
            try:
                file_count = len(list(watch_path.glob("*")))
                st.caption(f"Contains {file_count} items")
            except:
                pass
        else:
            st.warning(f"⚠️ Directory does not exist or is not accessible")
            st.info("💡 **Tips:**")
            st.info("1. Make sure the path is correct (no quotes needed)")
            st.info("2. Check that you have read/write permissions")
            st.info("3. For OneDrive folders, ensure they are synced locally")
        
        watcher_results_dir = st.text_input(
            "Results Directory:",
            value=st.session_state.get("watcher_results_dir", "results"),
            help="Directory where curve fitting results are saved",
            key="watcher_results_dir_input"
        )
        st.session_state.watcher_results_dir = watcher_results_dir
    
    with col_wat2:
        watcher_enabled = st.checkbox(
            "Enable Watcher",
            value=st.session_state.get("watcher_enabled", False),
            help="Enable automatic file system watching for workflow automation",
            key="watcher_enabled_input"
        )
        st.session_state.watcher_enabled = watcher_enabled
        
        if watcher_enabled:
            watcher_port = st.number_input(
                "Watcher Server Port:",
                min_value=1000,
                max_value=9999,
                value=st.session_state.get("watcher_port", 8000),
                help="Port for the watcher HTTP server",
                key="watcher_port_input"
            )
            st.session_state.watcher_port = watcher_port
            
            st.info("💡 **Next Steps:**")
            st.info("1. Go to **Watcher Control** page to start the server")
            st.info("2. Click '🚀 Start Server' to start the watcher")
            st.info("3. Click '▶️ Start Watching' to begin monitoring")
    
    st.divider()
    
    # Legacy Jupyter Configuration (kept for backward compatibility)
    st.markdown("##### Jupyter Server Configuration (Legacy Form)")

    with st.form("jupyter_server_config"):
        col_jup1, col_jup2 = st.columns(2)

        with col_jup1:
            jupyter_url = st.text_input(
                "Jupyter Server URL:",
                value=st.session_state.jupyter_config["server_url"],
                help="URL of Jupyter server (e.g., http://10.140.141.160:48888/)",
            )

            jupyter_token = st.text_input(
                "Jupyter Token (optional):",
                value=st.session_state.jupyter_config["token"],
                type="password",
                help="Authentication token for Jupyter server",
            )

        with col_jup2:
            jupyter_notebook_path = st.text_input(
                "Notebook Path:",
                value=st.session_state.jupyter_config["notebook_path"],
                help="Directory path in Jupyter (e.g., 'Dual GP 5AVA BDA')",
            )

            jupyter_upload_enabled = st.checkbox(
                "Enable Auto-Upload to Jupyter",
                value=st.session_state.jupyter_config["upload_enabled"],
                help="Automatically upload generated files to Jupyter server",
            )

        submitted = st.form_submit_button(
            "Update Jupyter Server Configuration", use_container_width=True
        )

    if submitted:
        st.session_state.jupyter_config = {
            "server_url": jupyter_url,
            "token": jupyter_token,
            "upload_enabled": jupyter_upload_enabled,
            "notebook_path": jupyter_notebook_path,
        }

        st.success("Jupyter Server Configuration loaded successfully!")

with watcher:
    st.markdown("##### Watcher Configuration")
    
    st.info("""
    **📝 Note:** Watcher configuration has been moved to the **Experiment** tab for better organization.
    
    **🔧 To control the watcher server:**
    - Go to **Watcher Control** page in the navigation menu
    - Or use the configuration in the **Experiment** tab above
    """)
    
    st.markdown("---")
    
    st.markdown("**Current Watcher Settings:**")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"""
        **Directory:** `{st.session_state.get("watcher_directory", "Not set")}`  
        **Results Dir:** `{st.session_state.get("watcher_results_dir", "results")}`  
        **Port:** `{st.session_state.get("watcher_port", 8000)}`
        """)
    
    with col_info2:
        watcher_enabled_status = "✅ Enabled" if st.session_state.get("watcher_enabled", False) else "❌ Disabled"
        st.markdown(f"**Status:** {watcher_enabled_status}")
        
        # Quick link to watcher control
        if st.button("🔍 Go to Watcher Control", use_container_width=True, type="primary"):
            st.switch_page("pages/watcher_control.py")
    
    st.markdown("---")
    
    # Quick reference
    with st.expander("📖 Quick Reference", expanded=False):
        st.markdown("""
        **Watcher Configuration Locations:**
        
        1. **Settings → Experiment Tab**: Configure watcher directory, port, and enable/disable
        2. **Watcher Control Page**: Start/stop server, view logs, scan directory
        
        **What the Watcher Does:**
        - Monitors a directory for CSV/Excel files
        - Automatically triggers curve fitting when files are detected
        - Runs in the background as a separate server process
        
        **Quick Start:**
        1. Set your API key in Settings → General
        2. Configure watcher directory in Settings → Experiment
        3. Go to Watcher Control page
        4. Click "🚀 Start Server"
        5. Click "▶️ Start Watching"
        """)

with cache:
    st.markdown("##### Cache Management")
    st.markdown("Clear cached data from the Streamlit application to force fresh computations.")
    
    col_cache1, col_cache2 = st.columns(2)
    
    with col_cache1:
        st.markdown("**Streamlit Cache**")
        st.markdown("Clear Streamlit's built-in cache decorators (`@st.cache_data`, `@st.cache_resource`).")
        
        if st.button("Clear Streamlit Cache", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Streamlit cache cleared successfully!")
            st.rerun()
    
    with col_cache2:
        st.markdown("**Session State**")
        st.markdown("Reset session state variables (this will restart your session).")
        
        if st.button("Clear Session State", type="secondary", use_container_width=True):
            # Clear all session state except essential keys
            essential_keys = ["start_time", "api_key", "api_key_source"]
            keys_to_clear = [k for k in st.session_state.keys() if k not in essential_keys]
            for key in keys_to_clear:
                del st.session_state[key]
            st.success("✅ Session state cleared! Page will reload.")
            st.rerun()
    
    st.markdown("---")
    st.markdown("**Clear Everything**")
    st.markdown("⚠️ **Warning:** This will clear all caches and reset your session completely.")
    
    if st.button("Clear All Caches & Reset Session", type="primary", use_container_width=True):
        # Clear Streamlit caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear session state (keep only minimal essentials)
        essential_keys = ["start_time"]
        keys_to_clear = [k for k in st.session_state.keys() if k not in essential_keys]
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Reinitialize session
        memory.init_session()
        
        st.success("✅ All caches cleared and session reset!")
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Cache Statistics**")
    
    # Display cache info if available
    try:
        cache_info = st.cache_data.get_stats()
        if cache_info:
            st.json(cache_info)
        else:
            st.info("No cache statistics available.")
    except Exception:
        st.info("Cache statistics not available in this Streamlit version.")