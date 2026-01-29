from datetime import datetime
import time
import logging
import os
import uuid

# Lazy import streamlit to avoid issues in headless mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except (ImportError, RuntimeError):
    STREAMLIT_AVAILABLE = False
    st = None
import json

class MemoryManager:
    def __init__(self):
        pass

    def init_session(self):
        """ Initialize the session and variables """
        defaults = {
            "start_time": time.time(),
            "conversation_events": [],
            "agent_usage_counts": {
                "hypothesis": 0,
                "experiment": 0,
                "curve_fit": 0,
                "analysis": 0,
                "router": 0,
                "watcher": 0
            },
            "api_key": "",
            "api_key_source": "",
            "stage": "initial",
            "cf_data_path": "",
            "cf_comp_path": "",
            "experimental_mode": False,
            "experimental_constraints": {
                "techniques": [],
                "equipment": [],
                "parameters": [],
                "focus_areas": [],
                "liquid_handling": {
                    "max_volume_per_mixture": 50,
                    "instruments": [],
                    "plate_format": "96-well",
                    "materials": [],
                    "csv_path": "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"
                }
            },
            "jupyter_config": {
                "server_url": "http://10.140.141.160:48888/",
                "token": "",
                "upload_enabled": False,
                "notebook_path": "Automated Agent"
            },
            "prompt_session": [],
            "current_prompt_session_id": str(uuid.uuid4()),
            "manual_workflow": ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"],
            "workflow_index": 0,
            "routing_mode": "Autonomous (LLM)",  # or "Manual"
            "workflow_auto_flags": {},  # Dict mapping step names to auto-execution flags
            "current_workflow_name": None,  # Name of currently active workflow
            "uploaded_files": [],
            "hypothesis_ready": False,
            "last_hypothesis": None,
            "experimental_outputs": None,
            "stop_hypothesis": False,
            "hypothesis_round_count": 0,
            "max_hypothesis_rounds": 5,
            "workflow_active": False,
            "workflow_step": "idle",
            "workflow_completed": False,
            "workflow_experiment_started": False,
            "workflow_experiment_completed": False,
            "workflow_experiment_outputs": None,
            "research_goal": "",
            "experiment_memory_file": "experiment_memory.json",
            "experiment_data_dir": "data",
            "watcher_directory": r"C:\Users\shery\OneDrive - University of Tennessee\Data Experiment files",
            "watcher_results_dir": "results",
            "watcher_enabled": False,
            "watcher_port": 8000,
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Only set API key from environment/secrets if user hasn't manually set one
        if not STREAMLIT_AVAILABLE:
            return
        
        if "api_key_source" not in st.session_state or st.session_state.api_key_source != "user":
            env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            # If we have an API key from environment, use it automatically
            if env_key:
                # Set in session state
                st.session_state.api_key = env_key
                st.session_state.api_key_source = "environment"
                # Ensure environment variables are set
                os.environ["GOOGLE_API_KEY"] = env_key
                os.environ["GEMINI_API_KEY"] = env_key

            elif "api_key" not in st.session_state or len(st.session_state.api_key) == 0:
                # Check if API Key is in secrets
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    st.session_state.api_key = api_key
                    st.session_state.api_key_source = "secrets"
                os.environ["GEMINI_API_KEY"] = api_key
                os.environ["GOOGLE_API_KEY"] = api_key
        else:
            # Session state exists but no env key - sync session state to environment
            if st.session_state.api_key:
                os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
                os.environ["GEMINI_API_KEY"] = st.session_state.api_key

    def log_event(self, event_type: str, payload: dict, mode: str):
        """ Unified event log to session state (or logger if Streamlit not available) """
        try:
            # Try to use Streamlit session state if available
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'conversation_events'):
                st.session_state.conversation_events.append({
                    "type": event_type,
                    "mode": mode,
                    "prompt_session_id": st.session_state.current_prompt_session_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "payload": payload,
                })
            else:
                # Streamlit not available (headless mode) - use Python logging
                logger = logging.getLogger(__name__)
                logger.info(f"Event [{mode}]: {event_type} - {payload}")
        except (RuntimeError, AttributeError, ImportError):
            # Streamlit not initialized (e.g., in watcher server) - use Python logging
            logger = logging.getLogger(__name__)
            logger.info(f"Event [{mode}]: {event_type} - {payload}")

    def save_to_history(self, question, mode, hypothesis=None, thoughts=None):
        """ Log important parts of hypothesis agent conversation"""
        self.log_event(
            "history",
            {
                "question": question,
                "hypothesis": hypothesis,
                "thoughts": thoughts,
            },
            mode=mode
        )

    def insert_interaction(self, role, message, component, mode):
        """ Adding interactions to session state """
        # Log to conversation_events for history tracking
        self.log_event(
            "interaction",
            {
                "role": role,
                "message": message,
                "component": component,
            },
            mode=mode
        )
        
        # Also add to st.session_state.interactions for UI display (if Streamlit available)
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                if "interactions" not in st.session_state:
                    st.session_state.interactions = []
                st.session_state.interactions.append({
                    "role": role,
                    "message": message,
                    "component": component,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
        except (RuntimeError, AttributeError, ImportError):
            # Streamlit not available - skip UI interaction logging
            pass

    def get_latest_history(self, mode=None):
        """ Get latest conversation history """
        try:
            import streamlit as st
            if not hasattr(st, 'session_state'):
                return None
            events = st.session_state.get("conversation_events", [])
            for event in reversed(events):
                if event["type"] == "history":
                    if mode is None or event["mode"] == mode:
                        return event
        except (RuntimeError, AttributeError, ImportError):
            pass
        return None

    def get_prompt_session_events(self, prompt_session_id: str):
        """ Get all events from prompt session """
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'conversation_events'):
                return [
                    event
                    for event in st.session_state.conversation_events
                    if event.get("prompt_session_id") == prompt_session_id
                ]
        except (RuntimeError, AttributeError, ImportError):
            pass
        return []

    def get_prompt_session_history(self, prompt_session_id: str):
        """ Get all history from prompt session """
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'conversation_events'):
                return [
                    event
                    for event in st.session_state.conversation_events
                    if event["type"] == "history" and
                       event.get("prompt_session_id") == prompt_session_id
                ]
        except (RuntimeError, AttributeError, ImportError):
            pass
        return []

    def get_prompt_session_interactions(self, prompt_session_id: str):
        """ Get latest interactions from prompt session """
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'conversation_events'):
                return [
                    event
                    for event in st.session_state.conversation_events
                    if event["type"] == "interaction" and
                       event.get("prompt_session_id") == prompt_session_id
                ]
        except (RuntimeError, AttributeError, ImportError):
            pass
        return []

    def view_component(self, component, prompt_session_id: str = None):
        """ Getting components from session state """
        try:
            import streamlit as st
            if not hasattr(st, 'session_state'):
                return None
            if prompt_session_id is None:
                prompt_session_id = st.session_state.current_prompt_session_id

            for event in reversed(st.session_state.conversation_events):
                if event["type"] == "interaction" and event.get("prompt_session_id") == prompt_session_id:
                    payload = event["payload"]
                    if payload["component"] == component:
                        return payload["message"]
        except (RuntimeError, AttributeError, ImportError):
            pass
        return None

    def add_uploaded_file(self, filename, path):
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                if "uploaded_files" not in st.session_state:
                    st.session_state.uploaded_files = []
                st.session_state.uploaded_files.append({
                    "name": filename,
                    "path": path,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        except (RuntimeError, AttributeError, ImportError):
            # Streamlit not available - skip file tracking
            pass

    # -------- Session state helpers --------

    def snapshot_session_state(self, note: str = "session_state_snapshot"):
        """
        Capture a JSON‑serialisable snapshot of the current session_state.

        This is useful for debugging complex workflows and for analytics
        (e.g. seeing which parameters were active when an agent ran).
        """
        serialisable = {}
        skipped_keys = []
        for k, v in st.session_state.items():
            try:
                # Only keep JSON‑serialisable entries
                json.dumps(v)
                serialisable[k] = v
            except (TypeError, ValueError) as e:
                # Skip non-serializable objects (including circular references)
                skipped_keys.append(f"{k} ({type(v).__name__}: {str(e)[:50]}...)")
                continue

        if skipped_keys:
            logging.debug(f"Skipped non-serializable session state keys: {skipped_keys}")

        self.log_event(
            "session_state",
            {
                "note": note,
                "state": serialisable,
            },
            mode="system",
        )

    def get_var(self, name: str, default=None):
        """Safe accessor for session variables."""
        return st.session_state.get(name, default)

    def set_var(self, name: str, value):
        """Safe setter for session variables."""
        st.session_state[name] = value

