import streamlit as st
import json
from datetime import datetime
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

def safe_json_dumps(obj, indent=2):
    """Safely serialize JSON, handling circular references and complex objects"""
    def make_serializable(o, visited=None):
        """Recursively convert object to JSON-serializable format"""
        if visited is None:
            visited = set()
        
        obj_id = id(o)
        if obj_id in visited:
            return "<circular reference>"
        
        visited.add(obj_id)
        
        try:
            if isinstance(o, dict):
                return {k: make_serializable(v, visited) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [make_serializable(item, visited) for item in o]
            elif isinstance(o, (str, int, float, bool, type(None))):
                return o
            else:
                # For other types, try to convert to string
                return str(o)
        except Exception:
            return f"<non-serializable: {type(o).__name__}>"
        finally:
            visited.discard(obj_id)
    
    try:
        serializable_obj = make_serializable(obj)
        return json.dumps(serializable_obj, indent=indent, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to serialize: {str(e)}"}, indent=indent)

def export_message_history(events=None):
    """Export conversation events as JSON or text"""
    if events is None:
        events = st.session_state.get("conversation_events", [])
    
    if not events:
        return "[]", "No conversation history available."

    json_data = safe_json_dumps(events, indent=2)
    text_data = "\n\n".join([
        f"[{event.get('timestamp', 'N/A')}] {event.get('type', 'unknown').upper()}\n"
        f"Mode: {event.get('mode', 'unknown')}\n"
        f"Payload: {safe_json_dumps(event.get('payload', {}), indent=2)}"
        for event in events
    ])

    return json_data, text_data

def get_events_by_mode(mode):
    """Get all events for a specific agent mode"""
    events = st.session_state.get("conversation_events", [])
    return [e for e in events if e.get("mode") == mode]

def format_interaction_text(event):
    """Format a single interaction event as readable text"""
    payload = event.get("payload", {})
    role = payload.get("role", "unknown")
    message = payload.get("message", "")
    component = payload.get("component", "")
    timestamp = event.get("timestamp", "N/A")
    
    # Truncate very long messages for readability
    if len(message) > 500:
        message = message[:500] + "..."
    
    return f"[{timestamp}] {role.upper()} - {component.upper()}: {message}"

def format_experiment_event(event):
    """Format experiment-specific events with better context"""
    payload = event.get("payload", {})
    role = payload.get("role", "unknown")
    message = payload.get("message", "")
    component = payload.get("component", "")
    timestamp = event.get("timestamp", "N/A")
    
    # Special formatting for experiment components
    if component == "experimental_plan":
        return f"[{timestamp}] EXPERIMENTAL PLAN:\n{message[:1000]}..." if len(message) > 1000 else f"[{timestamp}] EXPERIMENTAL PLAN:\n{message}"
    elif component in ["clarified_question", "socratic_pass", "socratic_answers", "hypothesis"]:
        return f"[{timestamp}] {role.upper()} - {component.upper()}:\n{message[:500]}..." if len(message) > 500 else f"[{timestamp}] {role.upper()} - {component.upper()}:\n{message}"
    else:
        return format_interaction_text(event)

st.set_page_config(layout="wide")
st.title("Interaction History")

# Get all events
all_events = st.session_state.get("conversation_events", [])

# Group events by mode (agent)
agent_modes = {}
for event in all_events:
    mode = event.get("mode", "general")
    if mode not in agent_modes:
        agent_modes[mode] = []
    agent_modes[mode].append(event)

# Create tabs for each agent plus "All Interactions"
# Ensure key agent tabs are always present
key_agents = ["hypothesis", "experiment", "curve fitting", "watcher"]
tab_names = ["All Interactions"]

# Add key agent tabs - always include hypothesis and experiment tabs, others only if they have events
for agent in key_agents:
    agent_key = agent.lower()
    # Always show hypothesis and experiment tabs, others only if they have events
    if agent_key in agent_modes or agent in ["hypothesis", "experiment"]:
        tab_names.append(agent.capitalize())

# Add other agent modes that aren't in key_agents
other_modes = sorted([m.capitalize() for m in agent_modes.keys() 
                     if m.lower() not in [k.lower() for k in key_agents] and m != "general"])
tab_names.extend(other_modes)

if "general" in agent_modes:
    tab_names.append("General")

tabs = st.tabs(tab_names)

# All Interactions tab
with tabs[0]:
    st.subheader("All Interactions")
    
    if not all_events:
        st.info("No conversation history available yet.")
    else:
        display_type = st.segmented_control("Display Type", ["Text", "JSON"], key="all_display_type")
        
        if display_type == "JSON":
            json_data, _ = export_message_history(all_events)
            st.json(json_data)
        else:
            text_lines = []
            for event in all_events:
                if event.get("type") == "interaction":
                    text_lines.append(format_interaction_text(event))
                elif event.get("type") == "history":
                    payload = event.get("payload", {})
                    text_lines.append(f"[{event.get('timestamp', 'N/A')}] HISTORY")
                    if payload.get("question"):
                        text_lines.append(f"  Question: {payload['question']}")
                    if payload.get("hypothesis"):
                        text_lines.append(f"  Hypothesis: {payload['hypothesis'][:200]}...")
                else:
                    text_lines.append(f"[{event.get('timestamp', 'N/A')}] {event.get('type', 'unknown').upper()}: {safe_json_dumps(event.get('payload', {}), indent=2)}")
            
            st.text("\n".join(text_lines))

# Agent-specific tabs
for i, tab_name in enumerate(tab_names[1:], 1):
    mode_key = tab_name.lower()
    with tabs[i]:
        st.subheader(f"{tab_name} Agent Interactions")
        
        mode_events = agent_modes.get(mode_key, [])
        
        if not mode_events:
            st.info(f"No interactions recorded for {tab_name} agent yet.")
        else:
            display_type = st.segmented_control("Display Type", ["Text", "JSON"], key=f"{mode_key}_display_type")
            
            if display_type == "JSON":
                json_data, _ = export_message_history(mode_events)
                st.json(json_data)
            else:
                text_lines = []
                for event in mode_events:
                    if event.get("type") == "interaction":
                        # Use experiment-specific formatting for experiment mode
                        if mode_key == "experiment":
                            text_lines.append(format_experiment_event(event))
                        else:
                            text_lines.append(format_interaction_text(event))
                    elif event.get("type") == "history":
                        payload = event.get("payload", {})
                        text_lines.append(f"[{event.get('timestamp', 'N/A')}] HISTORY")
                        if payload.get("question"):
                            text_lines.append(f"  Question: {payload['question']}")
                        if payload.get("hypothesis"):
                            text_lines.append(f"  Hypothesis: {payload['hypothesis'][:200]}...")
                        if payload.get("experimental_plan"):
                            text_lines.append(f"  Experimental Plan: {payload['experimental_plan'][:500]}...")
                    else:
                        text_lines.append(f"[{event.get('timestamp', 'N/A')}] {event.get('type', 'unknown').upper()}: {safe_json_dumps(event.get('payload', {}), indent=2)}")
                
                st.text("\n".join(text_lines))

# Export section
st.divider()
with st.expander("Export Data", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        json_data, _ = export_message_history(all_events)
        st.download_button(
            label="Export All as JSON",
            data=json_data,
            file_name=f"conversation_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        _, text_data = export_message_history(all_events)
        st.download_button(
            label="Export All as Text",
            data=text_data,
            file_name=f"conversation_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )