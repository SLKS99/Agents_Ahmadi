import streamlit as st
from tools.memory import MemoryManager
from agents.hypothesis_agent import HypothesisAgent

memory = MemoryManager()
memory.init_session()

def clear_conversation():
    st.session_state.stage = "initial"
    st.session_state.conversation_history = []
    st.session_state.allow_followup = False
    st.toast("Conversation restarted")

def stop_and_create_hypothesis():
    st.session_state.stop_hypothesis = True
    st.session_state.stage = "hypothesis"
    st.toast("Generating hypothesis...")
    st.rerun()

def go_back_stage():
    # Go back one stage of the process
    if st.session_state.interactions:
        st.session_state.interactions.pop()
        st.toast("Returned to previous stage")
    else:
        st.warning("No previous stage to go back to.")

st.set_page_config(layout="centered")
st.title("🧠 AI Hypothesis Agent")

# Add custom CSS for better visual distinction between chat sections
st.markdown("""
    <style>
    /* User messages - light blue background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUser"]) {
        background-color: #E3F2FD !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #2196F3 !important;
    }

    /* Assistant messages - light green background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAssistant"]) {
        background-color: #F1F8E9 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #4CAF50 !important;
    }

    /* Add extra spacing between chat messages */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1.5rem !important;
    }

    /* Style for section headers (bold text) */
    div[data-testid="stChatMessage"] strong {
        color: #1565C0 !important;
        font-size: 1.05em;
    }

    /* Visual separator for horizontal rules */
    div[data-testid="stChatMessage"] hr {
        border: none;
        border-top: 2px solid #BDBDBD;
        margin: 1rem 0;
    }

    /* Style for option numbers to make them stand out */
    div[data-testid="stChatMessage"] p strong:first-child {
        color: #7B1FA2 !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Layout styling
st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
        align-items: flex-end;
    }
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        border-top: 1px solid #ddd;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Display existing chat
chat_container = st.container()
with chat_container:
    # Ensure interactions list exists
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    for i in st.session_state.interactions:
        with st.chat_message(i["role"]):
            # Add section headers for specific components to make them clear
            if i.get("component") == "socratic_answers":
                st.markdown("**Socratic Reasoning (LLM Answers to Its Own Questions):**")
            st.markdown(i["message"])

st.markdown("<br>", unsafe_allow_html=True)

# Bottom controls
bottom = st.container()
with bottom:
    with st.popover("Options"):
        st.markdown("#### Conversation Controls")
        st.button("Restart", use_container_width=True, on_click=clear_conversation)
        st.button("Stop & Create Hypothesis", use_container_width=True, on_click=stop_and_create_hypothesis)
        st.button("Go Back", use_container_width=True, on_click=go_back_stage)

# Initialize and run the Hypothesis Agent
# Get initial question from session state or use a default
initial_question = st.session_state.get("initial_question", "")

# Create and run the agent
agent = HypothesisAgent(
    name="Hypothesis Agent",
    desc="Helps generate scientific hypotheses through Socratic questioning",
    question=initial_question
)

# Run the agent - it will handle all the UI and state management
agent.run_agent(memory)

# Workflow transition: offer manual Continue (no auto-switch)
if (
    st.session_state.get("workflow_active")
    and st.session_state.get("stage") == "analysis"
    and st.session_state.get("hypothesis_ready")
):
    st.session_state.workflow_step = "experiment"
    st.session_state.workflow_experiment_started = False
    st.divider()
    if st.button("Continue to Experiment Agent →", type="primary", use_container_width=True, key="hyp_continue_experiment"):
        st.switch_page("pages/experiment.py")
