import streamlit as st
from agents.analysis_agent import AnalysisAgent
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="wide")
st.title("🔎 Analysis Agent")
st.markdown("Analyze curve fitting results in relation to your hypothesis and experimental data.")

# Initialize analysis agent
agent = AnalysisAgent()

# Run the agent
agent.run_agent(memory)
