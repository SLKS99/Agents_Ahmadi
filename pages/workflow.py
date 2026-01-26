import streamlit as st

from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("Workflow Runner")
st.markdown("Start an end-to-end workflow: Hypothesis → Experiment → Curve Fitting.")

def _reset_workflow_state() -> None:
    st.session_state.workflow_active = True
    st.session_state.workflow_step = "hypothesis"
    st.session_state.workflow_completed = False
    st.session_state.workflow_experiment_started = False
    st.session_state.workflow_experiment_completed = False
    st.session_state.workflow_experiment_outputs = None
    st.session_state.experimental_outputs = None
    st.session_state.hypothesis_ready = False
    st.session_state.stop_hypothesis = False
    st.session_state.stage = "initial"

col_start, col_stop = st.columns(2)

with col_start:
    if st.button("Start Workflow", type="primary", use_container_width=True):
        _reset_workflow_state()
        st.toast("Workflow started.")
        st.switch_page("pages/hypothesis.py")
with col_stop:
    if st.button("Stop Workflow", use_container_width=True):
        st.session_state.workflow_active = False
        st.session_state.workflow_step = "idle"
        st.toast("Workflow stopped.")
        st.rerun()


if not st.session_state.workflow_active:
    st.info("Start the workflow to begin the Hypothesis Agent.")
    st.stop()

st.markdown(
    f"**Current Step:** `{st.session_state.workflow_step}`"
)

if st.session_state.workflow_step == "hypothesis":
    st.subheader("Hypothesis Agent")
    st.switch_page("pages/hypothesis.py")

if st.session_state.workflow_step == "experiment":
    st.subheader("Experiment Agent")
    st.switch_page("pages/experiment.py")

if st.session_state.workflow_step == "curve_fitting":
    st.subheader("Curve Fitting")
    st.switch_page("pages/curve_fitting.py")
