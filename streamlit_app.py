# Performance optimized imports
import streamlit as st
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:")
workflow = st.Page("pages/workflow.py", title="Workflow", icon=":material/automation:")
workflow_builder = st.Page("pages/workflow_builder.py", title="Workflow Builder", icon=":material/build:")
hypothesis = st.Page("pages/hypothesis.py", title="Hypothesis", icon=":material/cognition_2:")
experiment = st.Page("pages/experiment.py", title="Experiment", icon=":material/experiment:")
curve_fit = st.Page("pages/curve_fitting.py", title="Curve Fitting", icon=":material/bar_chart:")
ml_models = st.Page("pages/ml_models.py", title="ML Models", icon=":material/psychology:")
analysis = st.Page("pages/analysis.py", title="Analysis", icon=":material/analytics:")
settings = st.Page("pages/settings.py", title="Settings", icon=":material/settings:")
history = st.Page("pages/history.py", title="History", icon=":material/history:")
watcher_control = st.Page("pages/watcher_control.py", title="Watcher Control", icon=":material/visibility:")

pg = st.navigation({
    "General": [home, dashboard, workflow, workflow_builder],
    "Agents": [hypothesis, experiment, curve_fit, ml_models, analysis],
    "Tools": [settings, history, watcher_control],
})

pg.run()