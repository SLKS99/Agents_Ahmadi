import json
from typing import List, Dict, Any, Optional

import streamlit as st

from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="wide")
st.title("🧭 Workflow")
st.markdown("Run and configure end-to-end workflows across agents and tools.")

# Available agents/steps
AVAILABLE_STEPS = {
    "Hypothesis Agent": {
        "name": "Hypothesis Agent",
        "description": "Generate and refine research hypotheses",
        "can_auto": False,  # Requires user input
        "page": "pages/hypothesis.py",
    },
    "Experiment Agent": {
        "name": "Experiment Agent",
        "description": "Design experiments and generate protocols",
        "can_auto": False,  # Requires user review
        "page": "pages/experiment.py",
    },
    "Curve Fitting": {
        "name": "Curve Fitting",
        "description": "Fit curves to spectral data",
        "can_auto": True,  # Can run automatically if data is ready
        "page": "pages/curve_fitting.py",
    },
    "ML Models": {
        "name": "ML Models",
        "description": "Run ML models for optimization",
        "can_auto": True,  # Can run automatically after curve fitting
        "page": "pages/ml_models.py",
    },
    "Analysis Agent": {
        "name": "Analysis Agent",
        "description": "Analyze results and provide recommendations",
        "can_auto": True,  # Can run automatically after ML/curve fitting
        "page": "pages/analysis.py",
    },
}

# Initialize workflow storage
if "workflows" not in st.session_state:
    st.session_state.workflows = {}

if "current_workflow" not in st.session_state:
    st.session_state.current_workflow = {
        "name": "Default Workflow",
        "steps": [],
        "active": False,
    }

if "workflow_steps" not in st.session_state:
    st.session_state.workflow_steps = []


def save_workflow(
    workflow_name: str,
    steps: List[Dict[str, Any]],
    ml_model_choice: Optional[str] = None,
):
    """Save workflow to session state."""
    st.session_state.workflows[workflow_name] = {
        "name": workflow_name,
        "steps": steps,
        "ml_model_choice": ml_model_choice,
        "created_at": st.session_state.get("workflow_created_at", ""),
    }


def load_workflow(workflow_name: str) -> Optional[Dict[str, Any]]:
    """Load workflow from session state."""
    return st.session_state.workflows.get(workflow_name)


def apply_workflow(workflow_name: str):
    """Apply workflow settings to session state."""
    workflow = load_workflow(workflow_name)
    if workflow:
        # Set manual workflow
        step_names = [step["name"] for step in workflow["steps"]]
        st.session_state.manual_workflow = step_names

        # Set automatic execution flags
        auto_flags = {}
        for step in workflow["steps"]:
            if step.get("automatic", False):
                auto_flags[step["name"]] = True

        st.session_state.workflow_auto_flags = auto_flags
        st.session_state.current_workflow_name = workflow_name

        # Set ML model choice for this workflow
        ml_model_choice = workflow.get("ml_model_choice")
        if ml_model_choice:
            st.session_state.workflow_ml_model_choice = ml_model_choice
            st.session_state.optimization_model_choice = ml_model_choice

        st.success(f"Workflow '{workflow_name}' applied!")


def _reset_workflow_state() -> None:
    st.session_state.workflow_active = True
    st.session_state.workflow_step = "hypothesis"
    st.session_state.workflow_completed = False
    st.session_state.workflow_experiment_started = False
    st.session_state.workflow_experiment_completed = False
    st.session_state.workflow_experiment_outputs = None
    st.session_state.workflow_curve_fitting_completed = False
    st.session_state.experimental_outputs = None
    st.session_state.hypothesis_ready = False
    st.session_state.stop_hypothesis = False
    st.session_state.stage = "initial"


tab_run, tab_build = st.tabs(["Run Workflow", "Build Workflow"])

with tab_run:
    st.subheader("🚀 Workflow Runner")
    st.markdown("Start an end-to-end workflow: Hypothesis → Experiment → Curve Fitting → ML Models → Analysis.")

    st.markdown("#### 🧭 Routing")
    routing_mode = st.segmented_control(
        "Routing Mode",
        options=["Autonomous (LLM)", "Manual"],
        default=st.session_state.get("routing_mode", "Autonomous (LLM)"),
    )
    st.session_state.routing_mode = routing_mode

    if routing_mode == "Manual":
        st.info("Manual routing uses the active workflow order from the Builder tab.")
        current_workflow = st.session_state.get("manual_workflow", [])
        if current_workflow:
            st.caption(f"Current manual workflow: {', '.join(current_workflow)}")
        else:
            st.warning("No manual workflow applied yet. Build and apply a workflow first.")

        if st.button("Reset Workflow Progress", use_container_width=True):
            st.session_state.workflow_index = 0
            st.success("Workflow progress reset. The next routed call will start at Step 1.")

    st.divider()

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

    if not st.session_state.get("workflow_active", False):
        st.info("Start the workflow to begin the Hypothesis Agent.")
    else:
        st.markdown(
            f"**Current Step:** `{st.session_state.workflow_step}`"
        )

        if st.session_state.workflow_step == "hypothesis":
            st.subheader("🧠 Hypothesis Agent")
            st.switch_page("pages/hypothesis.py")

        if st.session_state.workflow_step == "experiment":
            st.subheader("🧪 Experiment Agent")
            st.switch_page("pages/experiment.py")

        if st.session_state.workflow_step == "curve_fitting":
            st.subheader("📈 Curve Fitting")
            st.switch_page("pages/curve_fitting.py")

        if st.session_state.workflow_step == "ml_models":
            st.subheader("🤖 ML Models")
            st.switch_page("pages/ml_models.py")

        if st.session_state.workflow_step == "analysis":
            st.subheader("🔎 Analysis Agent")
            st.switch_page("pages/analysis.py")

with tab_build:
    st.subheader("🏗️ Workflow Builder")
    st.markdown("Create custom workflows and mark steps for automatic execution.")

    # Sidebar for workflow management
    with st.sidebar:
        st.header("💾 Saved Workflows")

        # List saved workflows
        if st.session_state.workflows:
            for wf_name in st.session_state.workflows.keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"Load {wf_name}", key=f"load_{wf_name}", use_container_width=True):
                        workflow = load_workflow(wf_name)
                        if workflow:
                            st.session_state.current_workflow = workflow
                            st.session_state.workflow_steps = workflow["steps"]
                            # Restore ML model choice if it exists
                            if workflow.get("ml_model_choice"):
                                st.session_state.workflow_ml_model_choice = workflow["ml_model_choice"]
                            st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{wf_name}", help=f"Delete {wf_name}"):
                        del st.session_state.workflows[wf_name]
                        if st.session_state.get("current_workflow_name") == wf_name:
                            st.session_state.current_workflow = {"name": "Default Workflow", "steps": []}
                            st.session_state.workflow_steps = []
                        st.rerun()
        else:
            st.info("No saved workflows yet.")

        st.divider()

        # Create new workflow
        if st.button("New Workflow", use_container_width=True):
            st.session_state.current_workflow = {"name": "New Workflow", "steps": []}
            st.session_state.workflow_steps = []
            st.rerun()

    # Main workflow builder
    st.header("🧱 Build Your Workflow")

    # Workflow name
    col_name1, col_name2 = st.columns([3, 1])
    with col_name1:
        workflow_name = st.text_input(
            "Workflow Name",
            value=st.session_state.current_workflow.get("name", "My Workflow"),
            key="workflow_name_input",
        )

    with col_name2:
        if st.button("Save Workflow", use_container_width=True):
            if workflow_name and st.session_state.workflow_steps:
                # Get ML model choice if ML Models step exists
                ml_model_choice = None
                if "ML Models" in [s["name"] for s in st.session_state.workflow_steps]:
                    ml_model_choice = st.session_state.get("workflow_ml_model_choice") or st.session_state.get("optimization_model_choice")

                save_workflow(workflow_name, st.session_state.workflow_steps, ml_model_choice)
                st.session_state.current_workflow["name"] = workflow_name
                st.success(f"Workflow '{workflow_name}' saved!")
            else:
                st.warning("Please add steps to your workflow before saving.")

    st.divider()

    # ML Model Selection (if ML Models step is in workflow)
    if "ML Models" in [s["name"] for s in st.session_state.workflow_steps]:
        st.subheader("🤖 ML Model Configuration")
        st.info("Select which ML model to use when this workflow runs the ML Models step.")

        # Available ML models (matching ml_models.py)
        available_ml_models = [
            "Single-objective GP (scikit-learn)",
            "Dual-objective GP (PyTorch)",
            "Monte Carlo Decision Tree (external)"
        ]

        # Get current choice or default
        current_choice = st.session_state.get("workflow_ml_model_choice") or st.session_state.get("optimization_model_choice")
        if current_choice not in available_ml_models:
            current_choice = available_ml_models[0]

        ml_model_choice = st.selectbox(
            "ML Model for this Workflow:",
            available_ml_models,
            index=available_ml_models.index(current_choice) if current_choice in available_ml_models else 0,
            help="This model will be used automatically when the ML Models step runs in this workflow.",
            key="workflow_ml_model_select"
        )

        st.session_state.workflow_ml_model_choice = ml_model_choice

        st.divider()

    # Workflow steps builder
    st.subheader("🧩 Workflow Steps")

    if not st.session_state.workflow_steps:
        st.info("Add steps to your workflow using the controls below.")

    # Display current steps
    for idx, step in enumerate(st.session_state.workflow_steps):
        with st.container():
            col_step1, col_step2, col_step3, col_step4 = st.columns([3, 2, 1, 1])

            with col_step1:
                step_info = AVAILABLE_STEPS.get(step["name"], {})
                st.markdown(f"**{idx + 1}. {step['name']}**")
                if step_info.get("description"):
                    st.caption(step_info["description"])

            with col_step2:
                automatic = step.get("automatic", False)
                can_auto = step_info.get("can_auto", False)

                if can_auto:
                    auto_status = st.checkbox(
                        "Auto-execute",
                        value=automatic,
                        key=f"auto_{idx}",
                        help="This step will run automatically when reached",
                    )
                    st.session_state.workflow_steps[idx]["automatic"] = auto_status
                else:
                    st.caption("Manual step")

            with col_step3:
                if idx > 0:
                    if st.button("Up", key=f"up_{idx}", help="Move up"):
                        st.session_state.workflow_steps[idx], st.session_state.workflow_steps[idx - 1] = (
                            st.session_state.workflow_steps[idx - 1],
                            st.session_state.workflow_steps[idx],
                        )
                        st.rerun()
                if idx < len(st.session_state.workflow_steps) - 1:
                    if st.button("Down", key=f"down_{idx}", help="Move down"):
                        st.session_state.workflow_steps[idx], st.session_state.workflow_steps[idx + 1] = (
                            st.session_state.workflow_steps[idx + 1],
                            st.session_state.workflow_steps[idx],
                        )
                        st.rerun()

            with col_step4:
                if st.button("Remove", key=f"remove_{idx}", help="Remove step"):
                    st.session_state.workflow_steps.pop(idx)
                    st.rerun()

            st.divider()

    # Add new step
    st.subheader("🧩 Add Step")

    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        available_to_add = [
            name
            for name in AVAILABLE_STEPS.keys()
            if name not in [s["name"] for s in st.session_state.workflow_steps]
        ]

        if available_to_add:
            new_step_name = st.selectbox(
                "Select step to add",
                options=available_to_add,
                key="new_step_select",
            )
        else:
            st.info("All available steps have been added to the workflow.")
            new_step_name = None

    with col_add2:
        if st.button("Add Step", use_container_width=True, disabled=not new_step_name):
            if new_step_name:
                step_info = AVAILABLE_STEPS[new_step_name]
                new_step = {
                    "name": new_step_name,
                    "automatic": False,  # Default to manual
                    "description": step_info.get("description", ""),
                }
                st.session_state.workflow_steps.append(new_step)
                st.rerun()

    st.divider()

    # Workflow actions
    st.subheader("⚙️ Workflow Actions")

    col_action1, col_action2, col_action3 = st.columns(3)

    with col_action1:
        if st.button("Apply Workflow", use_container_width=True, type="primary"):
            if workflow_name and st.session_state.workflow_steps:
                # Save current workflow
                save_workflow(workflow_name, st.session_state.workflow_steps)
                # Apply it
                apply_workflow(workflow_name)
                st.rerun()
            else:
                st.warning("Please add steps to your workflow first.")

    with col_action2:
        if st.button("Reset Workflow", use_container_width=True):
            st.session_state.workflow_steps = []
            st.session_state.current_workflow = {"name": "Default Workflow", "steps": []}
            st.rerun()

    with col_action3:
        # Export workflow as JSON
        if st.session_state.workflow_steps:
            workflow_json = json.dumps(
                {
                    "name": workflow_name,
                    "steps": st.session_state.workflow_steps,
                },
                indent=2,
            )
            st.download_button(
                "Export JSON",
                workflow_json,
                file_name=f"{workflow_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

    # Workflow preview
    if st.session_state.workflow_steps:
        st.divider()
        st.subheader("👀 Workflow Preview")

        preview_cols = st.columns(len(st.session_state.workflow_steps))
        for idx, step in enumerate(st.session_state.workflow_steps):
            with preview_cols[idx]:
                auto_badge = "AUTO" if step.get("automatic", False) else "MANUAL"
                st.markdown(f"**{idx + 1}.** {auto_badge} {step['name']}")

        # Show automatic execution summary
        auto_steps = [s["name"] for s in st.session_state.workflow_steps if s.get("automatic", False)]
        if auto_steps:
            st.info(f"**Automatic steps:** {', '.join(auto_steps)}")
        else:
            st.info("**All steps are manual** - each step will require user interaction.")

    # Instructions
    with st.expander("📖 How to Use Workflow Builder"):
        st.markdown("""
        ### Creating a Workflow

        1. **Name your workflow** - Enter a descriptive name
        2. **Add steps** - Select steps from the dropdown and click "Add Step"
        3. **Configure automation** - For steps that support it, check "Auto-execute"
        4. **Reorder steps** - Use the Up and Down buttons to reorder
        5. **Save workflow** - Click "Save Workflow" to save for later use
        6. **Apply workflow** - Click "Apply Workflow" to activate it

        ### Automatic Execution

        Steps marked as AUTO will run automatically when reached:
        - **Curve Fitting**: Runs automatically if data files are ready
        - **ML Models**: Runs automatically after curve fitting completes (if enabled)
        - **Analysis Agent**: Runs automatically after ML/curve fitting completes

        ### Workflow Management

        - **Load**: Click a workflow name in the sidebar to load it
        - **Delete**: Click Delete next to a workflow name to delete it
        - **Export**: Export workflows as JSON for backup/sharing
        - **Reset**: Clear current workflow and start fresh

        ### Tips

        - Start with manual steps (Hypothesis, Experiment) to ensure quality
        - Use automatic execution for data processing steps
        - Save multiple workflows for different experiment types
        - Export workflows to share with team members
        """)
