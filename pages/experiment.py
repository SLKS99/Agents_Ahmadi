import streamlit as st
from tools.memory import MemoryManager
from agents.experiment_agent import ExperimentAgent

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("Experiment Agent")

# Optional manual trigger for the Experiment Agent
st.markdown("Use the controls below to configure experimental constraints or trigger the agent directly.")

def _run_experiment_agent() -> None:
    # Increment usage metrics
    st.session_state.agent_usage_counts["experiment"] = (
        st.session_state.agent_usage_counts.get("experiment", 0) + 1
    )

    # Snapshot current session state for history/analytics
    memory.snapshot_session_state("before_experiment_agent_run")

    # Instantiate and run the experiment agent using current constraints
    agent = ExperimentAgent(
        name="Experiment Agent",
        desc="Generates experimental plans and automation artifacts from the current hypothesis and constraints.",
        params_const=st.session_state.experimental_constraints,
    )
    agent.run_agent(memory)

# Manual Input Section for LLM Generation Components
with st.expander("Manual Input (Optional - Use if no hypothesis available)", expanded=False):
    st.markdown("**Provide the following components manually to generate experimental plans without a hypothesis:**")
    
    manual_clarified_question = st.text_area(
        "Clarified Question:",
        value=st.session_state.get("manual_clarified_question", ""),
        placeholder="e.g., How can we optimize the phase stability of perovskite materials?",
        help="The clarified research question"
    )
    
    manual_socratic_questions = st.text_area(
        "Socratic Questions (Probing Questions):",
        value=st.session_state.get("manual_socratic_questions", ""),
        placeholder="e.g., What specific mechanisms contribute to phase instability?\nWhat role do organic cations play?\nHow do temperature and humidity affect phase transitions?",
        help="3-5 probing questions that explore the research question",
        height=150
    )
    
    manual_socratic_answers = st.text_area(
        "Socratic Answers (Optional):",
        value=st.session_state.get("manual_socratic_answers", ""),
        placeholder="Detailed answers to the probing questions above",
        help="Optional: Answers to the probing questions for deeper context",
        height=150
    )
    
    manual_thoughts = st.text_area(
        "Three Lines of Thought (Optional):",
        value=st.session_state.get("manual_thoughts", ""),
        placeholder="Distinct Line of Thought 1: [First approach]\nDistinct Line of Thought 2: [Second approach]\nDistinct Line of Thought 3: [Third approach]",
        help="Optional: Three distinct lines of thought or mini-hypotheses",
        height=150
    )
    
    manual_hypothesis = st.text_area(
        "Hypothesis (Optional):",
        value=st.session_state.get("manual_hypothesis", ""),
        placeholder="A detailed hypothesis statement with predictions and tests",
        help="Optional: A full hypothesis statement",
        height=150
    )
    
    # Save manual inputs to session state
    if st.button("Save Manual Inputs", use_container_width=True):
        st.session_state.manual_clarified_question = manual_clarified_question
        st.session_state.manual_socratic_questions = manual_socratic_questions
        st.session_state.manual_socratic_answers = manual_socratic_answers
        st.session_state.manual_thoughts = manual_thoughts
        st.session_state.manual_hypothesis = manual_hypothesis
        st.success("Manual inputs saved!")
        st.rerun()

# Experimental Parameters and Constraints Section (without form to avoid errors)
with st.expander("Experimental Parameters and Constraints", expanded=False):
    # Techniques
    techniques = st.multiselect(
        "Experimental Techniques:",
        ["in-situ PL", "spin coating", "absorbance spectroscopy", "XRD", "SEM", "TEM", "UV-Vis",
         "photoluminescence", "time-resolved PL", "impedance spectroscopy"],
        default=st.session_state.experimental_constraints.get("techniques", [])
    )

    # Equipment
    equipment = st.multiselect(
        "Available Equipment:",
        ["spin bot", "pipetting robot", "glove box", "solar simulator", "spectrometer", "microscope",
         "thermal evaporator", "spin coater", "Tecan liquid handler", "Opentrons liquid handler"],
        default=st.session_state.experimental_constraints.get("equipment", [])
    )

    # Liquid Handling Configuration
    st.markdown("#### Liquid Handling Setup")
    col_lh1, col_lh2 = st.columns(2)

    with col_lh1:
        # Instruments multiselect - ensure no duplicates
        available_instruments = ["Tecan", "Opentrons", "manual pipettes", "multichannel pipettes"]
        current_instruments = st.session_state.experimental_constraints.get("liquid_handling", {}).get("instruments", [])
        # Remove duplicates from current instruments
        current_instruments = list(dict.fromkeys(current_instruments))  # Preserves order, removes duplicates

        lh_instruments = st.multiselect(
            "Liquid Handling Instruments:",
            available_instruments,
            default=current_instruments
        )
        # Ensure no duplicates in the selected list
        lh_instruments = list(dict.fromkeys(lh_instruments))

        plate_format = st.selectbox(
            "Plate Format:",
            ["96-well", "384-well", "24-well"],
            index=["96-well", "384-well", "24-well"].index(
                st.session_state.experimental_constraints.get("liquid_handling", {}).get("plate_format", "96-well"))
        )

    with col_lh2:
        max_volume = st.slider(
            "Max Volume per Mixture (µL):",
            min_value=10,
            max_value=200,
            value=st.session_state.experimental_constraints.get("liquid_handling", {}).get("max_volume_per_mixture", 50)
        )

        # Materials input - allow typing and adding custom materials
        st.markdown("**Available Materials:**")

        # Show current materials as chips
        current_materials = st.session_state.experimental_constraints.get("liquid_handling", {}).get("materials", [])
        if current_materials:
            # Display as chips/badges
            material_chips = " ".join([f"`{mat}`" for mat in current_materials])
            st.markdown(f"Current: {material_chips}")

        # Text input for adding new material
        st.markdown("**Add New Material:**")
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            new_material = st.text_input(
                "Material name:",
                placeholder="e.g., Cs, BDA, 5AVA, or custom name",
                key="new_material_input",
                label_visibility="visible"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
            add_btn = st.button("➕ Add", key="add_material_btn", use_container_width=True)

        # Process new material input
        materials = list(current_materials) if current_materials else []

        if add_btn and new_material and new_material.strip():
            material_to_add = new_material.strip()
            # Add if not already in list (case-insensitive check)
            if material_to_add.lower() not in [m.lower() for m in materials]:
                materials.append(material_to_add)
                st.session_state.experimental_constraints["liquid_handling"]["materials"] = materials
                st.success(f"Added: {material_to_add}")
                st.rerun()
            else:
                st.warning(f"Material '{material_to_add}' already exists")

        # Preset materials for quick selection
        st.markdown("**Quick Add:**")
        preset_cols = st.columns(4)
        preset_materials = ["Cs", "BDA", "BDA_2", "5AVA", "FAPbI3", "Material 1", "Material 2", "Material 3"]
        for i, preset in enumerate(preset_materials):
            col_idx = i % 4
            with preset_cols[col_idx]:
                if st.button(f"+ {preset}", key=f"add_preset_{preset}", use_container_width=True):
                    if preset.lower() not in [m.lower() for m in materials]:
                        materials.append(preset)
                        st.session_state.experimental_constraints["liquid_handling"]["materials"] = materials
                        st.rerun()
                    else:
                        st.warning(f"'{preset}' already added")

        # Remove materials option
        if materials:
            st.markdown("**Remove Materials:**")
            remove_cols = st.columns(min(len(materials), 4))
            for i, mat in enumerate(materials):
                col_idx = i % 4
                with remove_cols[col_idx]:
                    if st.button(f"❌ {mat}", key=f"remove_{mat}", use_container_width=True):
                        materials.remove(mat)
                        st.session_state.experimental_constraints["liquid_handling"]["materials"] = materials
                        st.rerun()

        csv_path = st.text_input(
            "CSV File Path (for Opentrons):",
            value=st.session_state.experimental_constraints.get("liquid_handling", {}).get("csv_path",
                                                                                       "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"),
            help="Path where CSV file will be stored on Opentrons robot"
        )

        # Parameters
        parameters = st.multiselect(
            "Key Parameters to Optimize:",
            ["spin speed", "concentration", "temperature", "humidity", "annealing time", "layer thickness",
             "mixing ratio", "deposition rate"],
            default=st.session_state.experimental_constraints.get("parameters", [])
        )

        # Focus Areas
        focus_areas = st.multiselect(
            "Primary Focus Areas:",
            ["device performance", "material stability", "process optimization", "characterization", "scaling",
             "cost reduction"],
            default=st.session_state.experimental_constraints.get("focus_areas", [])
        )

        # Save button (replaces form submit)
        if st.button("Save Constraints and Parameters", type="primary", use_container_width=True):
            st.session_state.experimental_constraints = {
                "techniques": techniques,
                "equipment": equipment,
                "parameters": parameters,
                "focus_areas": focus_areas,
                "liquid_handling": {
                    "max_volume_per_mixture": max_volume,
                    "instruments": list(dict.fromkeys(lh_instruments)),  # Ensure no duplicates
                    "plate_format": plate_format,
                    "materials": list(dict.fromkeys(materials)),  # Ensure no duplicates
                    "csv_path": csv_path if csv_path else "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"
                }
            }
            st.success("Parameters and constraints saved!")
            st.rerun()

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

# Run Experiment Agent button
if st.button("Run Experiment Agent", type="primary", use_container_width=True):
    _run_experiment_agent()

# Workflow auto-run: execute experiment once when routed here
if (
    st.session_state.get("workflow_active")
    and st.session_state.get("workflow_step") == "experiment"
    and not st.session_state.get("workflow_experiment_started")
):
    st.session_state.workflow_experiment_started = True
    _run_experiment_agent()

# Workflow transition: move to Curve Fitting after outputs exist
if (
    st.session_state.get("workflow_active")
    and st.session_state.get("experimental_outputs")
    and not st.session_state.get("workflow_experiment_completed")
):
    st.session_state.workflow_experiment_outputs = (
        st.session_state.experimental_outputs
    )
    st.session_state.workflow_experiment_completed = True
    st.session_state.workflow_step = "curve_fitting"
    st.switch_page("pages/curve_fitting.py")
