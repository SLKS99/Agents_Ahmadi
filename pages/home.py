import streamlit as st
import json
import time
from pathlib import Path
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

# Check for auto-triggered file from watcher and navigate to curve fitting
trigger_info_file = Path(__file__).parent.parent / "watcher_trigger_info.json"
if trigger_info_file.exists():
    try:
        with open(trigger_info_file, 'r') as f:
            trigger_info = json.load(f)
        
        trigger_time = trigger_info.get("trigger_time", 0)
        time_since_trigger = time.time() - trigger_time
        
        # Auto-navigate if triggered within last 5 minutes (300 seconds)
        # This gives more time for curve fitting to complete
        if time_since_trigger < 300:
            triggered_file = trigger_info.get("triggered_file", "")
            st.session_state.watcher_auto_triggered_file = triggered_file
            st.session_state.watcher_auto_trigger_time = trigger_time
            
            # Show a brief message before navigating
            st.info("**Auto-triggered curve fitting detected!** Navigating to results...")
            time.sleep(0.5)  # Brief pause for user to see the message
            
            # Navigate to curve fitting page
            st.switch_page("pages/curve_fitting.py")
    except Exception:
        pass

st.set_page_config(layout="centered")
st.title("🤖 Multi-Agent AI Framework")

tabs = st.tabs(["Getting Started", "Navigation", "System Overview", "Configuration"])

with tabs[0]:
    st.markdown("""
    
    ### 🚀 Getting Started

    1. **Configure API Key**: Go to Settings → General and enter your Google Gemini API key
    2. **Configure Experiment Settings**: Go to Settings → Experiment and configure Jupyter upload options
    3. **Configure Watcher (Optional)**: Go to Watcher Control → Configuration to set the watch directory and enable the watcher
    4. **Start with Hypothesis**: Navigate to the Hypothesis page and enter your research question
    5. **Refine Your Thinking**: Select from generated options and iterate on your reasoning
    6. **Generate Hypothesis**: Once satisfied, generate your hypothesis
    7. **Plan Experiments**: Use the Experiment Agent to create experimental protocols
    8. **Analyze Data**: Upload data files to the Curve Fitting Agent for analysis
    9. **Run ML Models**: Use the ML Models page to train models and generate recommendations
    10. **Review History**: Check the History page to review all interactions and export data
    """)

with tabs[1]:
    st.markdown("### 🧭 Navigation")
    st.write("""
    - **Home**: This introduction page
    - **Dashboard**: System analytics, performance metrics, and agent usage statistics
    - **Hypothesis**: Interactive hypothesis generation and refinement
    - **Experiment**: Experimental planning and protocol generation
    - **Curve Fitting**: Data analysis and curve fitting
    - **ML Models**: Gaussian Process models for prediction and exploration
    - **Analysis**: Analyze results in relation to hypothesis and experiments
    - **Workflow**: Run and build workflows with routing and automation controls
    - **Watcher Control**: Configure and control the watcher server and directory
    - **Settings**: Configuration for API keys, experiment settings, and cache management
    - **History**: Complete interaction history with per-agent tabs and export options
    - **Export Data**: Download your interactions and results for documentation and sharing
    """)

with tabs[2]:
    st.markdown("""
    ### 📌 Table of Contents
    - [System Overview](#system-overview)
      - [Hypothesis Agent](#hypothesis-agent)
      - [Experiment Agent](#experiment-agent)
      - [Curve Fitting Agent](#curve-fitting-agent)
      - [ML Models (Gaussian Process)](#ml-models-gaussian-process)
      - [Analysis Agent](#analysis-agent)
      - [Router Agent](#router-agent)
      - [Watcher Agent](#watcher-agent)
      - [Workflow and Automation](#workflow-and-automation)
    - [Configuration Tips](#configuration-tips)
    """)
    st.divider()

    st.markdown("### 🧩 System Overview")
    st.write("This application consists of several interconnected agents that work together to support the research process:")
    st.markdown("""
    #### 🧠 Hypothesis Agent
The Hypothesis Agent uses Socratic questioning and tree-of-thought (TOT) reasoning to help you:
- **Clarify your research questions** through AI-powered question refinement
- **Explore multiple lines of reasoning** by generating distinct thought paths
- **Synthesize hypotheses** from your selected reasoning paths
- **Analyze and evaluate** your hypotheses for scientific rigor

**Workflow:**
1. Enter your initial research question
2. Review the AI-clarified question and probing questions
3. Select from three distinct lines of thought
4. Iteratively refine your thinking through continuation options
5. Generate a comprehensive hypothesis
6. Receive analysis and evaluation of your hypothesis
""")
    st.divider()
    st.markdown("""#### 🧪 Experiment Agent
The Experiment Agent transforms your hypotheses into actionable experimental plans:
- **Generates experimental protocols** based on your hypothesis and constraints
- **Creates worklists** for automated liquid handling systems (Opentrons, Tecan)
- **Designs plate layouts** for multi-well experiments
- **Produces executable protocols** for robotic systems

**Features:**
- Configurable experimental constraints (techniques, equipment, parameters)
- Support for multiple plate formats (96-well, 384-well, 24-well)
- Custom material and parameter specifications
- Integration with Jupyter notebooks for protocol execution""")
    st.divider()
    st.markdown("""
#### 📈 Curve Fitting Agent
The Curve Fitting Agent analyzes experimental data:
- **Processes luminescence data** from CSV files
- **Fits multi-peak Gaussian models** to spectral data
- **Generates visualization plots** of data and fits
- **Provides quantitative analysis** with R² scores and peak parameters

**Capabilities:**
- Automatic peak detection and parameter optimization
- Configurable wavelength ranges and read numbers
- Support for multiple wells and time-resolved measurements
- Export of fitting results and visualizations
""")
    st.divider()
    st.markdown("""
#### 🤖 ML Models (Gaussian Process)
The ML Models page provides machine learning capabilities for experimental optimization:
- **Gaussian Process regression** to predict properties from composition and curve fitting features
- **Uncertainty quantification** to assess prediction confidence
- **Exploration recommendations** using acquisition functions (UCB, EI, PI)
- **Integration with Analysis Agent** to interpret ML predictions and suggest next experiments

**Features:**
- Train GP models on curve fitting results and composition data
- Visualize predictions with uncertainty bounds
- Generate candidate compositions for exploration
- Cross-validation for model evaluation
- Seamless integration with the analysis workflow
""")
    st.divider()
    st.markdown("""
#### 🔎 Analysis Agent
The Analysis Agent evaluates results and guides next steps:
- **Relates curve fitting results** to the original hypothesis and experimental plan
- **Evaluates hypothesis status** (confirmed, needs revision, rejected, or needs more data)
- **Determines if more experiments are needed** with specific recommendations
- **Provides literature-backed explanations** of results using established scientific principles
- **Assesses impact and significance** of findings

**Features:**
- Comprehensive analysis comparing results to predictions
- Literature context explaining mechanisms and known examples
- Actionable recommendations for additional experiments
- Integration with hypothesis and experiment agents for workflow continuity
""")
    st.divider()
    st.markdown("""
#### 🧭 Router Agent
The Router Agent intelligently manages workflow transitions:
- **Confidence-based routing** selects the most appropriate agent based on context
- **LLM-based routing** uses AI analysis to determine next steps
- **Manual workflow control** allows you to define custom agent sequences
- **Automatic progression** through hypothesis → experiment → analysis workflows
""")
    st.divider()
    st.markdown("""
#### 👀 Watcher Agent
The Watcher Agent monitors your filesystem for changes:
- **Detects new output files** from agent executions
- **Triggers next agents** automatically when files are created
- **Supervises workflows** by monitoring file system events
- **Cross-platform support** (Windows, macOS, Linux)
""")
    st.divider()
    st.markdown("""
#### 🔁 Workflow and Automation
The Workflow system coordinates multi-agent execution:
- **Workflow Runner** starts and tracks end-to-end pipelines
- **Workflow Builder** creates custom step sequences and auto-execution flags
- **Routing controls** support autonomous and manual execution modes
- **ML automation** runs models after curve fitting using captured inputs
""")

with tabs[3]:
    st.markdown("""
    ### ⚙️ Configuration Tips
    Use these pages to keep the system consistent:
    - **Settings → General** sets your API key for all agents and servers
    - **Settings → Experiment** configures Jupyter uploads and experiment paths
    - **Watcher Control → Configuration** sets watch directory, port, and enable flags
    - **Workflow → Run/Build** controls routing, workflow order, and automation
    """)
