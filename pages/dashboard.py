import streamlit as st
import psutil
import time
import json
from datetime import datetime
from io import BytesIO
from tools.memory import MemoryManager

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

memory = MemoryManager()
memory.init_session()

st.title("📊 Dashboard")
st.set_page_config(layout="wide")

# System Performance Metrics
st.markdown("#### 📈 System Performance")
cols = st.columns(5)

if "metrics_prev" not in st.session_state:
    st.session_state.metrics_prev = {
        "prev_cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "uptime": 0,
        "interactions": 0,
        "events": 0,
    }

# Get current CPU usage
current_cpu = psutil.cpu_percent(interval=None)
cpu_delta = current_cpu - st.session_state.metrics_prev["prev_cpu"]

cols[0].metric(
    label="CPU Usage",
    value=f"{current_cpu:.1f}%",
    delta=f"{cpu_delta:+.1f}%",
    border=True
)
st.session_state.metrics_prev["prev_cpu"] = current_cpu

# Memory metrics
ram = psutil.virtual_memory()
ram_delta = ram.percent - st.session_state.metrics_prev["ram"]
cols[1].metric(
    label="Memory Usage",
    value=f"{ram.percent:.1f}%",
    delta=f"{ram_delta:+.1f}%",
    border=True
)
st.session_state.metrics_prev["ram"] = ram.percent

# Disk usage
try:
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    cols[2].metric(
        label="Disk Usage",
        value=f"{disk_percent:.1f}%",
        border=True
    )
except Exception:
    cols[2].metric(label="Disk Usage", value="N/A", border=True)

# Uptime
uptime = int(time.time() - st.session_state.start_time)
uptime_delta = uptime - st.session_state.metrics_prev["uptime"]
uptime_hours = uptime // 3600
uptime_mins = (uptime % 3600) // 60
cols[3].metric(
    label="Uptime",
    value=f"{uptime_hours}h {uptime_mins}m",
    delta=f"{uptime_delta:+d}s",
    border=True
)
st.session_state.metrics_prev["uptime"] = uptime

# Total events
total_events = len(st.session_state.get("conversation_events", []))
events_delta = total_events - st.session_state.metrics_prev["events"]
cols[4].metric(
    label="Total Events",
    value=total_events,
    delta=f"{events_delta:+d}",
    border=True
)
st.session_state.metrics_prev["events"] = total_events

st.markdown("---")
# Agent Usage Analytics
st.markdown("#### 🤖 Agent Usage Analytics")
usage = st.session_state.get("agent_usage_counts", {})
total_agent_usage = sum(usage.values()) if usage else 0
conversation_events = st.session_state.get("conversation_events", [])

def _count_events_by_mode(mode_name: str) -> int:
    return len([e for e in conversation_events if e.get("mode") == mode_name])

def _count_events_by_type(type_name: str) -> int:
    return len([e for e in conversation_events if e.get("type") == type_name])

if total_agent_usage > 0:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Agent usage metrics
        agent_cols = st.columns(len(usage))
        for i, (agent, count) in enumerate(usage.items()):
            if count > 0:
                percentage = (count / total_agent_usage) * 100
                agent_cols[i].metric(
                    label=agent.replace("_", " ").title(),
                    value=count,
                    delta=f"{percentage:.1f}% of total"
                )
            else:
                agent_cols[i].metric(
                    label=agent.replace("_", " ").title(),
                    value=0
                )
    
    with col2:
        # Most used agent
        if usage:
            most_used = max(usage.items(), key=lambda x: x[1])
            st.metric(
                label="Most Used Agent",
                value=most_used[0].replace("_", " ").title(),
                delta=f"{most_used[1]} times"
            )
else:
    st.info("No agent usage data yet. Start using agents to see analytics here.")

st.markdown("---")
# Watcher Status
st.markdown("#### 👀 Watcher Status")
watcher_cols = st.columns(4)

watcher_enabled = st.session_state.get("watcher_enabled", False)
watcher_server_url = st.session_state.get("watcher_server_url", "Not set")
watcher_watch_dir = st.session_state.get("watcher_watch_dir", "Not set")
watcher_last_trigger = st.session_state.get("watcher_auto_trigger_time")

watcher_cols[0].metric("Watcher Enabled", "Yes" if watcher_enabled else "No")
watcher_cols[1].metric("Watcher Events", _count_events_by_type("watcher"))
watcher_cols[2].metric("Watcher Server", watcher_server_url)
watcher_cols[3].metric(
    "Last Trigger",
    datetime.fromtimestamp(watcher_last_trigger).strftime("%Y-%m-%d %H:%M:%S")
    if watcher_last_trigger
    else "N/A",
)

st.caption(f"Watch Directory: `{watcher_watch_dir}`")

st.markdown("---")
# File Uploads
st.markdown("#### 📁 Uploaded Files")
uploaded_files = st.session_state.get("uploaded_files", [])

if uploaded_files:
    file_data = []
    for file_info in uploaded_files:
        file_data.append([
            file_info.get("name", "Unknown"),
            file_info.get("path", "N/A"),
            file_info.get("timestamp", "N/A")
        ])
    
    st.dataframe(
        file_data,
        column_config={
            "0": "Filename",
            "1": "Path",
            "2": "Upload Time"
        },
        hide_index=True
    )
    st.metric("Total Files Uploaded", len(uploaded_files))
else:
    st.info("No files uploaded yet.")

st.markdown("---")
# Agent Reports Section
st.markdown("#### 📄 Agent Analysis Reports")

def generate_pdf_report(agent_name: str = "All Agents") -> bytes:
    """Generate a PDF report of agent outputs and interactions"""
    if not REPORTLAB_AVAILABLE:
        return b"PDF generation requires reportlab. Install with: pip install reportlab"
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
    )
    story.append(Paragraph(f"POLARIS Agent Report: {agent_name}", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # System Metrics
    story.append(Paragraph("System Metrics", styles['Heading2']))
    metrics_data = [
        ["Metric", "Value"],
        ["CPU Usage", f"{psutil.cpu_percent(interval=None):.1f}%"],
        ["Memory Usage", f"{psutil.virtual_memory().percent:.1f}%"],
        ["Uptime", f"{int((time.time() - st.session_state.start_time) // 3600)} hours"],
        ["Total Events", str(len(st.session_state.get("conversation_events", [])))],
    ]
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Agent Usage
    story.append(Paragraph("Agent Usage Statistics", styles['Heading2']))
    usage = st.session_state.get("agent_usage_counts", {})
    usage_data = [["Agent", "Usage Count"]]
    for agent, count in usage.items():
        usage_data.append([agent.replace("_", " ").title(), str(count)])
    
    usage_table = Table(usage_data)
    usage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(usage_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recent Interactions
    story.append(Paragraph("Recent Interactions", styles['Heading2']))
    events = st.session_state.get("conversation_events", [])
    
    if agent_name != "All Agents":
        events = [e for e in events if e.get("mode", "").lower() == agent_name.lower()]
    
    if events:
        # Show last 20 events
        recent_events = events[-20:]
        for event in recent_events:
            event_type = event.get("type", "unknown")
            timestamp = event.get("timestamp", "N/A")
            mode = event.get("mode", "unknown")
            story.append(Paragraph(f"[{timestamp}] {event_type.upper()} ({mode})", styles['Heading3']))
            payload = event.get("payload", {})
            if payload:
                payload_text = json.dumps(payload, indent=2)
                # Truncate very long payloads
                if len(payload_text) > 500:
                    payload_text = payload_text[:500] + "... (truncated)"
                story.append(Paragraph(payload_text, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    else:
        story.append(Paragraph("No interactions recorded.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Report generation section
col1, col2 = st.columns(2)

with col1:
    report_agent = st.selectbox(
        "Select Agent for Report",
        ["All Agents"] + [agent.replace("_", " ").title() for agent in usage.keys() if usage.get(agent, 0) > 0],
        key="report_agent_select"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if REPORTLAB_AVAILABLE:
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(report_agent)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"polaris_report_{report_agent.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    else:
        st.warning("PDF generation requires reportlab. Install with: pip install reportlab")

st.markdown("---")
# Additional Analytics
st.markdown("#### 📊 Additional Analytics")

col1, col2, col3 = st.columns(3)

with col1:
    # Hypothesis status
    hypothesis_ready = st.session_state.get("hypothesis_ready", False)
    last_hypothesis = st.session_state.get("last_hypothesis")
    if last_hypothesis:
        st.metric("Last Hypothesis", "Available" if hypothesis_ready else "In Progress")
    else:
        st.metric("Last Hypothesis", "None")

with col2:
    # Experimental outputs
    exp_outputs = st.session_state.get("experimental_outputs")
    st.metric("Experimental Outputs", "Available" if exp_outputs else "None")

with col3:
    # Routing mode
    routing_mode = st.session_state.get("routing_mode", "Autonomous (LLM)")
    st.metric("Routing Mode", routing_mode)

st.markdown("---")
# Workflow & Automation
st.markdown("#### 🔁 Workflow & Automation")
wf_cols = st.columns(4)

wf_cols[0].metric("Workflow Active", "Yes" if st.session_state.get("workflow_active") else "No")
wf_cols[1].metric("Workflow Step", st.session_state.get("workflow_step", "N/A"))
wf_cols[2].metric("Auto-ML After Curve Fitting", "On" if st.session_state.get("auto_ml_after_curve_fitting") else "Off")
wf_cols[3].metric("Analysis Ready", "Yes" if st.session_state.get("analysis_ready") else "No")

ml_model_choice = st.session_state.get("optimization_model_choice") or "Not set"
st.caption(f"ML Model Choice: `{ml_model_choice}`")

st.markdown("---")
# Session Statistics
st.markdown("#### 📌 Session Statistics")
stat_cols = st.columns(4)

stat_cols[0].metric("Total Interactions", len(st.session_state.get("conversation_events", [])))
stat_cols[1].metric("Uploaded Files", len(uploaded_files))
stat_cols[2].metric("Active Sessions", 1)  # Single session for now
stat_cols[3].metric("Workflow Progress", f"{st.session_state.get('workflow_index', 0)} steps")

st.markdown("---")
# ML & Analysis Activity
st.markdown("#### 🧠 ML & Analysis Activity")
ml_cols = st.columns(4)

ml_cols[0].metric("Curve Fitting Runs", _count_events_by_mode("curve fitting"))
ml_cols[1].metric("ML Models Runs", _count_events_by_mode("ml_models"))
ml_cols[2].metric("Analysis Runs", _count_events_by_mode("analysis"))
ml_cols[3].metric("ML Auto Runs", _count_events_by_type("ml_automation"))

