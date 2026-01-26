import os
import tempfile
import json
import re
from io import BytesIO

import pandas as pd
import streamlit as st
import numpy as np
from tools.memory import MemoryManager
from agents.curve_fitting_agent import CurveFittingAgent
import matplotlib.pyplot as plt
import time

memory = MemoryManager()
memory.init_session()

# Initialize session state for well selection
if "selected_wells" not in st.session_state:
    st.session_state.selected_wells = set()
if "plate_format" not in st.session_state:
    st.session_state.plate_format = "96-well (8x12)"

# Initialize curve fitting agent
agent = CurveFittingAgent()

st.set_page_config(layout="wide")
st.title("Curve Fitting")
st.markdown("Upload CSV files, adjust parameters, and generate fitted curves with interactive visualizations.")

if st.button("Clear Cache and Restart Program"):
    st.cache_data.clear()
    st.rerun()

workflow_outputs = st.session_state.get("workflow_experiment_outputs")
if workflow_outputs:
    with st.expander("Workflow Context (Experiment Outputs)", expanded=False):
        plan = workflow_outputs.get("plan") or ""
        worklist = workflow_outputs.get("worklist") or ""
        layout = workflow_outputs.get("layout") or ""
        protocol = workflow_outputs.get("protocol") or ""

        if plan:
            st.markdown("**Experimental Plan**")
            st.text_area("Plan", plan, height=120, disabled=True)
        if worklist:
            st.markdown("**Worklist (CSV)**")
            st.code(worklist, language="csv")
        if layout:
            st.markdown("**Plate Layout**")
            st.code(layout, language="text")
        if protocol:
            st.markdown("**Opentrons Protocol**")
            st.code(protocol, language="python")

def parse_reads_spec(spec: str, default_max: int) -> list:
    """Parse reads specification string into list of integers"""
    spec = spec.strip().lower()
    if spec == "odd":
        return [i for i in range(1, default_max+1) if i % 2 == 1]
    if spec == "even":
        return [i for i in range(1, default_max+1) if i % 2 == 0]
    if "-" in spec:
        a, b = spec.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        return list(range(min(a, b), max(a, b)+1))
    # comma-separated
    return [int(x.strip()) for x in spec.split(",") if x.strip()]

import re

READ_HEADER_PATTERN = re.compile(r"^Read\s+(\d+):EM\s+Spectrum\s*$", re.I)

def convert_excel_to_spectroscopy_csv(file_bytes: bytes, filename: str, start_row: int = 4, read_start_col: int = 1) -> bytes:
    """
    Convert Excel file with wavelength in column 0 and reads in columns 1+ 
    to the CSV format expected by the fitting agent.
    
    Args:
        file_bytes: Excel file bytes
        filename: Original filename
        start_row: Row where data starts (0-indexed, default 4)
        read_start_col: Column index where read/acquisition data starts (default 1, column 0 is wavelength)
    
    Returns:
        CSV file bytes in the format expected by the fitting agent
    """
    try:
        # Read Excel file
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=0, header=None)
        
        # Find where actual numeric data starts by checking for numeric values in column 0
        # Skip header rows (like "Wavelength [nm]", "Timestamp", etc.)
        data_start_row = start_row
        for i in range(min(10, len(df))):  # Check first 10 rows
            try:
                val = df.iloc[i, 0]
                if pd.notna(val):
                    numeric_val = pd.to_numeric(val, errors='coerce')
                    if pd.notna(numeric_val) and numeric_val > 100:  # Wavelengths are typically > 100nm
                        data_start_row = i
                        break
            except:
                continue
        
        # Extract wavelength column (column 0, starting from data_start_row)
        wavelength = df.iloc[data_start_row:, 0].copy()
        
        # Extract read columns (columns from read_start_col onward)
        read_data = df.iloc[data_start_row:, read_start_col:].copy()
        
        # Convert to numeric
        wavelength = pd.to_numeric(wavelength, errors='coerce')
        read_data = read_data.apply(pd.to_numeric, errors='coerce')
        
        # Remove rows with NaN wavelength
        valid_mask = ~wavelength.isna()
        wavelength = wavelength[valid_mask]
        read_data = read_data[valid_mask]
        
        # Use all available data rows (no artificial limit)
        # Note: User's plotting code uses 1:400, but we'll process all available data
        # If needed, users can filter by wavelength range in the UI
        
        # Create CSV format expected by fitting agent
        # Format: Read 1:EM Spectrum header, then wavelength and well columns
        # The fitting agent's _slice_block drops column 0, so we need:
        # - Column 0: "Read N:EM Spectrum" (will be in first col, then dropped)
        # - Empty line
        # - Column 0: "Wavelength" (will be dropped), Column 1: Well name (R1, R2, etc.)
        # - Data rows: wavelength in col 0 (dropped), intensity in col 1
        csv_lines = []
        
        # For each read column, create a "Read N:EM Spectrum" block
        num_reads = read_data.shape[1]
        
        for read_num in range(1, num_reads + 1):
            # Add read header - use consistent 3-column format
            # Column 0: Read header, Columns 1-2: empty (will be dropped)
            csv_lines.append(f"Read {read_num}:EM Spectrum,,")
            
            # Empty line after header - use 3 columns to match format
            csv_lines.append(",,,")
            
            # Create data block with wavelength and this read's data
            read_col = read_data.iloc[:, read_num - 1]
            
            # Create well names - use R1, R2, etc. format (now accepted by fitting agent)
            # YES, we're keeping BOTH formats:
            # - A1-H12 format for regular well plates (still fully supported)
            # - R1-R2 format for acquisitions/reads (newly added support)
            well_name = f"R{read_num}"
            
            # Build the data block header
            # Format: Column 0 = placeholder (dropped), Column 1 = Wavelength, Column 2 = Well name
            # After _slice_block drops col 0: Column 0 = Wavelength, Column 1 = Well name
            csv_lines.append(f"Placeholder,Wavelength,{well_name}")
            
            # Add data rows - all with 3 columns for consistency
            for idx in range(len(wavelength)):
                wl_val = wavelength.iloc[idx]
                intensity_val = read_col.iloc[idx]
                if pd.notna(wl_val) and pd.notna(intensity_val):
                    csv_lines.append(f"P,{wl_val},{intensity_val}")
            
            # Add empty line between reads - use 3 columns to match
            csv_lines.append(",,,")
        
        # Convert to bytes
        csv_content = "\n".join(csv_lines)
        return csv_content.encode('utf-8')
        
    except Exception as e:
        raise ValueError(f"Error converting Excel file: {e}")

def create_dummy_composition_csv(well_names: list) -> str:
    """
    Create a dummy composition CSV file for Excel-based analysis.
    Since Excel files don't have composition data, we create a placeholder.
    
    Args:
        well_names: List of well/acquisition names (e.g., ['R1', 'R2', ...])
    
    Returns:
        CSV content as string (UTF-8)
    """
    # Create a simple composition CSV with a dummy material
    csv_lines = []
    
    # Header row: Material name followed by all well/acquisition names
    csv_lines.append("Material," + ",".join(well_names))
    
    # Data row: Dummy material with all values set to 1.0
    csv_lines.append("Sample," + ",".join(["1.0"] * len(well_names)))
    
    csv_content = "\n".join(csv_lines)
    return csv_content

def parse_spectroscopy_csv(file_bytes: bytes, filename: str):
    """
    Parse CSV files with spectroscopy metadata headers (like Spectropus format).
    Looks for 'Read N:EM Spectrum' patterns and extracts data blocks.
    """
    try:
        # First try to read as regular CSV
        df = pd.read_csv(BytesIO(file_bytes), header=None)

        # Check if this is a spectroscopy file with metadata headers
        first_col = df.iloc[:, 0].astype(str)
        read_blocks = {}

        # Find all "Read N:EM Spectrum" headers
        for idx, val in first_col.items():
            m = READ_HEADER_PATTERN.match(val)
            if m:
                read_num = int(m.group(1))
                read_blocks[read_num] = idx

        if read_blocks:
            # This is a spectroscopy file - parse it properly
            parsed_reads = {}

            # Sort read blocks
            sorted_reads = sorted(read_blocks.items())

            for i, (read_num, start_row) in enumerate(sorted_reads):
                # Determine end row (next read block or end of file)
                if i + 1 < len(sorted_reads):
                    end_row = sorted_reads[i + 1][1]
                else:
                    end_row = len(df)

                # Extract this read's data block
                # Skip the header row and any empty rows
                block_start = start_row + 2  # Skip header and usually one empty row
                if block_start >= end_row:
                    continue

                # Find the actual data (look for rows that start with wavelength-like numbers)
                data_rows = []
                for row_idx in range(block_start, end_row):
                    row_data = df.iloc[row_idx]
                    first_val = str(row_data.iloc[0]).strip()

                    # Check if this looks like a data row (starts with number)
                    try:
                        float(first_val)
                        data_rows.append(row_idx)
                    except (ValueError, TypeError):
                        # Check if it's a header row (contains well names like A1, B2, etc.)
                        if re.match(r'^[A-H](?:[1-9]|1[0-2])$', first_val):
                            # This is a header row - next row should be data
                            continue
                        elif first_val == '' or first_val.lower() in ['nan', 'null']:
                            # Empty row
                            continue

                if data_rows:
                    # Extract the data block
                    block_df = df.iloc[data_rows[0]:data_rows[-1]+1].copy()

                    # Clean up the block
                    # Drop completely empty columns
                    block_df = block_df.dropna(axis=1, how='all')

                    # Try to find header row (well names or column headers like "Wavelength")
                    header_found = False
                    header_row_idx = None
                    
                    for row_idx in range(len(block_df)):
                        row = block_df.iloc[row_idx]
                        first_val = str(row.iloc[0]).strip().lower()

                        # Look for well name pattern in first column
                        if re.match(r'^[a-h](?:[1-9]|1[0-2])$', first_val):
                            # This row contains well names - use as header
                            block_df.columns = block_df.iloc[row_idx]
                            header_row_idx = row_idx
                            header_found = True
                            break
                        # Also check for common header names like "Wavelength"
                        elif first_val in ['wavelength', 'wl', 'lambda', 'nm']:
                            # This looks like a header row with column names
                            block_df.columns = block_df.iloc[row_idx]
                            header_row_idx = row_idx
                            header_found = True
                            break

                    if header_found and header_row_idx is not None:
                        # Remove the header row from data
                        block_df = block_df.iloc[header_row_idx + 1:].copy()
                    elif not header_found:
                        # No header found, use generic column names
                        block_df.columns = [f'col_{i}' for i in range(len(block_df.columns))]

                    # Convert to numeric where possible, but skip if column name itself is non-numeric
                    for col in block_df.columns:
                        try:
                            # Skip conversion if column name looks like a header string
                            col_str = str(col).strip().lower()
                            if col_str in ['wavelength', 'wl', 'lambda', 'nm']:
                                # This is likely the wavelength column - convert to numeric
                                block_df[col] = pd.to_numeric(block_df[col], errors='coerce')
                            else:
                                # Try to convert, but handle errors gracefully
                                block_df[col] = pd.to_numeric(block_df[col], errors='coerce')
                        except Exception as e:
                            # If conversion fails, leave as is
                            pass
                    
                    # Remove any rows where the first column is still a string header name
                    if len(block_df) > 0:
                        first_col_name = block_df.columns[0]
                        first_col_values = block_df.iloc[:, 0].astype(str).str.strip().str.lower()
                        # Remove rows where first column value matches common header names
                        header_names = ['wavelength', 'wl', 'lambda', 'nm']
                        mask = ~first_col_values.isin(header_names)
                        block_df = block_df[mask].copy()
                        
                        # Reset index after filtering
                        block_df = block_df.reset_index(drop=True)

                    parsed_reads[read_num] = block_df

            if parsed_reads:
                # Return the first read's data for compatibility
                # Note: all_reads metadata removed to avoid circular references
                first_read = min(parsed_reads.keys())
                return parsed_reads[first_read]

        # Not a spectroscopy file, fall back to regular CSV parsing
        df_regular = pd.read_csv(BytesIO(file_bytes))

        # Remove any rows where values match column header names (duplicate headers)
        if len(df_regular) > 0:
            for col in df_regular.columns:
                col_str = str(col).strip().lower()
                # Check if any row has a value matching the column name
                mask = df_regular[col].astype(str).str.strip().str.lower() != col_str
                df_regular = df_regular[mask].copy()
            df_regular = df_regular.reset_index(drop=True)

        # Ensure numeric types for plotting
        for col in df_regular.columns:
            try:
                # Convert all columns to numeric (wavelength and data columns)
                df_regular[col] = pd.to_numeric(df_regular[col], errors='coerce')
            except Exception as e:
                # If conversion fails, leave column as is
                pass


        return df_regular

    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None

@st.cache_data
def load_csv_data(file_bytes: bytes, filename: str):
    """Load and cache CSV data with spectroscopy-aware parsing"""
    return parse_spectroscopy_csv(file_bytes, filename)

# ============================================================================
# SPECTROPUS-STYLE CURVE FITTING AGENT SECTION
# ============================================================================

def render_well_plate(format_name):
    """Render a compact clickable well plate selector."""
    if format_name == "96-well (8x12)":
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        cols = list(range(1, 13))
    else:  # 35-well (5x7)
        rows = ['A', 'B', 'C', 'D', 'E']
        cols = list(range(1, 8))

    # CSS for smaller circular buttons with tighter spacing
    st.markdown("""
        <style>
        .well-plate-container .stButton > button {
            border-radius: 50%;
            width: 20px !important;
            height: 20px !important;
            padding: 0 !important;
            min-width: 20px !important;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0.5px !important;
            font-size: 8px !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.caption("Click wells to toggle (Red = Included, Grey = Excluded)")
    
    # Compact selection control buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Select All", key="select_all_wells", use_container_width=True):
            for r in rows:
                for c in cols:
                    st.session_state.selected_wells.add(f"{r}{c}")
            st.rerun()
    with c2:
        if st.button("Clear All", key="clear_all_wells", use_container_width=True):
            st.session_state.selected_wells = set()
            st.rerun()

    # Compact grid rendering with tighter spacing
    # Header row for column numbers (smaller)
    header_cols = st.columns([0.3] + [0.5] * len(cols))
    with header_cols[0]:
        st.write("")
    for i, c in enumerate(cols):
        with header_cols[i+1]:
            st.markdown(f"<small><b>{c}</b></small>", unsafe_allow_html=True)

    # Well grid with tighter spacing
    for r in rows:
        row_cols = st.columns([0.3] + [0.5] * len(cols))
        with row_cols[0]:
            st.markdown(f"<small><b>{r}</b></small>", unsafe_allow_html=True)
        for i, c in enumerate(cols):
            well_id = f"{r}{c}"
            is_selected = well_id in st.session_state.selected_wells
            with row_cols[i+1]:
                if st.button(
                    "",  # Empty button - no text inside circle
                    key=f"btn_{well_id}",
                    help=f"Well {well_id}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                ):
                    if is_selected:
                        st.session_state.selected_wells.remove(well_id)
                    else:
                        st.session_state.selected_wells.add(well_id)
                    st.rerun()

# File upload section
st.header("📁 Data Upload")

# Initialize session state for uploaded files
if "cf_data_file" not in st.session_state:
    st.session_state.cf_data_file = None
if "cf_composition_file" not in st.session_state:
    st.session_state.cf_composition_file = None

col1, col2 = st.columns(2)

with col1:
    data_file = st.file_uploader(
        "Upload Spectral Data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload the spectral data CSV or Excel file with multiple reads and wells",
        key="cf_data_uploader"
    )
    if data_file:
        st.session_state.cf_data_file = data_file
        file_type = "Excel" if data_file.name.lower().endswith(('.xlsx', '.xls')) else "CSV"
        st.success(f"✅ Data file uploaded: {data_file.name} ({file_type})")

with col2:
    composition_file = st.file_uploader(
        "Upload Composition CSV",
        type=["csv"],
        help="Upload the composition CSV file with well compositions",
        key="cf_composition_uploader"
    )
    if composition_file:
        st.session_state.cf_composition_file = composition_file
        st.success(f"✅ Composition file uploaded: {composition_file.name}")

# Use stored files if available
data_file = st.session_state.cf_data_file
composition_file = st.session_state.cf_composition_file

# Show current file status
if data_file or composition_file:
    st.subheader("📋 Current Files")
    col1, col2 = st.columns(2)
    with col1:
        if data_file:
            st.info(f"📊 Data: {data_file.name}")
        else:
            st.warning("❌ No data file uploaded")
    with col2:
        if composition_file:
            st.info(f"🧪 Composition: {composition_file.name}")
        else:
            st.warning("❌ No composition file uploaded")

# Analysis parameters section
st.header("⚙️ Analysis Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    max_peaks = st.slider(
        "Maximum Peaks",
        min_value=1,
        max_value=8,
        value=4,
        help="Maximum number of peaks to fit per spectrum"
    )

    r2_target = st.slider(
        "R² Target",
        min_value=0.5,
        max_value=0.99,
        value=0.90,
        step=0.01,
        help="Target R² value for acceptable fit quality"
    )

with col2:
    max_attempts = st.slider(
        "Maximum Attempts",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum fitting attempts per well (includes model switching)"
    )

    # Read type selection
    st.markdown("**Read Type Selection**")
    read_type = st.radio(
        "What type of reads do you want to analyze?",
        ["PL Reads (EM Spectrum)", "Absorbance Reads"],
        index=0,
        help="Choose the type of spectroscopic reads to analyze. PL reads are emission spectra, absorbance reads are absorption spectra."
    )

    # Read selection with multiple options
    st.markdown("**Read Selection**")
    read_type_desc = "EM Spectrum" if "PL" in read_type else "Absorbance"
    st.caption(f"💡 Note: Only {read_type_desc} reads will be analyzed. Other read types are automatically excluded.")

    read_mode = st.radio(
        "Select reads to analyze:",
        ["Auto (first read)", "All reads", "Range", "Odd/Even", "Specific reads", "Custom (advanced)"],
        index=0,
        help=f"Choose how to select which {read_type_desc.lower()} reads to analyze."
    )
    
    reads_selection = "auto"  # Default
    
    if read_mode == "Auto (first read)":
        reads_selection = "auto"
        st.info("Will analyze the first EM spectrum read found")
    elif read_mode == "All reads":
        reads_selection = "all"
        st.info("Will analyze all EM spectrum reads")
    elif read_mode == "Range":
        col_range_start, col_range_end = st.columns(2)
        with col_range_start:
            range_start = st.number_input("Start read", min_value=1, value=1, step=1)
        with col_range_end:
            range_end = st.number_input("End read", min_value=1, value=10, step=1)
        reads_selection = f"{range_start}-{range_end}"
        st.info(f"Will analyze reads {range_start} through {range_end}")
    elif read_mode == "Odd/Even":
        odd_even = st.radio("Select:", ["Odd reads (1,3,5...)", "Even reads (2,4,6...)"], index=0)
        reads_selection = "odd" if "Odd" in odd_even else "even"
        st.info(f"Will analyze {'odd' if reads_selection == 'odd' else 'even'} numbered reads")
    elif read_mode == "Specific reads":
        specific_reads = st.text_input(
            "Enter read numbers (comma-separated)",
            value="1,3,5",
            help="e.g., '1,3,5' or '1,2,3,5,8'"
        )
        reads_selection = specific_reads.strip() if specific_reads.strip() else "auto"
        st.info(f"Will analyze reads: {reads_selection}")
    else:  # Custom (advanced)
        reads_selection = st.text_input(
            "Custom read specification",
            value="auto",
            help="'auto', 'all', 'odd', 'even', '1-10', '1,3,5', or combinations"
        )
        st.info("Advanced: Use any valid format")

with col3:
    st.markdown("**Well Selection Mode**")
    plate_format = st.selectbox(
        "Select Plate Format",
        ["96-well (8x12)", "35-well (5x7)"],
        index=0 if st.session_state.plate_format == "96-well (8x12)" else 1,
        key="plate_format_selector"
    )
    if plate_format != st.session_state.plate_format:
        st.session_state.plate_format = plate_format
        # Optional: Clear selection when changing format
        # st.session_state.selected_wells = set()
        st.rerun()
    
    # Compact well plate selector right below format selection
    render_well_plate(st.session_state.plate_format)

    save_plots = st.checkbox("Save Fitting Plots", value=True, help="Generate and save fitting visualization plots")
    
    # API Rate Limiting
    st.markdown("**⚡ API Rate Limiting**")
    api_delay_seconds = st.number_input(
        "Delay Between API Calls (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Delay between Gemini API calls to prevent rate limit (RPD) errors. Default: 0.5s. Increase if you hit rate limits."
    )
    st.caption(f"💡 Current delay: {api_delay_seconds}s ({api_delay_seconds*1000:.0f}ms) between calls")

# Wavelength filtering section (optional)
st.subheader("🔬 Wavelength Filtering (Optional)")
st.caption("💡 Leave empty to use full data range. These filters apply before peak detection.")

wavelength_col1, wavelength_col2, wavelength_col3 = st.columns(3)

with wavelength_col1:
    use_wavelength_filter = st.checkbox(
        "Filter Wavelength Range",
        value=False,
        help="Enable to filter data to a specific wavelength range"
    )
    
    start_wavelength = st.number_input(
        "Start Wavelength (nm)",
        min_value=100,
        max_value=2000,
        value=400,
        step=1,
        disabled=not use_wavelength_filter,
        help="Minimum wavelength to include in analysis"
    )

with wavelength_col2:
    end_wavelength = st.number_input(
        "End Wavelength (nm)",
        min_value=100,
        max_value=2000,
        value=900,
        step=1,
        disabled=not use_wavelength_filter,
        help="Maximum wavelength to include in analysis"
    )

with wavelength_col3:
    use_step_size = st.checkbox(
        "Filter by Step Size",
        value=False,
        help="Enable to downsample data by step size"
    )
    
    wavelength_step_size = st.number_input(
        "Wavelength Step Size",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
        disabled=not use_step_size,
        help="Step size for wavelength sampling (e.g., 2 = every 2nm, 5 = every 5nm)"
    )

# Set to None if not enabled
if not use_wavelength_filter:
    start_wavelength = None
    end_wavelength = None
if not use_step_size:
    wavelength_step_size = None

# Analysis section
st.header("🚀 Run Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    run_button = st.button("Start Curve Fitting Analysis", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Clear Files", use_container_width=True):
        st.session_state.cf_data_file = None
        st.session_state.cf_composition_file = None
        st.rerun()

if run_button:
    if not st.session_state.cf_data_file:
        st.error("❌ Please upload a spectral data file (CSV or Excel) before running analysis.")
        st.stop()

    if not st.session_state.get('api_key'):
        st.error("❌ Please set your Google Gemini API key in Settings before running analysis.")
        st.stop()

    # Get files from session state
    data_file = st.session_state.cf_data_file
    composition_file = st.session_state.cf_composition_file
    
    # Check if data file is Excel
    is_excel = data_file.name.lower().endswith(('.xlsx', '.xls'))

    # Save uploaded files temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, "data.csv")
        comp_path = os.path.join(temp_dir, "composition.csv")

        # Convert Excel to CSV format if needed
        if is_excel:
            try:
                st.info("📊 Converting Excel file to CSV format...")
                csv_bytes = convert_excel_to_spectroscopy_csv(
                    data_file.getvalue(), 
                    data_file.name,
                    start_row=4,  # Data starts at row 4 (0-indexed)
                    read_start_col=1  # Reads start at column 1 (column 0 is wavelength)
                )
                # Write as text file with UTF-8 encoding
                with open(data_path, "w", encoding='utf-8') as f:
                    f.write(csv_bytes.decode('utf-8'))
                st.success("✅ Excel file converted successfully!")
            except Exception as e:
                st.error(f"❌ Error converting Excel file: {str(e)}")
                import traceback
                st.exception(e)
                st.stop()
        else:
            # Save CSV file as-is (handle encoding)
            try:
                # Try to read and write as UTF-8
                csv_content = data_file.getvalue()
                if isinstance(csv_content, bytes):
                    # Try to decode as UTF-8, fallback to latin-1 if needed
                    try:
                        csv_content = csv_content.decode('utf-8')
                    except UnicodeDecodeError:
                        csv_content = csv_content.decode('latin-1')
                with open(data_path, "w", encoding='utf-8') as f:
                    f.write(csv_content)
            except Exception as e:
                # Fallback to binary write
                with open(data_path, "wb") as f:
                    f.write(data_file.getvalue() if isinstance(data_file.getvalue(), bytes) else data_file.getvalue().encode('utf-8'))

        # Handle composition file
        if composition_file:
            try:
                comp_content = composition_file.getvalue()
                if isinstance(comp_content, bytes):
                    try:
                        comp_content = comp_content.decode('utf-8')
                    except UnicodeDecodeError:
                        comp_content = comp_content.decode('latin-1')
                with open(comp_path, "w", encoding='utf-8') as f:
                    f.write(comp_content)
            except Exception as e:
                # Fallback to binary write
                with open(comp_path, "wb") as f:
                    f.write(composition_file.getvalue())
        else:
            # Create dummy composition CSV for Excel files
            if is_excel:
                st.warning("⚠️ No composition file provided. Creating dummy composition file for Excel data.")
                try:
                    # Read the converted CSV to get well names
                    with open(data_path, 'r', encoding='utf-8') as f:
                        csv_content = f.read()
                    
                    # Find all well names (R1, R2, etc.) from the CSV
                    well_pattern = re.compile(r'R\d+')
                    well_names = list(set(well_pattern.findall(csv_content)))
                    well_names.sort(key=lambda x: int(x[1:]))  # Sort by number
                    
                    if not well_names:
                        # Fallback: create wells based on number of reads detected
                        # Count "Read N:" headers
                        read_pattern = re.compile(r'Read\s+(\d+):')
                        read_numbers = [int(m.group(1)) for m in read_pattern.findall(csv_content)]
                        well_names = [f"R{n}" for n in sorted(read_numbers)]
                    
                    dummy_csv_content = create_dummy_composition_csv(well_names)
                    with open(comp_path, "w", encoding='utf-8') as f:
                        f.write(dummy_csv_content)
                    st.info(f"✅ Created dummy composition file with {len(well_names)} acquisitions/reads: {well_names[:10]}{'...' if len(well_names) > 10 else ''}")
                except Exception as e:
                    st.error(f"❌ Error creating composition file: {str(e)}")
                    import traceback
                    st.exception(e)
                    st.stop()
            else:
                st.error("❌ Please upload a composition CSV file before running analysis.")
                st.stop()

        # Determine wells to analyze based on visual selection
        # If no wells selected, default to all wells (passing None to the agent)
        wells_to_analyze = list(st.session_state.selected_wells) if st.session_state.selected_wells else None
        
        # Determine read type for filtering
        read_type_filter = "em_spectrum" if "PL" in read_type else "absorbance"

        # Quick validation: Analyze read types to ensure selection is valid
        # Use the already converted CSV file for validation
        # Read as text first to avoid pandas parsing issues with empty lines
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                csv_lines = f.readlines()
            
            # Find read headers by scanning lines
            read_pattern = re.compile(r"^Read\s+(\d+):(.*)$", re.I)
            read_analysis = {"em_spectrum": [], "absorbance": [], "unknown": []}
            
            for line in csv_lines:
                line = line.strip()
                m = read_pattern.match(line)
                if m:
                    read_num = int(m.group(1))
                    read_desc = m.group(2).strip().lower()

                    if "em spectrum" in read_desc or "emission" in read_desc or "fluorescence" in read_desc:
                        read_analysis["em_spectrum"].append(read_num)
                    elif "absorbance" in read_desc or "absorption" in read_desc:
                        read_analysis["absorbance"].append(read_num)
                    else:
                        read_analysis["unknown"].append(read_num)
        except Exception as e:
            # Fallback: try reading with pandas, but handle errors gracefully
            try:
                debug_data = pd.read_csv(data_path, header=None, encoding='utf-8', on_bad_lines='skip', engine='python')
                first_col = debug_data.iloc[:, 0].astype(str)
                read_pattern = re.compile(r"^Read\s+(\d+):(.*)$", re.I)

                read_analysis = {"em_spectrum": [], "absorbance": [], "unknown": []}
                for idx, val in first_col.items():
                    m = read_pattern.match(str(val))
                    if m:
                        read_num = int(m.group(1))
                        read_desc = m.group(2).strip().lower()

                        if "em spectrum" in read_desc or "emission" in read_desc or "fluorescence" in read_desc:
                            read_analysis["em_spectrum"].append(read_num)
                        elif "absorbance" in read_desc or "absorption" in read_desc:
                            read_analysis["absorbance"].append(read_num)
                        else:
                            read_analysis["unknown"].append(read_num)
            except Exception as e2:
                st.warning(f"Could not parse CSV for validation: {e2}. Proceeding with analysis...")
                read_analysis = {"em_spectrum": [1], "absorbance": [], "unknown": []}  # Default assumption

        # Check if the requested read type exists
        if read_type_filter == "absorbance" and not read_analysis["absorbance"]:
            st.error("❌ No absorbance reads found in your data file! Only EM Spectrum reads are available.")
            st.stop()
        elif read_type_filter == "em_spectrum" and not read_analysis["em_spectrum"]:
            st.error("❌ No EM Spectrum reads found in your data file! Check your data format.")
            st.stop()

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Starting curve fitting analysis...")

            # Run the analysis
            result = agent.run_curve_fitting(
                data_csv_path=data_path,
                composition_csv_path=comp_path,
                wells_to_analyze=wells_to_analyze,
                wells_to_ignore=None,
                reads_to_analyze=reads_selection,
                read_type=read_type_filter,
                max_peaks=max_peaks,
                r2_target=r2_target,
                max_attempts=max_attempts,
                save_plots=save_plots,
                start_wavelength=start_wavelength,
                end_wavelength=end_wavelength,
                wavelength_step_size=wavelength_step_size,
                api_delay_seconds=api_delay_seconds
            )

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            if result["success"]:
                st.success(f"✅ Analysis completed successfully! Processed {result['summary']['total_wells']} wells.")

                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Wells Analyzed", result['summary']['total_wells'])
                with col2:
                    st.metric("Successful Fits", result['summary']['successful_fits'])
                with col3:
                    success_rate = result['summary']['successful_fits'] / result['summary']['total_wells'] * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                # Show results for each well
                st.subheader("📊 Analysis Results")

                for well_result in result["results"]:
                    well_name = well_result['well_name']
                    fit_result = well_result['fit_result']
                    read_num = well_result.get('read', '')
                    # Include read number in label to make it unique
                    read_label = f" (Read {read_num})" if read_num else ""
                    
                    with st.expander(f"Well {well_name}{read_label} - R²: {fit_result.stats.r2:.4f}", expanded=False):
                        # Quality assessment
                        quality = well_result['quality_assessment']
                        st.write("**Fit Quality:**")
                        for key, value in quality.items():
                            if key == 'r2':
                                st.write(f"• R²: {value}")
                            elif key == 'chi2':
                                st.write(f"• χ²: {value}")

                        # Peak information
                        st.write(f"**Peaks Found:** {len(fit_result.peaks)}")
                        if fit_result.peaks:
                            peak_data = []
                            for i, p in enumerate(fit_result.peaks):
                                center_val = f"{p.center:.1f}" if hasattr(p, 'center') and p.center is not None else 'N/A'
                                height_val = f"{p.height:.2f}" if hasattr(p, 'height') and p.height is not None else 'N/A'
                                fwhm_val = f"{p.fwhm:.1f}" if hasattr(p, 'fwhm') and p.fwhm is not None else 'N/A'
                                
                                peak_data.append({
                                    'Peak': i+1,
                                    'Center (nm)': center_val,
                                    'Height': height_val,
                                    'FWHM (nm)': fwhm_val
                                })
                            
                            peaks_df = pd.DataFrame(peak_data)
                            st.dataframe(peaks_df, use_container_width=True)

                        # Show fitting plot if available (matplotlib with download)
                        if 'files' in well_result and 'fitting_plot' in well_result['files']:
                            plot_path = well_result['files']['fitting_plot']
                            if os.path.exists(plot_path):
                                # Load and display the matplotlib plot
                                import matplotlib.image as mpimg
                                img = mpimg.imread(plot_path)
                                st.image(img, caption=f"Fitting plot for well {well_name}", width="stretch")

                                # Add download button for the plot
                                with open(plot_path, "rb") as file:
                                    read_num = well_result.get('read', '')
                                    unique_key = f"download_plot_{well_name}_read{read_num}"
                                    btn = st.download_button(
                                        label=f"📥 Download Plot (Well {well_name})",
                                        data=file,
                                        file_name=f"fitting_plot_{well_name}_read{read_num}.png",
                                        mime="image/png",
                                        use_container_width=True,
                                        key=unique_key
                                    )

                # Download section
                st.subheader("📥 Download Results")

                if 'files' in result:
                    col1, col2 = st.columns(2)
                    
                    # Get base name from uploaded file for download filenames
                    data_file_name = data_file.name if data_file else "unknown_file"
                    base_name = os.path.splitext(data_file_name)[0]
                    # Remove any problematic characters for filenames
                    base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                    base_name = base_name.strip()

                    # JSON results
                    if 'json_results' in result['files']:
                        json_path = result['files']['json_results']
                        if os.path.exists(json_path):
                            with open(json_path, 'r') as f:
                                json_data = f.read()
                            col1.download_button(
                                "📄 Download JSON Results",
                                json_data,
                                f"{base_name}_peak_fit_results.json",
                                "application/json",
                                use_container_width=True,
                                key=f"download_json_{int(time.time())}"
                            )

                    # CSV export
                    if 'csv_export' in result['files']:
                        csv_path = result['files']['csv_export']
                        if os.path.exists(csv_path):
                            with open(csv_path, 'r') as f:
                                csv_data = f.read()
                            col2.download_button(
                                "📊 Download CSV Export",
                                csv_data,
                                f"{base_name}_peak_fit_export.csv",
                                "text/csv",
                                use_container_width=True,
                                key=f"download_csv_{int(time.time())}"
                            )

            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"❌ Analysis failed with error: {str(e)}")
            st.error("Please check your CSV files are in the correct format and try again.")

# Instructions section
st.header("📖 Instructions")

with st.expander("How to Use the Curve Fitting Agent", expanded=False):
    st.markdown("""
    ### Data Format Requirements

    **Spectral Data (CSV or Excel):**
    - **CSV Format**: Must contain "Read N:EM Spectrum" headers for each read block
      - Well columns should be named A1, B2, C3, etc. (A-H, 1-12)
      - Wavelength/intensity data in rows for each well
    - **Excel Format**: 
      - Wavelength data in column 0 (first column), starting at row 4 (0-indexed)
      - Read data in columns 5+ (starting from column index 5)
      - Each column from column 5 onward represents a different read
      - The agent will automatically convert Excel files to the required CSV format

    **Composition CSV (Optional for Excel files):**
    - First column should be material names (index)
    - Subsequent columns should be well names matching the data CSV
    - Values represent material concentrations/compositions
    - **Note**: For Excel files, if no composition file is provided, a dummy composition file will be created automatically

    ### Analysis Workflow

    1. **Upload Files**: Upload both spectral data and composition CSVs
    2. **Configure Parameters**: Adjust peak detection and fitting settings
    3. **Run Analysis**: Click "Start Curve Fitting Analysis"
    4. **Review Results**: Check fit quality, peak parameters, and visualizations
    5. **Download**: Export results as JSON or CSV

    ### AI-Powered Features

    - **LLM Peak Detection**: AI analyzes spectra to identify emission peaks
    - **Model Selection**: AI chooses optimal fitting models (Gaussian, Voigt, etc.)
    - **Iterative Refinement**: Automatically improves fits based on residuals
    - **Quality Assessment**: Visual and statistical evaluation of fit quality
    - **Batch Processing**: Analyze multiple wells and reads efficiently
    
    ### Rate Limiting
    
    - **API Delay**: A default 0.5 second delay is added between Gemini API calls to prevent rate limit (RPD) errors
    - **Adjusting Delay**: If you encounter rate limit errors, increase the "Delay Between API Calls" parameter
    - **Environment Variables**: You can also set `GEMINI_MIN_DELAY_MS` (milliseconds) or `GEMINI_MIN_DELAY_S` (seconds) environment variables
    - **Recommended Settings**: 
      - Small batches (< 10 wells): 0.5s delay
      - Medium batches (10-50 wells): 1.0s delay  
      - Large batches (> 50 wells): 1.5-2.0s delay
    """)

# Sample data section
st.header("📋 Sample Data")

if st.button("Load Sample Data for Testing"):
    # Create sample data for testing
    sample_data = """Read 1:EM Spectrum
Wavelength,A1,A2,A3
400,0.1,0.05,0.08
410,0.2,0.1,0.15
420,0.8,0.4,0.6
430,2.1,1.0,1.5
440,3.2,1.6,2.4
450,4.1,2.0,3.0
460,3.8,1.9,2.8
470,2.9,1.4,2.1
480,1.8,0.9,1.3
490,0.9,0.4,0.7
500,0.3,0.15,0.25
"""

    sample_composition = """Material,A1,A2,A3
CsPbBr3,0.3,0.0,0.0
FAPbBr3,0.7,1.0,0.5
MAPbI3,0.0,0.0,0.5
"""

    st.download_button(
        "📥 Download Sample Spectral Data",
        sample_data,
        "sample_spectral_data.csv",
        "text/csv",
        use_container_width=True,
        key="download_sample_spectral"
    )

    st.download_button(
        "📥 Download Sample Composition Data",
        sample_composition,
        "sample_composition.csv",
        "text/csv",
        use_container_width=True,
        key="download_sample_composition"
    )

st.markdown("---")
st.caption("Powered by Spectropus LLM-assisted curve fitting technology")
