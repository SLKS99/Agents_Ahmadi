import os
import tempfile
import json
import re
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
from tools.memory import MemoryManager
from agents.curve_fitting_agent import CurveFittingAgent
import matplotlib.pyplot as plt

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

# Check for auto-triggered file from watcher
trigger_info_file = Path(__file__).parent.parent / "watcher_trigger_info.json"
auto_triggered = False
triggered_file = None

# Track if we've already processed this trigger to avoid re-running
if "last_processed_trigger_time" not in st.session_state:
    st.session_state.last_processed_trigger_time = 0

if trigger_info_file.exists():
    try:
        import time
        with open(trigger_info_file, 'r') as f:
            trigger_info = json.load(f)
        
        trigger_time = trigger_info.get("trigger_time", 0)
        time_since_trigger = time.time() - trigger_time
        
        # Process if triggered within last 5 minutes and not already processed
        if time_since_trigger < 300 and trigger_time != st.session_state.last_processed_trigger_time:
            auto_triggered = True
            triggered_file = trigger_info.get("triggered_file", "Unknown")
            timestamp = trigger_info.get("timestamp", "")
            inferred_params = trigger_info.get("parameters", {})
            
            # Mark as processed
            st.session_state.last_processed_trigger_time = trigger_time
            st.session_state.watcher_auto_triggered_file = triggered_file
            st.session_state.watcher_auto_trigger_time = trigger_time
            
            # Show banner
            st.success(
                f"🔄 **Auto-triggered by Watcher!** File detected: `{Path(triggered_file).name}`\n"
                f"📅 {timestamp}",
                icon="🤖"
            )
            
            # Check if results already exist (from previous headless run)
            results_dir = Path(__file__).parent.parent / "results"
            results_exist = False
            if results_dir.exists():
                json_files = list(results_dir.glob("*_peak_fit_results.json"))
                if json_files:
                    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    latest_result = json_files[0]
                    result_age = time.time() - latest_result.stat().st_mtime
                    if result_age < 300:  # Results from last 5 minutes
                        results_exist = True
                        st.info(f"✅ **Results found from previous run:** `{latest_result.name}`")
                        try:
                            with open(latest_result, 'r') as f:
                                result_data = json.load(f)
                            st.session_state.auto_triggered_results = result_data
                            st.session_state.auto_triggered_results_file = str(latest_result)
                        except Exception:
                            pass
            
            # If no results exist, automatically run curve fitting in Streamlit context
            if not results_exist:
                st.info("🚀 **Running curve fitting with UI support...**")
                
                # Prepare file path for curve fitting
                data_path = Path(triggered_file)
                if data_path.exists():
                    # Extract parameters from trigger info
                    max_peaks = inferred_params.get("max_peaks", 4)
                    r2_target = inferred_params.get("r2_target", 0.90)
                    max_attempts = inferred_params.get("max_attempts", 3)
                    reads_to_analyze = inferred_params.get("read_selection", "auto")
                    read_type = inferred_params.get("read_type", "em_spectrum")
                    wells_to_analyze = inferred_params.get("wells_to_analyze", None)
                    api_delay = inferred_params.get("api_delay_seconds", 0.5)
                    
                    # Look for composition file, fallback to default
                    # Search for common composition file names in the same directory as the data file
                    comp_file = None
                    data_dir = data_path.parent
                    # List of possible composition file names (searched in order)
                    composition_file_names = [
                        "composition.csv",
                        "compositions.csv", 
                        "comp.csv",
                        "sample_info.csv",
                        "sample_composition.csv",
                        "well_composition.csv",
                        "composition_data.csv"
                    ]
                    for comp_name in composition_file_names:
                        comp_path = data_dir / comp_name
                        if comp_path.exists():
                            comp_file = str(comp_path)
                            st.info(f"Found composition file in data directory: {comp_name}")
                            break
                    
                    # If no composition file found, use default
                    if not comp_file:
                        default_comp_path = Path(__file__).parent.parent / "data" / "2D-3D.csv"
                        if default_comp_path.exists():
                            comp_file = str(default_comp_path)
                            st.info(f"Using default composition file: {default_comp_path.name}")
                    
                    # Store parameters in session state for the run_curve_fitting call below
                    st.session_state.auto_run_curve_fitting = True
                    st.session_state.auto_run_data_file = str(data_path)
                    st.session_state.auto_run_comp_file = comp_file or ""
                    st.session_state.auto_run_params = {
                        "max_peaks": max_peaks,
                        "r2_target": r2_target,
                        "max_attempts": max_attempts,
                        "reads_to_analyze": reads_to_analyze,
                        "read_type": read_type,
                        "wells_to_analyze": wells_to_analyze,
                        "api_delay_seconds": api_delay,
                    }
                else:
                    st.error(f"File not found: {triggered_file}")
        elif time_since_trigger >= 300:
            # Clean up old trigger info
            trigger_info_file.unlink(missing_ok=True)
    except Exception as e:
        # Silently handle errors reading trigger info
        pass

st.title("Curve Fitting")
st.markdown("Upload CSV files, adjust parameters, and generate fitted curves with interactive visualizations.")

# Check for automatic ML execution setting
if "auto_ml_after_curve_fitting" not in st.session_state:
    st.session_state.auto_ml_after_curve_fitting = False
if "auto_route_to_analysis" not in st.session_state:
    st.session_state.auto_route_to_analysis = False

if st.button("Clear Cache and Restart Program"):
    st.cache_data.clear()
    st.rerun()

# Workflow automation settings
with st.expander("⚙️ Workflow Automation Settings", expanded=False):
    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        auto_ml_enabled = st.checkbox(
            "Auto-run ML model after curve fitting",
            value=st.session_state.auto_ml_after_curve_fitting,
            help="Automatically run the selected ML model when curve fitting completes",
            key="auto_ml_checkbox"
        )
        st.session_state.auto_ml_after_curve_fitting = auto_ml_enabled
    
    with col_auto2:
        auto_route_analysis = st.checkbox(
            "Auto-route to Analysis Agent after ML",
            value=st.session_state.auto_route_to_analysis,
            help="Automatically navigate to Analysis Agent after ML model completes",
            key="auto_route_checkbox"
        )
        st.session_state.auto_route_to_analysis = auto_route_analysis
    
    if auto_ml_enabled:
        selected_model = st.session_state.get("optimization_model_choice", "No model selected")
        st.info(f"📊 Selected ML Model: **{selected_model}**")
        if selected_model == "No model selected":
            st.warning("⚠️ Please select an ML model on the ML Models page first.")

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

def convert_excel_to_spectroscopy_csv(file_bytes: bytes, filename: str, start_row: int = 2, read_start_col: int = 1, plate_format: str = None) -> bytes:
    """
    Convert Excel file with wavelength in column 0 and data in columns 1+ 
    to the CSV format expected by the fitting agent.
    
    Can handle two formats:
    1. Well plate format: Columns represent wells (A1-E7 for 35-well, A1-H12 for 96-well)
    2. Acquisition format: Each column is a separate read/acquisition (R1, R2, etc.)
    
    Args:
        file_bytes: Excel file bytes
        filename: Original filename
        start_row: Row where data starts (0-indexed, default 4)
        read_start_col: Column index where data starts (default 1, column 0 is wavelength)
        plate_format: Plate format string ("35-well (5x7)" or "96-well (8x12)") to determine well naming
    
    Returns:
        CSV file bytes in the format expected by the fitting agent
    """
    try:
        # Read Excel file
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=0, header=None)
        
        # Check if this is a well plate format (35-well or 96-well) that needs Read header parsing
        is_well_plate_format = plate_format and ("35-well" in plate_format or "96-well" in plate_format)
        
        data_start_row = None
        wavelength_col = 1  # Default to column 1
        
        # If this is a well-plate export, we want to preserve multiple "Read N" blocks
        # (Cytation exports frequently stack multiple reads vertically).
        read_block_starts: Dict[int, int] = {}

        if is_well_plate_format:
            # For 35-well and 96-well plates: Parse "Read N:EM Spectrum" headers
            read_header_pattern = re.compile(r"^Read\s+(\d+)[:\s]+(?:EM\s+)?Spectrum", re.I)
            
            # Check column 0 and column 1 for Read headers
            for col_idx in [0, 1]:
                for i in range(min(500, len(df))):  # Scan deeper; Cytation exports can be long
                    try:
                        val = str(df.iloc[i, col_idx]).strip()
                        m = read_header_pattern.match(val)
                        if m:
                            read_num = int(m.group(1))
                            read_block_starts.setdefault(read_num, i)
                    except Exception:
                        continue

            read_block_starts = dict(sorted(read_block_starts.items()))
            
            # If we found Read headers, we'll extract each Read block separately later.
            # Otherwise, we'll fall back to numeric detection and treat as a single block.
        
        # If data_start_row not found yet (no Read headers or Insitu format), use numeric detection
        if data_start_row is None:
            # Strategy 1: Check column 1 (wavelength column) for wavelength-like values (>= 50nm)
            for i in range(min(100, len(df))):  # Check first 100 rows
                try:
                    val = df.iloc[i, 1]  # Check column 1 (wavelength column)
                    if pd.notna(val):
                        numeric_val = pd.to_numeric(val, errors='coerce')
                        if pd.notna(numeric_val) and numeric_val >= 50 and numeric_val <= 2000:
                            data_start_row = i
                            wavelength_col = 1
                            break
                except:
                    continue
            
            # Strategy 2: Check column 0 as fallback
            if data_start_row is None:
                for i in range(min(100, len(df))):
                    try:
                        val = df.iloc[i, 0]
                        if pd.notna(val):
                            numeric_val = pd.to_numeric(val, errors='coerce')
                            if pd.notna(numeric_val) and numeric_val >= 50 and numeric_val <= 2000:
                                data_start_row = i
                                wavelength_col = 0
                                break
                    except:
                        continue
            
            # Strategy 3: Use default start_row if still not found
            if data_start_row is None:
                data_start_row = start_row if start_row < len(df) else 2
                wavelength_col = 1
        
        # -------- Extract data into blocks --------
        blocks: List[tuple[int, pd.Series, pd.DataFrame]] = []

        def _find_block_data_start(read_header_row: int) -> tuple[int, int]:
            """Return (data_start_row, wavelength_col) for a given read header row."""
            for rr in range(read_header_row + 1, min(read_header_row + 25, len(df))):
                for cc in [1, 2]:
                    try:
                        numeric_val = pd.to_numeric(df.iloc[rr, cc], errors='coerce')
                        if pd.notna(numeric_val) and 50 <= float(numeric_val) <= 2000:
                            return rr, cc
                    except Exception:
                        continue
            return read_header_row + 2, 1

        def _contiguous_wavelength_end(start_r: int, wl_c: int, end_r: int) -> int:
            """Return the first row AFTER the contiguous wavelength run."""
            r = start_r
            while r < end_r:
                try:
                    v = pd.to_numeric(df.iloc[r, wl_c], errors='coerce')
                    if pd.isna(v) or not (50 <= float(v) <= 2000):
                        break
                except Exception:
                    break
                r += 1
            return r

        if is_well_plate_format and read_block_starts:
            read_nums = sorted(read_block_starts.keys())
            for i_rn, read_num in enumerate(read_nums):
                header_row = read_block_starts[read_num]
                next_header = read_block_starts[read_nums[i_rn + 1]] if i_rn + 1 < len(read_nums) else len(df)
                dsr, wlc = _find_block_data_start(header_row)
                end_data = _contiguous_wavelength_end(dsr, wlc, next_header)

                wavelength = pd.to_numeric(df.iloc[dsr:end_data, wlc].copy(), errors='coerce')
                read_data_start_col = wlc + 1
                read_data = df.iloc[dsr:end_data, read_data_start_col:].copy().apply(pd.to_numeric, errors='coerce')

                valid_mask = wavelength.notna() & (wavelength >= 50) & (wavelength <= 2000)
                wavelength = wavelength[valid_mask].reset_index(drop=True)
                read_data = read_data[valid_mask].reset_index(drop=True)

                if len(wavelength) > 0 and read_data.shape[1] > 0:
                    blocks.append((read_num, wavelength, read_data))
        else:
            # Single-block fallback (insitu or no Read headers)
            wavelength = pd.to_numeric(df.iloc[data_start_row:, wavelength_col].copy(), errors='coerce')
            read_start_col_actual = wavelength_col + 1
            read_data = df.iloc[data_start_row:, read_start_col_actual:].copy().apply(pd.to_numeric, errors='coerce')

            # If wavelength column doesn't look like wavelengths, check if first data column might be wavelengths
            wavelength_valid = wavelength.notna().sum()
            wavelength_in_range = ((wavelength >= 50) & (wavelength <= 2000)).sum() if wavelength_valid > 0 else 0
            if wavelength_in_range == 0 and read_data.shape[1] > 0:
                first_data_col = read_data.iloc[:, 0]
                first_data_numeric = pd.to_numeric(first_data_col, errors='coerce')
                first_data_in_range = ((first_data_numeric >= 50) & (first_data_numeric <= 2000)).sum()
                if first_data_in_range > wavelength_valid:
                    wavelength = first_data_numeric.reset_index(drop=True)
                    read_data = read_data.iloc[:, 1:].reset_index(drop=True)

            valid_mask = wavelength.notna() & (wavelength >= 50) & (wavelength <= 2000)
            wavelength = wavelength[valid_mask].reset_index(drop=True)
            read_data = read_data[valid_mask].reset_index(drop=True)

            if len(wavelength) > 0 and read_data.shape[1] > 0:
                blocks.append((1, wavelength, read_data))

        if not blocks:
            raise ValueError("No valid wavelength/data blocks could be extracted from this Excel file.")
        
        # Use all available data rows (no artificial limit)
        # Note: User's plotting code uses 1:400, but we'll process all available data
        # If needed, users can filter by wavelength range in the UI
        
        # Determine if this is a well plate format or acquisition format
        num_columns = blocks[0][2].shape[1]
        is_well_plate = False
        well_names = []
        
        # Check if plate_format is specified and matches well plate format
        if plate_format and ("35-well" in plate_format or "96-well" in plate_format):
            is_well_plate = True
            # Generate well names based on plate format
            # If we have exactly 35 data columns, prefer 35-well naming even if UI selection is off.
            # This matches typical Cytation5 35-well exports.
            if num_columns == 35 or "35-well" in plate_format:
                # 35-well plate: A1-E7 (5 rows x 7 columns)
                rows = ['A', 'B', 'C', 'D', 'E']
                cols = list(range(1, 8))
            else:  # 96-well
                # 96-well plate: A1-H12 (8 rows x 12 columns)
                rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                cols = list(range(1, 13))
            
            # Generate well names in order
            for row in rows:
                for col in cols:
                    well_names.append(f"{row}{col}")
            
            # Limit to actual number of columns available
            well_names = well_names[:num_columns]
        elif plate_format and "insitu" in plate_format.lower():
            # Insitu Well format: Each column is a separate acquisition/read
            # Use acquisition format (separate read blocks)
            is_well_plate = False
            well_names = [f"R{i+1}" for i in range(num_columns)]
        else:
            # Default: Acquisition format: use R1, R2, etc.
            is_well_plate = False
            well_names = [f"R{i+1}" for i in range(num_columns)]
        
        # Create CSV format expected by fitting agent
        csv_lines = []
        
        if is_well_plate:
            # Well plate format (35-well or 96-well): Single read block with multiple well columns
            # Format per block: Read N:EM Spectrum header -> spacer -> column header -> data
            
            # Keep column counts consistent across all lines so pandas doesn't drop lines.
            # Total columns = Wavelength + wells
            total_cols = 1 + len(well_names)

            for read_num, wavelength, read_data in blocks:
                # Read header (pad with commas to match total_cols)
                csv_lines.append(f"Read {read_num}:EM Spectrum" + "," * (total_cols - 1))
                # Spacer line (all empty fields, but with correct column count)
                csv_lines.append("," * (total_cols - 1))
                # Column header
                csv_lines.append("Wavelength," + ",".join(well_names))
                # Data rows
                for idx in range(len(wavelength)):
                    wl_val = wavelength.iloc[idx]
                    if pd.isna(wl_val):
                        continue
                    row_data = [str(wl_val)]
                    for col_idx in range(num_columns):
                        if col_idx < read_data.shape[1]:
                            intensity_val = read_data.iloc[idx, col_idx]
                            row_data.append(str(intensity_val) if pd.notna(intensity_val) else "")
                        else:
                            row_data.append("")
                    csv_lines.append(",".join(row_data))
        else:
            # Acquisition format (Insitu Well): Each column is a separate read
            # For each read column, create a "Read N:EM Spectrum" block
            # Insitu uses the first (and only) block in `blocks`
            _, wavelength, read_data = blocks[0]
            num_columns = read_data.shape[1]
            for read_num in range(1, num_columns + 1):
                # Keep column counts consistent (Wavelength + 1 acquisition column)
                # Add read header (pad with comma)
                csv_lines.append(f"Read {read_num}:EM Spectrum,")
                
                # Empty line after header
                csv_lines.append(",")
                
                # Create data block with wavelength and this read's data
                read_col = read_data.iloc[:, read_num - 1]
                well_name = well_names[read_num - 1]
                
                # Build the data block header (NO Placeholder column)
                csv_lines.append(f"Wavelength,{well_name}")
                
                # Add data rows
                for idx in range(len(wavelength)):
                    wl_val = wavelength.iloc[idx]
                    intensity_val = read_col.iloc[idx]
                    if pd.notna(wl_val) and pd.notna(intensity_val):
                        csv_lines.append(f"{wl_val},{intensity_val}")
                
                # Add empty line between reads
                csv_lines.append(",")
        
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
    elif format_name == "35-well (5x7)":
        rows = ['A', 'B', 'C', 'D', 'E']
        cols = list(range(1, 8))
    else:  # Insitu Well (Acquisitions) - no well selector needed
        st.info("💡 Insitu Well format: Each column represents a separate acquisition/read (R1, R2, R3, etc.)")
        return  # Don't render well selector for acquisition format

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
        if st.button("Select All", key="select_all_wells", width='stretch'):
            for r in rows:
                for c in cols:
                    st.session_state.selected_wells.add(f"{r}{c}")
            st.rerun()
    with c2:
        if st.button("Clear All", key="clear_all_wells", width='stretch'):
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
                    width='stretch'
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
        "Upload Composition CSV (Optional)",
        type=["csv"],
        help="Upload the composition CSV file with well compositions. If not provided, default file (2D-3D.csv) will be used.",
        key="cf_composition_uploader"
    )
    if composition_file:
        st.session_state.cf_composition_file = composition_file
        st.success(f"✅ Composition file uploaded: {composition_file.name}")
    else:
        # Show info about default file
        default_comp_path = Path(__file__).parent.parent / "data" / "2D-3D.csv"
        if default_comp_path.exists():
            st.info(f"💡 Default composition file will be used: `{default_comp_path.name}`")
        else:
            st.warning(f"⚠️ Default composition file not found at: `{default_comp_path}`")

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
    plate_format_options = ["96-well (8x12)", "35-well (5x7)", "Insitu Well (Acquisitions)"]
    current_format = st.session_state.get('plate_format', "96-well (8x12)")
    default_index = 0
    if current_format == "35-well (5x7)":
        default_index = 1
    elif "insitu" in current_format.lower():
        default_index = 2
    
    plate_format = st.selectbox(
        "Select Plate Format",
        plate_format_options,
        index=default_index,
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
    
    skip_no_peaks = st.checkbox(
        "Skip Fitting for Noisy/Flat Spectra",
        value=False,
        help="If enabled, skip fitting when LLM detects no substantial peaks (saves time on noisy/flat spectra). Results will show 0 peaks."
    )
    
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
    run_button = st.button("Start Curve Fitting Analysis", type="primary", width='stretch')

with col2:
    if st.button("🗑️ Clear Files", width='stretch'):
        st.session_state.cf_data_file = None
        st.session_state.cf_composition_file = None
        st.rerun()

# Check if we should auto-run from watcher trigger
auto_run_triggered = st.session_state.get("auto_run_curve_fitting", False)

if run_button or auto_run_triggered:
    # If auto-running, use the triggered file
    if auto_run_triggered:
        triggered_file_path = st.session_state.auto_run_data_file
        # Create a file-like object from the path for compatibility
        class FileFromPath:
            def __init__(self, path):
                self.path = path
                self.name = Path(path).name
                self._path_obj = Path(path)
            
            def getvalue(self):
                """Read file content from disk."""
                if self._path_obj.exists():
                    # Read as bytes first
                    with open(self._path_obj, 'rb') as f:
                        return f.read()
                else:
                    raise FileNotFoundError(f"File not found: {self.path}")
        
        data_file = FileFromPath(triggered_file_path)
        st.session_state.cf_data_file = data_file
        composition_file = None
        if st.session_state.auto_run_comp_file:
            composition_file = FileFromPath(st.session_state.auto_run_comp_file)
            st.session_state.cf_composition_file = composition_file
    elif not st.session_state.cf_data_file:
        st.error("❌ Please upload a spectral data file (CSV or Excel) before running analysis.")
        st.stop()
    else:
        data_file = st.session_state.cf_data_file
        composition_file = st.session_state.cf_composition_file

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
                # Get plate format from session state if available
                plate_format = st.session_state.get('plate_format', None)
                
                # Get file content - handle both file uploader and FileFromPath
                if hasattr(data_file, 'getvalue'):
                    file_bytes = data_file.getvalue()
                elif hasattr(data_file, 'path'):
                    # It's a FileFromPath, read from disk
                    with open(data_file.path, 'rb') as f:
                        file_bytes = f.read()
                else:
                    # Try to read as path
                    with open(data_file, 'rb') as f:
                        file_bytes = f.read()
                
                csv_bytes = convert_excel_to_spectroscopy_csv(
                    file_bytes, 
                    data_file.name,
                    start_row=4,  # Data starts at row 4 (0-indexed)
                    read_start_col=1,  # Data starts at column 1 (column 0 is wavelength)
                    plate_format=plate_format  # Pass plate format to determine well naming
                )
                # Write as text file with UTF-8 encoding
                csv_content = csv_bytes.decode('utf-8')
                with open(data_path, "w", encoding='utf-8') as f:
                    f.write(csv_content)
                
                # Debug: Show first few lines of generated CSV and verify structure
                lines = csv_content.split('\n')[:15]
                st.text(f"DEBUG: First 15 lines of generated CSV:\n" + "\n".join(lines))
                
                # Debug: Check column counts
                line_col_counts = [len(line.split(',')) for line in lines if line.strip()]
                st.text(f"DEBUG: Column counts per line: {line_col_counts}")
                
                st.success("✅ Excel file converted successfully!")
            except Exception as e:
                st.error(f"❌ Error converting Excel file: {str(e)}")
                import traceback
                st.exception(e)
                st.stop()
        else:
            # Save CSV file as-is (handle encoding)
            try:
                # Get file content - handle both file uploader and FileFromPath
                if hasattr(data_file, 'getvalue'):
                    csv_content = data_file.getvalue()
                elif hasattr(data_file, 'path'):
                    # It's a FileFromPath, read from disk
                    with open(data_file.path, 'rb') as f:
                        csv_content = f.read()
                else:
                    # Try to read as path
                    with open(data_file, 'rb') as f:
                        csv_content = f.read()
                
                # Handle encoding
                if isinstance(csv_content, bytes):
                    # Try to decode as UTF-8, fallback to latin-1 if needed
                    try:
                        csv_content = csv_content.decode('utf-8')
                    except UnicodeDecodeError:
                        csv_content = csv_content.decode('latin-1')
                
                with open(data_path, "w", encoding='utf-8') as f:
                    f.write(csv_content)
            except Exception as e:
                # Fallback: copy file directly if it's a path
                if hasattr(data_file, 'path') and Path(data_file.path).exists():
                    import shutil
                    shutil.copy2(data_file.path, data_path)
                elif hasattr(data_file, 'getvalue'):
                    # It's a file uploader object - last resort: binary write
                    with open(data_path, "wb") as f:
                        file_content = data_file.getvalue()
                        f.write(file_content if isinstance(file_content, bytes) else file_content.encode('utf-8'))
                else:
                    raise Exception(f"Cannot read file: {data_file}. Error: {e}")

        # Handle composition file - use default if not provided
        default_comp_path = Path(__file__).parent.parent / "data" / "2D-3D.csv"
        
        if composition_file:
            try:
                # Get file content - handle both file uploader and FileFromPath
                if hasattr(composition_file, 'getvalue'):
                    comp_content = composition_file.getvalue()
                elif hasattr(composition_file, 'path'):
                    # It's a FileFromPath, read from disk
                    with open(composition_file.path, 'rb') as f:
                        comp_content = f.read()
                else:
                    # Try to read as path
                    with open(composition_file, 'rb') as f:
                        comp_content = f.read()
                
                if isinstance(comp_content, bytes):
                    try:
                        comp_content = comp_content.decode('utf-8')
                    except UnicodeDecodeError:
                        comp_content = comp_content.decode('latin-1')
                with open(comp_path, "w", encoding='utf-8') as f:
                    f.write(comp_content)
            except Exception as e:
                # Fallback: copy file directly if it's a path
                if hasattr(composition_file, 'path') and Path(composition_file.path).exists():
                    import shutil
                    shutil.copy2(composition_file.path, comp_path)
                else:
                    # Last resort: binary write
                    with open(comp_path, "wb") as f:
                        if hasattr(composition_file, 'getvalue'):
                            f.write(composition_file.getvalue())
                        else:
                            # Use default file if reading fails
                            if default_comp_path.exists():
                                import shutil
                                shutil.copy2(default_comp_path, comp_path)
                                st.info(f"Using default composition file: {default_comp_path.name}")
                            else:
                                st.warning(f"Could not read composition file: {e}")
        else:
            # Use default composition file
            if default_comp_path.exists():
                import shutil
                shutil.copy2(default_comp_path, comp_path)
                st.info(f"Using default composition file: {default_comp_path.name}")
            else:
                # Create dummy composition CSV for Excel files as last resort
                if is_excel:
                    st.warning("⚠️ No composition file provided and default not found. Creating dummy composition file for Excel data.")
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
                    st.error(f"❌ No composition file provided and default file not found at: {default_comp_path}")
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

        # Check if we should auto-run curve fitting from watcher trigger
        if st.session_state.get("auto_run_curve_fitting", False):
            # Use auto-run parameters
            data_path = st.session_state.auto_run_data_file
            comp_path = st.session_state.auto_run_comp_file
            auto_params = st.session_state.auto_run_params
            wells_to_analyze = auto_params.get("wells_to_analyze")
            reads_selection = auto_params.get("reads_to_analyze", "auto")
            read_type_filter = auto_params.get("read_type", "em_spectrum")
            max_peaks = auto_params.get("max_peaks", 4)
            r2_target = auto_params.get("r2_target", 0.90)
            max_attempts = auto_params.get("max_attempts", 3)
            api_delay_seconds = auto_params.get("api_delay_seconds", 0.5)
            
            # Clear the auto-run flag
            st.session_state.auto_run_curve_fitting = False
            
            # Show that we're running automatically
            st.info("🚀 **Auto-running curve fitting with detected file...**")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Starting curve fitting analysis...")

            # Run the analysis
            # Check if this is an auto-triggered run
            is_auto_trigger = st.session_state.get("auto_run_curve_fitting", False) or auto_triggered
            
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
                skip_no_peaks=skip_no_peaks,
                start_wavelength=start_wavelength,
                end_wavelength=end_wavelength,
                wavelength_step_size=wavelength_step_size,
                api_delay_seconds=api_delay_seconds,
                auto_trigger=is_auto_trigger
            )

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            if result["success"]:
                st.success(f"✅ Analysis completed successfully! Processed {result['summary']['total_wells']} wells.")
                
                # Check if ML Models step is next in workflow and marked as automatic
                workflow_auto_flags = st.session_state.get("workflow_auto_flags", {})
                manual_workflow = st.session_state.get("manual_workflow", [])
                workflow_index = st.session_state.get("workflow_index", 0)
                
                # Check if ML Models is next and should auto-execute
                ml_auto_from_workflow = (
                    workflow_index < len(manual_workflow)
                    and manual_workflow[workflow_index] == "ML Models"
                    and workflow_auto_flags.get("ML Models", False)
                )
                
                # Automatic ML model execution (if enabled via checkbox OR workflow)
                auto_ml_enabled = (
                    st.session_state.get("auto_ml_after_curve_fitting", False)
                    or ml_auto_from_workflow
                )
                selected_ml_model = st.session_state.get("optimization_model_choice")
                
                if auto_ml_enabled and selected_ml_model:
                    with st.spinner("🤖 Automatically running ML model..."):
                        try:
                            from tools.ml_automation import run_automated_ml_model
                            
                            # Get ML model configuration from session state
                            ml_config = st.session_state.get("ml_model_config", {})
                            
                            # Get file paths from result
                            json_file = result.get("files", {}).get("json_results")
                            csv_file = result.get("files", {}).get("csv_export")
                            
                            # Run ML model automatically
                            ml_result = run_automated_ml_model(
                                model_choice=selected_ml_model,
                                json_path=json_file,
                                csv_path=csv_file,
                                auto_config=ml_config,
                            )
                            
                            if ml_result.get("success"):
                                st.success(f"✅ ML Model ({selected_ml_model}) executed successfully!")
                                
                                # Display top candidates
                                if ml_result.get("top_candidates"):
                                    st.subheader("🤖 ML Model Recommendations")
                                    import pandas as pd
                                    candidates_df = pd.DataFrame(ml_result["top_candidates"])
                                    st.dataframe(candidates_df, width='stretch')
                                
                                # Save ML results to session state for Analysis Agent
                                st.session_state.gp_results = {
                                    "model_type": ml_result.get("model_type", "Unknown"),
                                    "automated": True,
                                    "top_candidates": ml_result.get("top_candidates", []),
                                    "predictions": ml_result.get("predictions", {}),
                                    "uncertainty_stats": ml_result.get("uncertainty_stats", {}),
                                }
                                st.session_state.analysis_ready = True
                                
                                # Auto-route to Analysis Agent if enabled
                                if st.session_state.get("auto_route_to_analysis", False):
                                    st.info("🔄 Routing to Analysis Agent...")
                                    st.session_state.next_agent = "analysis"
                                    st.rerun()
                            else:
                                st.warning(f"⚠️ ML Model execution failed: {ml_result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.warning(f"⚠️ Automatic ML model execution failed: {str(e)}")
                            import traceback
                            st.exception(e)

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
                            st.dataframe(peaks_df, width='stretch')

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
                                        width='stretch',
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
                            # Use stable key based on file path to prevent rerun
                            json_key = f"download_json_{hash(json_path)}"
                            col1.download_button(
                                "📄 Download JSON Results",
                                json_data,
                                f"{base_name}_peak_fit_results.json",
                                "application/json",
                                width='stretch',
                                key=json_key
                            )

                    # CSV export
                    if 'csv_export' in result['files']:
                        csv_path = result['files']['csv_export']
                        if os.path.exists(csv_path):
                            with open(csv_path, 'r') as f:
                                csv_data = f.read()
                            # Use stable key based on file path to prevent rerun
                            csv_key = f"download_csv_{hash(csv_path)}"
                            col2.download_button(
                                "📊 Download CSV Export",
                                csv_data,
                                f"{base_name}_peak_fit_export.csv",
                                "text/csv",
                                width='stretch',
                                key=csv_key
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
        width='stretch',
        key="download_sample_spectral"
    )

    st.download_button(
        "📥 Download Sample Composition Data",
        sample_composition,
        "sample_composition.csv",
        "text/csv",
        width='stretch',
        key="download_sample_composition"
    )

st.markdown("---")
st.caption("Powered by Spectropus LLM-assisted curve fitting technology")
