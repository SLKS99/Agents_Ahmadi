from agents.base import BaseAgent
from tools.fitting_agent import (
    LLMClient, CurveFittingConfig, build_agent_config,
    curate_dataset, run_complete_analysis, get_xy_for_well,
    save_all_wells_results, export_peak_data_to_csv,
    infer_wells_from_file_metadata,
)
import streamlit as st
from tools.memory import MemoryManager
import os
import tempfile
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import base64

# Lazy import for requests
try:
    import requests
except ImportError:
    requests = None


def _sort_wells(wells: List[str]) -> List[str]:
    """Sort wells in standard plate order: A1, A2, ..., A12, B1, B2, ..., H12
    Also handles R1, R2, etc. format for acquisitions"""
    def well_key(well: str) -> tuple:
        """Extract (row_letter, column_number) for sorting"""
        # Handle A1-H12 format
        match = re.match(r'^([A-H])(\d+)$', well.upper())
        if match:
            row_letter = match.group(1)
            col_num = int(match.group(2))
            return (0, row_letter, col_num)  # 0 = well plate format
        
        # Handle R1, R2, etc. format
        r_match = re.match(r'^R(\d+)$', well.upper())
        if r_match:
            r_num = int(r_match.group(1))
            return (1, 'R', r_num)  # 1 = R format
        
        return (2, well, 0)  # Fallback for other formats
    
    return sorted(wells, key=well_key)


def _display_fitting_plot(result: Dict[str, Any], container, well_name: str):
    """Display fitting plot in real-time during analysis."""
    try:
        # Try to load from saved plot file first
        if 'files' in result and 'fitting_plot' in result['files']:
            plot_path = result['files']['fitting_plot']
            if os.path.exists(plot_path):
                with container:
                    read_info = f" Read {result.get('read', '')}" if result.get('read') else ""
                    st.subheader(f"📊 Well {well_name}{read_info} - R²: {result['fit_result'].stats.r2:.4f}")
                    img = mpimg.imread(plot_path)
                    st.image(img, caption=f"Fitting plot for well {well_name}{read_info}", width="stretch")
                    return
        
        # If no saved plot, create one on-the-fly from data
        if 'data' in result and 'fit_result' in result:
            x = result['data']['x']
            y = result['data']['y']
            fit_result = result['fit_result']
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Top plot: data and fit
            ax1.plot(x, y, 'b-', label='Data', linewidth=1.5, alpha=0.7)
            if fit_result.lmfit_result is not None:
                ax1.plot(x, fit_result.lmfit_result.best_fit, 'r-', label='Fit', linewidth=2)
            
            # Plot individual peaks if available
            if fit_result.peaks:
                for i, peak in enumerate(fit_result.peaks):
                    if hasattr(peak, 'center') and fit_result.lmfit_result is not None:
                        # Create peak curve
                        peak_curve = _create_peak_curve(x, peak, fit_result.lmfit_result)
                        if peak_curve is not None:
                            ax1.plot(x, peak_curve, '--', alpha=0.6, label=f'Peak {i+1}')
            
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Intensity')
            ax1.set_title(f'Peak Fitting - Well {well_name} (R² = {fit_result.stats.r2:.4f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: residuals
            if fit_result.lmfit_result is not None:
                residuals = y - fit_result.lmfit_result.best_fit
                ax2.plot(x, residuals, 'g-', linewidth=1)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Display in Streamlit
            with container:
                read_info = f" Read {result.get('read', '')}" if result.get('read') else ""
                st.subheader(f"📊 Well {well_name}{read_info} - R²: {fit_result.stats.r2:.4f}")
                st.pyplot(fig)
                plt.close(fig)
    except Exception as e:
        # If plotting fails, just show a message
        with container:
            st.warning(f"Could not display plot for well {well_name}: {str(e)}")


def _create_peak_curve(x: np.ndarray, peak, lmfit_result) -> Optional[np.ndarray]:
    """Create a curve for a single peak component."""
    try:
        if not hasattr(peak, 'center') or lmfit_result is None:
            return None
        
        # Try to extract peak parameters from lmfit result
        # This is a simplified version - you may need to adjust based on your peak model
        center = peak.center
        amplitude = getattr(peak, 'amplitude', getattr(peak, 'height', 1.0))
        sigma = getattr(peak, 'sigma', getattr(peak, 'fwhm', 10.0) / 2.3548)
        
        # Create Gaussian peak
        peak_curve = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return peak_curve
    except:
        return None


class CurveFittingAgent(BaseAgent):
    """Curve fitting agent using Spectropus-style LLM-assisted multi-peak fitting."""

    def __init__(self, name: str = "Curve Fitting Agent", desc: str = "LLM-assisted multi-peak curve fitting"):
        super().__init__(name, desc)
        self.memory = MemoryManager()
        self.llm_client = None

    def _get_llm_client(self, min_delay_seconds: Optional[float] = None) -> LLMClient:
        """Initialize or return cached LLM client."""
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Get delay from provided value, session state, or default
        delay = min_delay_seconds
        if delay is None:
            try:
                import streamlit as st
                delay = st.session_state.get('gemini_delay_seconds', 0.5)
            except (ImportError, RuntimeError, AttributeError):
                delay = 0.5  # Default in headless mode
        
        # Get API key from environment or session state
        api_key = None
        try:
            import streamlit as st
            api_key = st.session_state.get('api_key')
        except (ImportError, RuntimeError, AttributeError):
            pass  # Not in Streamlit context
        
        # Fallback to environment variables if not in session state
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("API key not found. Set GEMINI_API_KEY environment variable or configure in Settings.")
        
        # Check if we need to recreate the client (if delay changed or client doesn't exist)
        if self.llm_client is None or (hasattr(self.llm_client, 'min_delay') and self.llm_client.min_delay != delay):
            self.llm_client = LLMClient(
                provider="gemini",
                model_id="gemini-2.0-flash-lite",  # Use available model that supports both text and image
                api_key=api_key,
                min_delay_seconds=delay
            )
        return self.llm_client

    def confidence(self, payload: Dict[str, Any]) -> float:
        """Return confidence score for curve fitting tasks."""
        try:
            # High confidence if auto-triggered from watcher with data file
            if payload.get("action") == "curve_fitting" and payload.get("auto_trigger"):
                return 0.95
            
            # High confidence if data file is provided
            if payload.get("data_file") or payload.get("trigger_file"):
                trigger_file = payload.get("data_file") or payload.get("trigger_file", "")
                if trigger_file and trigger_file.lower().endswith(('.csv', '.xlsx', '.xls')):
                    return 0.9
            
            # Curve fitting is generally applicable to spectral data
            return 0.8
        except Exception:
            # Fallback confidence if anything goes wrong
            return 0.5

    def run_agent(self, memory: MemoryManager, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Render UI and handle agent interactions.
        
        If auto-triggered from watcher with a data file, automatically run curve fitting.
        Otherwise, the UI is handled in pages/curve_fitting.py
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Debug logging
        logger.info(f"CurveFittingAgent.run_agent called with payload: {payload is not None}")
        if payload:
            logger.info(f"Payload keys: {list(payload.keys())}")
            logger.info(f"Payload auto_trigger: {payload.get('auto_trigger')}")
            logger.info(f"Payload data_file: {payload.get('data_file')}")
            logger.info(f"Payload trigger_file: {payload.get('trigger_file')}")
            print(f"[DEBUG] CurveFittingAgent.run_agent - payload received: {payload}")
        
        # Check if this is an auto-trigger from watcher
        auto_trigger = False
        data_file = None
        
        # Try to get payload from parameter or memory
        if payload:
            auto_trigger = payload.get("auto_trigger", False)
            data_file = payload.get("data_file") or payload.get("trigger_file")
            logger.info(f"Extracted from payload: auto_trigger={auto_trigger}, data_file={data_file}")
            print(f"[DEBUG] Extracted: auto_trigger={auto_trigger}, data_file={data_file}")
        else:
            # Check memory for recent watcher trigger
            try:
                events = memory.get_latest_history(limit=5)
                for event in events:
                    if isinstance(event, dict) and event.get("type") == "watcher":
                        event_payload = event.get("payload", {})
                        if isinstance(event_payload, dict):
                            trigger_info = event_payload.get("payload", {})
                            if trigger_info.get("auto_trigger") and trigger_info.get("action") == "curve_fitting":
                                auto_trigger = True
                                data_file = trigger_info.get("data_file") or trigger_info.get("trigger_file")
                                break
            except Exception:
                pass
        
        # If auto-triggered with a data file, run curve fitting automatically
        if auto_trigger and data_file:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-triggered curve fitting for: {data_file}")
            print(f"[INFO] Auto-triggered curve fitting detected for: {data_file}")
            
            try:
                from pathlib import Path
                data_path = Path(data_file)
                
                if not data_path.exists():
                    logger.error(f"Data file does not exist: {data_path}")
                    print(f"[ERROR] Data file does not exist: {data_path}")
                    return
                
                # Get inferred parameters from payload if available
                inferred_params = {}
                if payload:
                    inferred_params = payload.get("parameters", {})
                
                logger.info(f"Starting automated curve fitting for {data_path.name}")
                print(f"[INFO] Starting automated curve fitting for {data_path.name}")
                if inferred_params:
                    logger.info(f"Using inferred parameters: {inferred_params}")
                    print(f"[INFO] Using inferred parameters: {inferred_params}")
                
                # Try to find a composition file in the same directory
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
                # Look for common composition file names
                for comp_name in composition_file_names:
                    comp_path = data_dir / comp_name
                    if comp_path.exists():
                        comp_file = str(comp_path)
                        logger.info(f"Found composition file: {comp_file}")
                        print(f"[INFO] Found composition file in data directory: {comp_name}")
                        break
                
                # If no composition file found, use default
                if not comp_file:
                    default_comp_path = Path(__file__).parent.parent / "data" / "2D-3D.csv"
                    if default_comp_path.exists():
                        comp_file = str(default_comp_path)
                        logger.info(f"Using default composition file: {default_comp_path.name}")
                        print(f"[INFO] Using default composition file: {default_comp_path.name}")
                    else:
                        logger.warning(f"Default composition file not found at: {default_comp_path}")
                        print(f"[WARNING] Default composition file not found at: {default_comp_path}")
                
                # Extract parameters with defaults
                max_peaks = inferred_params.get("max_peaks", 4)
                r2_target = inferred_params.get("r2_target", 0.90)
                max_attempts = inferred_params.get("max_attempts", 3)
                reads_to_analyze = inferred_params.get("read_selection", "auto")
                read_type = inferred_params.get("read_type", "em_spectrum")
                wells_to_analyze = inferred_params.get("wells_to_analyze", None)
                # Auto-guess wells from file metadata (Full Plate, A1..A7, 35 slideglass)
                if wells_to_analyze is None:
                    inferred_wells = infer_wells_from_file_metadata(str(data_path))
                    if inferred_wells:
                        wells_to_analyze = inferred_wells
                        logger.info(f"Auto-inferred wells from metadata: {len(wells_to_analyze)} wells")
                        print(f"[INFO] Auto-inferred wells from file metadata: {len(wells_to_analyze)} wells ({wells_to_analyze[0]}..{wells_to_analyze[-1]})")
                api_delay = inferred_params.get("api_delay_seconds", 0.5)
                
                logger.info(f"Parameters: max_peaks={max_peaks}, r2_target={r2_target}, "
                          f"max_attempts={max_attempts}, reads={reads_to_analyze}, read_type={read_type}")
                print(f"[INFO] Parameters: max_peaks={max_peaks}, r2_target={r2_target}, "
                      f"max_attempts={max_attempts}, reads={reads_to_analyze}, read_type={read_type}")
                
                # Run curve fitting with inferred parameters
                print(f"[INFO] Calling run_curve_fitting...")
                logger.info(f"Calling run_curve_fitting with data: {data_path}, comp: {comp_file or 'None'}")
                # Check if this is an auto-triggered run
                is_auto_trigger = payload.get("auto_trigger", False) if payload else False
                
                results = self.run_curve_fitting(
                    data_csv_path=str(data_path),
                    composition_csv_path=comp_file or "",  # Empty string if not found
                    wells_to_analyze=wells_to_analyze,
                    reads_to_analyze=reads_to_analyze,
                    read_type=read_type,
                    max_peaks=max_peaks,
                    r2_target=r2_target,
                    max_attempts=max_attempts,
                    save_plots=True,
                    api_delay_seconds=api_delay,
                    auto_trigger=is_auto_trigger,
                )
                
                logger.info(f"Curve fitting completed for {data_path.name}")
                logger.info(f"Results saved to: {results.get('output_dir', 'results/')}")
                print(f"[INFO] Curve fitting completed for {data_path.name}")
                print(f"[INFO] Results saved to: {results.get('output_dir', 'results/')}")
                
                # Store results in memory
                try:
                    if hasattr(memory, 'session_state') and memory.session_state:
                        memory.session_state.curve_fitting_results = results
                except Exception:
                    pass
                    
            except Exception as e:
                logger.error(f"Error running auto-triggered curve fitting: {e}", exc_info=True)
                print(f"[ERROR] Error running auto-triggered curve fitting: {e}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                return
        else:
            # Debug: log why curve fitting wasn't triggered
            import logging
            logger = logging.getLogger(__name__)
            if not auto_trigger:
                logger.debug(f"Not auto-triggered. Payload: {payload}")
            if not data_file:
                logger.debug(f"No data file. Payload: {payload}")
        
        # Otherwise, UI is handled in pages/curve_fitting.py
        # This method exists to satisfy the abstract base class requirement
        pass

    def run_curve_fitting(
        self,
        data_csv_path: str,
        composition_csv_path: str,
        wells_to_analyze: Optional[List[str]] = None,
        reads_to_analyze: Optional[str] = "auto",
        read_type: str = "em_spectrum",
        max_peaks: int = 4,
        r2_target: float = 0.90,
        max_attempts: int = 3,
        save_plots: bool = True,
        start_wavelength: Optional[int] = None,
        end_wavelength: Optional[int] = None,
        wavelength_step_size: Optional[int] = None,
        api_delay_seconds: Optional[float] = None,
        wells_to_ignore: Optional[List[str]] = None,
        skip_no_peaks: bool = False,  # Skip fitting if LLM detects no substantial peaks
        auto_trigger: bool = False  # Whether this is an auto-triggered run from watcher
    ) -> Dict[str, Any]:
        """
        Run complete curve fitting analysis using Spectropus methodology.

        Args:
            data_csv_path: Path to the spectral data CSV
            composition_csv_path: Path to the composition CSV
            wells_to_analyze: List of wells to analyze (None = all)
            reads_to_analyze: Reads to analyze ("auto", "all", or comma-separated)
            max_peaks: Maximum number of peaks to fit
            r2_target: Target R² value for good fit
            max_attempts: Maximum fitting attempts per well
            save_plots: Whether to save fitting plots
            start_wavelength: Minimum wavelength to include (None = use full range)
            end_wavelength: Maximum wavelength to include (None = use full range)
            wavelength_step_size: Step size for wavelength sampling (None = use all data points)
            api_delay_seconds: Delay between API calls to prevent rate limiting (None = use default 0.5s)

        Returns:
            Dictionary with analysis results
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Validate inputs
            if not os.path.exists(data_csv_path):
                raise FileNotFoundError(f"Data CSV not found: {data_csv_path}")
            if not os.path.exists(composition_csv_path):
                raise FileNotFoundError(f"Composition CSV not found: {composition_csv_path}")

            # If read_type is specified, we need to select all reads first, then filter by type
            # Otherwise use the normal read selection
            if read_type and read_type.lower() != "all":
                # For read type filtering, first select all reads, then filter by type
                config = build_agent_config(
                    data_csv=data_csv_path,
                    composition_csv=composition_csv_path,
                    read_selection="all",  # Select all reads first
                    wells_to_ignore=wells_to_ignore,
                    start_wavelength=start_wavelength,
                    end_wavelength=end_wavelength,
                    wavelength_step_size=wavelength_step_size,
                    fill_na_value=np.nan
                )
            else:
                # Normal read selection
                config = build_agent_config(
                    data_csv=data_csv_path,
                    composition_csv=composition_csv_path,
                    read_selection=reads_to_analyze,
                    wells_to_ignore=wells_to_ignore,
                    start_wavelength=start_wavelength,
                    end_wavelength=end_wavelength,
                    wavelength_step_size=wavelength_step_size,
                    fill_na_value=np.nan
                )

            # Get LLM client with rate limiting delay
            llm = self._get_llm_client(min_delay_seconds=api_delay_seconds)

            # Curate dataset with read type filtering (if needed)
            # This returns the filtered reads that match the requested type
            if read_type and read_type.lower() != "all":
                # Only apply first/last read filtering in auto mode
                # In manual mode, use the user's selection as-is
                read_selection_for_filtering = reads_to_analyze if not auto_trigger else reads_to_analyze
                curated, filtered_reads = self._curate_dataset_with_read_type(config, read_type, read_selection_for_filtering, auto_trigger=auto_trigger)
                # Use the filtered reads for analysis
                actual_reads_to_analyze = filtered_reads
                if hasattr(st, 'info'):
                    st.info(f"📊 Filtered to {len(filtered_reads)} {read_type} reads: {filtered_reads}")
                logger.info(f"Final reads to analyze: {filtered_reads} (from original selection: {reads_to_analyze}, auto_trigger: {auto_trigger})")
                print(f"[INFO] Final reads to analyze: {filtered_reads} (from original selection: {reads_to_analyze}, auto_trigger: {auto_trigger})")
            else:
                from tools.fitting_agent import CurveFitting
                agent = CurveFitting(config)
                curated = agent.curate_dataset()
                actual_reads_to_analyze = reads_to_analyze

            available_wells = curated["wells"]
            
            # Debug: Check what wells were found
            st.info(f"🔍 Debug: Found {len(available_wells)} wells in curated dataset: {available_wells[:10]}{'...' if len(available_wells) > 10 else ''}")
            
            if not available_wells:
                st.error(f"❌ No wells found in curated dataset! Available reads: {curated.get('reads', [])}")
                st.error("This usually means the CSV format isn't being parsed correctly.")
                st.error("Please check that your Excel file has the correct format:")
                st.error("- Column 0: Wavelength values")
                st.error("- Columns 1+: Each column is a read/acquisition")
                st.error("\nDebug info:")
                st.error(f"- Total reads found: {len(curated.get('reads', []))}")
                # Try to get more debug info
                try:
                    from tools.fitting_agent import CurveFitting
                    agent = CurveFitting(config)
                    raw_data, composition = agent.load_csvs(config.data_csv, config.composition_csv)
                    all_blocks = agent.parse_all_reads(raw_data)
                    if all_blocks:
                        first_read = list(all_blocks.keys())[0]
                        first_block = all_blocks[first_read]
                        st.error(f"- First read ({first_read}) columns: {list(first_block.columns)}")
                        st.error(f"- First read shape: {first_block.shape}")
                except Exception as e:
                    st.error(f"- Could not get debug info: {e}")
                st.stop()

            # Filter wells (exclusion takes precedence over inclusion)
            if wells_to_ignore:
                wells_to_process = [w for w in available_wells if w not in wells_to_ignore]
            elif wells_to_analyze:
                wells_to_process = [w for w in wells_to_analyze if w in available_wells]
            else:
                wells_to_process = available_wells

            # Sort wells in standard plate order (A1, A2, ..., A12, B1, B2, ..., H12)
            # For R1, R2 format, sort by number
            wells_to_process = _sort_wells(wells_to_process)

            if not wells_to_process:
                raise ValueError(f"No valid wells found to analyze. Available wells: {available_wells}")

            st.info(f"Starting analysis of {len(wells_to_process)} wells: {wells_to_process}")

            # Analyze each well
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a container for real-time plots
            plot_container = st.container()
            
            for i, well_name in enumerate(wells_to_process):
                status_text.text(f"Analyzing well {well_name} ({i+1}/{len(wells_to_process)})")
                progress_bar.progress((i) / len(wells_to_process))

                try:
                    # Run complete analysis for this well using the FILTERED reads
                    result = run_complete_analysis(
                        config=config,
                        well_name=well_name,
                        llm=llm,
                        reads=actual_reads_to_analyze,
                        max_peaks=max_peaks,
                        skip_no_peaks=skip_no_peaks,
                        r2_target=r2_target,
                        max_attempts=max_attempts,
                        save_plots=save_plots
                    )

                    # Handle both single result and list of results (multiple reads)
                    if isinstance(result, list):
                        for r in result:
                            all_results.append(r)
                            # Display plot for each read result
                            _display_fitting_plot(r, plot_container, well_name)
                    else:
                        all_results.append(result)
                        # Display plot for single result
                        _display_fitting_plot(result, plot_container, well_name)

                except Exception as e:
                    st.error(f"Error analyzing well {well_name}: {str(e)}")
                    import traceback
                    st.exception(e)
                    continue

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            # Save consolidated results
            if all_results:
                # Get base name from data file for naming exports
                data_file_name = os.path.basename(data_csv_path)
                # Remove extension and clean up the name
                base_name = os.path.splitext(data_file_name)[0]
                # Remove any problematic characters for filenames
                base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                base_name = base_name.strip()
                
                # Save comprehensive JSON results
                json_filename = f"results/{base_name}_peak_fit_results.json"
                json_file = save_all_wells_results(all_results, json_filename)

                # Export to CSV
                csv_filename = f"results/{base_name}_peak_fit_export.csv"
                csv_file = export_peak_data_to_csv(all_results, csv_filename)
                
                # Upload CSV to Jupyter server if enabled
                jupyter_upload_status = None
                try:
                    if hasattr(st, 'session_state'):
                        jupyter_config = st.session_state.get("jupyter_config", {})
                        upload_enabled = jupyter_config.get("upload_enabled", False)
                        
                        if upload_enabled:
                            server_url = jupyter_config.get("server_url", "")
                            token = jupyter_config.get("token", "")
                            base_path = jupyter_config.get("notebook_path", "Automated Agent")
                            
                            if server_url:
                                # Read the CSV file content
                                csv_file_path = Path(csv_file) if isinstance(csv_file, str) else csv_file
                                if isinstance(csv_file_path, Path) and csv_file_path.exists():
                                    with open(csv_file_path, 'r', encoding='utf-8') as f:
                                        csv_content = f.read()
                                    
                                    # Extract just the filename (without extension) for folder name
                                    csv_filename_only = csv_file_path.name
                                    csv_filename_base = csv_file_path.stem  # filename without extension
                                    
                                    # Create folder name with filename and date
                                    from datetime import datetime
                                    date_str = datetime.now().strftime("%Y-%m-%d")
                                    # Clean filename for folder name (remove problematic characters)
                                    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', csv_filename_base)
                                    folder_name = f"{safe_filename}_{date_str}"
                                    
                                    # Full path: base_path/folder_name/filename.csv
                                    csv_path = f"{base_path}/{folder_name}"
                                    
                                    # First, create the folder if it doesn't exist
                                    # Jupyter API: create directory by PUTting a directory type
                                    folder_api_url = f"{server_url.rstrip('/')}/api/contents/{csv_path}"
                                    folder_headers = {
                                        "Authorization": f"token {token}" if token else None
                                    }
                                    folder_headers = {k: v for k, v in folder_headers.items() if v is not None}
                                    
                                    try:
                                        # Try to create directory (idempotent - won't fail if exists)
                                        folder_response = requests.put(
                                            folder_api_url,
                                            json={"type": "directory"},
                                            headers=folder_headers,
                                            timeout=10
                                        )
                                        # Directory creation is optional - continue even if it fails
                                        if folder_response.status_code in [200, 201, 409]:  # 409 = already exists
                                            logger.info(f"Created/verified folder: {csv_path}")
                                    except Exception as folder_e:
                                        logger.warning(f"Could not create folder (may already exist): {folder_e}")
                                    
                                    # Upload CSV file to the folder
                                    success, message = self.upload_to_jupyter(
                                        server_url, token, csv_content, csv_filename_only, csv_path
                                    )
                                    jupyter_upload_status = {"success": success, "message": message, "path": csv_path}
                                    if success:
                                        logger.info(f"Successfully uploaded CSV to Jupyter: {message}")
                                        print(f"[INFO] Successfully uploaded CSV to Jupyter: {message}")
                                    else:
                                        logger.warning(f"Failed to upload CSV to Jupyter: {message}")
                                        print(f"[WARNING] Failed to upload CSV to Jupyter: {message}")
                except Exception as e:
                    logger.warning(f"Error uploading CSV to Jupyter: {e}")
                    print(f"[WARNING] Error uploading CSV to Jupyter: {e}")
                    jupyter_upload_status = {"success": False, "message": str(e)}

                return {
                    "success": True,
                    "results": all_results,
                    "files": {
                        "json_results": json_file,
                        "csv_export": csv_file
                    },
                    "jupyter_upload": jupyter_upload_status,
                    "summary": {
                        "total_wells": len(all_results),
                        "successful_fits": len([r for r in all_results if r['fit_result'].success]),
                        "wells_analyzed": wells_to_process
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No wells were successfully analyzed",
                    "results": []
                }

        except Exception as e:
            error_msg = str(e)
            st.error(f"Curve fitting analysis failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "results": []
            }

    def _curate_dataset_with_read_type(self, config: Any, read_type: str, original_read_selection: str = None, auto_trigger: bool = False) -> tuple:
        """
        Curate dataset with read type filtering.

        Args:
            config: Curve fitting configuration
            read_type: Type of reads to select ("em_spectrum" or "absorbance")
            original_read_selection: Original user read selection (e.g., "auto", "1,3")
            auto_trigger: Whether this is an auto-triggered run from watcher

        Returns:
            Tuple of (curated dataset with filtered reads, list of filtered read numbers)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        from tools.fitting_agent import CurveFitting

        # First curate normally (this will have all reads since we set read_selection="all")
        agent = CurveFitting(config)
        curated = agent.curate_dataset()

        # If read_type is specified and not "all", filter the reads
        if read_type and read_type.lower() != "all":
            # Get the raw data to check read types
            data, _ = CurveFitting.load_csvs(config.data_csv, config.composition_csv)

            # Find all reads and their types
            first_col = data.iloc[:, 0].astype(str)
            read_pattern = re.compile(r"^Read\s+(\d+):(.*)$", re.I)

            # Map read numbers to their types
            read_types = {}
            for idx, val in first_col.items():
                m = read_pattern.match(val)
                if m:
                    read_num = int(m.group(1))
                    read_desc = m.group(2).strip().lower()
                    # Determine if this is EM spectrum or absorbance
                    if "em spectrum" in read_desc or "emission" in read_desc or "fluorescence" in read_desc:
                        read_types[read_num] = "em_spectrum"
                    elif "absorbance" in read_desc or "absorption" in read_desc:
                        read_types[read_num] = "absorbance"
                    else:
                        # Unknown type - include by default for now
                        read_types[read_num] = "unknown"

            # Get list of reads that match the desired type
            matching_reads = [read_num for read_num, rtype in read_types.items()
                            if rtype == read_type.lower()]

            # Apply read filtering based on mode
            if auto_trigger:
                # AUTO MODE: Only use first and last read
                if len(matching_reads) >= 2:
                    final_reads = [matching_reads[0], matching_reads[-1]]
                    logger.info(f"Auto mode: filtering to first and last reads from {len(matching_reads)} matching reads: {final_reads}")
                    print(f"[INFO] Auto mode: filtering to first and last reads from {len(matching_reads)} matching reads: {final_reads}")
                else:
                    # Only one read available, use it
                    final_reads = matching_reads
                    logger.info(f"Auto mode: only one read available, using: {final_reads}")
                    print(f"[INFO] Auto mode: only one read available, using: {final_reads}")
            else:
                # MANUAL MODE: Use user's selection or all reads
                if original_read_selection and original_read_selection.lower() not in ["auto", "all"]:
                    # Parse the user's specific selection
                    from tools.fitting_agent import CurveFittingConfig
                    try:
                        desired_reads = CurveFittingConfig._parse_int_list(original_read_selection)
                        logger.info(f"Manual mode: user selected reads {desired_reads} from '{original_read_selection}'")
                        print(f"[INFO] Manual mode: user selected reads {desired_reads} from '{original_read_selection}'")
                        # Only keep reads that are both matching type AND in the desired list
                        final_reads = [r for r in matching_reads if r in desired_reads]
                        logger.info(f"Manual mode: filtered from {len(matching_reads)} matching reads to {len(final_reads)} final reads: {final_reads}")
                        print(f"[INFO] Manual mode: filtered from {len(matching_reads)} matching reads to {len(final_reads)} final reads: {final_reads}")
                    except Exception as e:
                        # If parsing fails, use all matching reads
                        logger.warning(f"Failed to parse read selection '{original_read_selection}': {e}, using all matching reads")
                        print(f"[WARNING] Failed to parse read selection '{original_read_selection}': {e}, using all matching reads")
                        final_reads = matching_reads
                else:
                    # No specific selection in manual mode - use all matching reads
                    final_reads = matching_reads
                    logger.info(f"Manual mode: no specific selection, using all {len(matching_reads)} matching reads")
                    print(f"[INFO] Manual mode: no specific selection, using all {len(matching_reads)} matching reads")

            # Filter blocks based on final read list
            filtered_blocks = {}
            for read_num in final_reads:
                if read_num in curated["blocks"]:
                    filtered_blocks[read_num] = curated["blocks"][read_num]

            # Update curated data with filtered blocks
            curated["blocks"] = filtered_blocks

            # Update wells to only include those present in filtered reads
            if filtered_blocks:
                # Find common wells across filtered reads
                well_sets = []
                for block in filtered_blocks.values():
                    well_cols = [str(c) for c in block.columns if str(c).strip() and str(c).strip().upper() not in ['WAVELENGTH', 'WL']]
                    well_sets.append(set(well_cols))

                if well_sets:
                    common_wells = sorted(set.intersection(*well_sets)) if len(well_sets) > 1 else sorted(well_sets[0])
                    curated["wells"] = common_wells
                else:
                    curated["wells"] = []
            else:
                curated["wells"] = []
                final_reads = []

            # Return both the curated data AND the list of filtered reads
            return curated, final_reads

        # If no read_type filtering, return curated data and all available reads
        return curated, curated.get("reads", [])

    def analyze_single_well(
        self,
        data_csv_path: str,
        composition_csv_path: str,
        well_name: str,
        read: Optional[int] = None,
        max_peaks: int = 4,
        model_kind: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single well with detailed output for interactive use.

        Args:
            data_csv_path: Path to spectral data CSV
            composition_csv_path: Path to composition CSV
            well_name: Name of well to analyze
            read: Specific read number (None = auto)
            max_peaks: Maximum peaks to fit
            model_kind: Peak model type (None = auto-select)

        Returns:
            Analysis results for the single well
        """
        try:
            # Build configuration
            config = build_agent_config(
                data_csv=data_csv_path,
                composition_csv=composition_csv_path,
                read_selection="all",
                wells_to_ignore=[],
                fill_na_value=0.0
            )

            # Get LLM client
            llm = self._get_llm_client()

            # Run analysis
            result = run_complete_analysis(
                config=config,
                well_name=well_name,
                llm=llm,
                reads=read or "auto",
                max_peaks=max_peaks,
                skip_no_peaks=skip_no_peaks,
                model_kind=model_kind,
                save_plots=True
            )

            return {
                "success": True,
                "result": result,
                "well_name": well_name,
                "read": read
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "well_name": well_name,
                "read": read
            }
    
    def upload_to_jupyter(self, server_url, token, file_content, filename, notebook_path):
        """Upload file to Jupyter server using Jupyter API"""
        if requests is None:
            return False, "requests library not available"
        
        try:
            # Clean up URL
            server_url = server_url.rstrip('/')
            if not server_url.startswith('http'):
                server_url = f"http://{server_url}"

            # Construct API endpoint
            api_path = f"{notebook_path}/{filename}"
            api_url = f"{server_url}/api/contents/{api_path}"

            # Prepare headers
            headers = {
                "Authorization": f"token {token}" if token else None
            }
            headers = {k: v for k, v in headers.items() if v is not None}

            # Prepare file content (base64 encoded for binary, plain text for text files)
            if filename.endswith('.csv') or filename.endswith('.py') or filename.endswith('.txt'):
                # Text files
                content_data = {
                    "type": "file",
                    "format": "text",
                    "content": file_content
                }
            else:
                # Binary files (base64 encoded)
                content_data = {
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(
                        file_content.encode() if isinstance(file_content, str) else file_content).decode()
                }

            # Make PUT request to create/update file
            response = requests.put(
                api_url,
                json=content_data,
                headers=headers,
                timeout=10
            )

            if response.status_code in [200, 201]:
                return True, f"Successfully uploaded {filename} to {api_path}"
            else:
                return False, f"Failed to upload: {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"